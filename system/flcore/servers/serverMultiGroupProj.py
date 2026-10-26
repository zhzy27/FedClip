"""Fixed K full models, fixed training-update-derived groups, independent columns."""
import copy
import json
from pathlib import Path
import time

import torch

from common.client_data import audit_image_clients
from common.experiment import rng_state, write_json
from flcore.clients.clientMultiGroupProj import clientMultiGroupProj
from flcore.servers.serverbase import Server
from utils.cnn512_control import evaluation_due, seed_control
from utils.cnn512_reporting import model_digest
from utils.group_projection import aggregate_group_updates
from utils.multi_group_clustering import (balanced_cluster, grouping_context,
                                         digest_json, validate_group_file)
from utils.multi_group_reporting import (record_evaluation, record_warmup_evaluation, save_multi_reports,
                                        write_multi_manifest, model_geometry)
from utils.multi_group_protocol import (GroupEvidence, grouping_protocol, compatibility_default, phase_for_loop,
                                       read_group_file, atomic_publish_group)
from utils import multi_group_svd as svd


class FedMultiGroupProj(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.K = args.num_server_models
        self.data_manifest = audit_image_clients(args)
        self.args.control_effective_config['actual_data_alpha'] = self.data_manifest['actual_alpha']
        write_json(Path(self.report_dir())/'data_manifest.json', self.data_manifest)
        (Path(self.report_dir())/'data_config.json').write_bytes(Path(self.data_manifest['config_path']).read_bytes())
        initial = svd.initial_model(args).to(self.device)
        self.initial_hash = model_digest(initial)
        self.initial_schema = {name: list(p.shape) for name, p in initial.state_dict().items()}
        self.global_model_set = [initial] + [copy.deepcopy(initial) for _ in range(self.K-1)]
        self.global_model = initial
        self.set_slow_clients()
        if svd.enabled(args):
            from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
            self.set_clients(clientMultiGroupSVD)
        else:
            self.set_clients(clientMultiGroupProj)
        if any(model_digest(m) != self.initial_hash for c in self.clients for m in c.model_set):
            raise RuntimeError('Common K model initialization mismatch')
        self.context = grouping_context(args, self.data_manifest, self.initial_hash)
        self.groups, self.group_payload = None, None
        self.group_protocol = grouping_protocol(args)
        self.group_evidence = GroupEvidence(args)
        # Fail early as before unless the user explicitly enables a bounded wait.
        # Validation here never activates membership or skips warmup.
        if args.multi_group_file and (Path(args.multi_group_file).exists() or not args.multi_group_wait_seconds):
            read_group_file(args.multi_group_file, validate=lambda p: validate_group_file(p, self.context))
        self.training_records, self.primary_records, self.endpoint_records = [], [], []
        from utils.multi_group_kd_options import requested,protocol
        if requested(args):self.kd_training_records,self.kd_probe_records=[],[]
        if svd.enabled(args):
            self.capacity_records = []
        self.warmup_records = []
        self.group_records, self.aggregation_records, self.weight_records, self.geometry_records = [], [], [], []
        self.loop_round, self.Budget = -1, []
        self.round_records = []
        seed_control(args)  # AFTER base Client's historical manual_seed(0).
        self.training_start_rng = rng_state()
        write_multi_manifest(self)
        print(f'FedMultiGroupProj: N={self.num_clients}, K={self.K}; sizes={args.multi_capacities}; '
              f'{args.multi_group_mode}; beta={args.multi_group_weight}, tau={args.group_softmax_tau}')
        if svd.enabled(args):
            print(f'MultiSVD initialization={svd.initialization_protocol(args)}; initial_sha256={self.initial_hash}')
        objective = ('CE + original Frobenius; heterogeneous SVD CNN including classifier' if svd.enabled(args) and args.is_regular else
                     'CE; heterogeneous SVD CNN including classifier' if svd.enabled(args) else 'CE only; full CNN including classifier')
        print(f'{objective}; SGD lr={self.learning_rate}, momentum=0, weight_decay=0; '
              f'clip={args.local_grad_clip}; each client K x epochs={self.K*self.local_epochs} data epochs/round')
        print('Buffers are client-local; only named_parameters uploaded/aggregated. Fixed membership chooses deployment, never test/CE selection.')
        print(f'Grouping protocol={self.group_protocol}; total budget={self.global_rounds+1}, '
              f'warmup={args.multi_warmup_rounds}, grouped={self.global_rounds+1-args.multi_warmup_rounds}')
        if requested(args):print(f'Online cross-group KD: {protocol(args)}')

    def client_init_kwargs(self, client_id):
        return {'initial_model': self.global_model}

    def _initialize_groups(self):
        if self.loop_round != self.group_protocol['formation_loop_round'] or self.groups is not None:
            raise RuntimeError('Groups may be formed only once, at the configured completed-training boundary')
        similarity, evidence = self.group_evidence.finalize()
        diagnostics = (evidence['rounds'][0]['diagnostics'] if len(evidence['rounds'])==1 else
                       dict(basis='delta_history', evidence_sha256=evidence['evidence_sha256']))
        if not self.args.multi_group_file:
            self.groups, algorithm = balanced_cluster(similarity, self.args.multi_capacities,
                seed=self.args.multi_cluster_seed, restarts=self.args.multi_cluster_restarts,
                iterations=self.args.multi_cluster_iterations)
            self.group_payload = dict(schema_version=1 if compatibility_default(self.args) else 2, groups=self.groups, context=self.context,
                context_sha256=digest_json(self.context), similarity_matrix=similarity.tolist(), algorithm=algorithm,
                first_loop_round=self.loop_round, completed_train_count=self.loop_round+1, diagnostics=diagnostics,
                grouping_protocol=self.group_protocol, evidence=evidence,
                client_to_model={str(cid): k for k, g in enumerate(self.groups) for cid in g},
                group_fingerprint=digest_json(dict(context=self.context, groups=self.groups)))
        else:
            self.group_payload, self.group_source_sha256, waited = read_group_file(self.args.multi_group_file,
                wait_seconds=self.args.multi_group_wait_seconds, poll_seconds=self.args.multi_group_poll_seconds,
                validate=lambda p: validate_group_file(p, self.context))
            self.groups = validate_group_file(self.group_payload, self.context)
            original = self.group_payload.get('evidence', {}).get('evidence_sha256')
            # The supplied artifact fixes membership, including across devices.
            # Audit current uploads, but never recluster/choose a new group when
            # GPU floating point differs from the source run's first uploads.
            self.group_reuse_receipt = dict(
                grouping_source_sha256=self.group_source_sha256,
                data_initialization_protocol_match=True,
                evidence_bitwise_match=original == evidence['evidence_sha256'] if original is not None else None,
                first_upload_bitwise_match=(self.group_payload['diagnostics'].get('first_upload_delta_sha256') ==
                    diagnostics.get('first_upload_delta_sha256')) if compatibility_default(self.args) else None,
                waited_seconds=waited, grouping_protocol=self.group_protocol,
                similarity_bitwise_match=self.group_payload['similarity_matrix'] == similarity.tolist(),
                membership_reused_unchanged=True)
            write_json(Path(self.report_dir())/'group_reuse_verification.json', self.group_reuse_receipt)
        validate_group_file(self.group_payload, self.context)
        write_json(Path(self.report_dir())/'fixed_groups.json', self.group_payload)
        write_json(Path(self.report_dir())/'group_similarity.json', dict(matrix=similarity, diagnostics=diagnostics, evidence=evidence))
        write_json(Path(self.report_dir())/'group_history.json', evidence)
        if self.args.multi_group_publish:
            atomic_publish_group(self.args.multi_group_publish, self.group_payload)
        print(f'FIXED groups={self.groups}; fingerprint={self.group_payload["group_fingerprint"]}')

    def aggregate_parameters(self):
        # All references stay unchanged until ALL K kernels have completed.
        phase = phase_for_loop(self.args, self.loop_round)
        if (phase == 'grouped' and self.groups is None) or any(c.upload_round != self.loop_round or len(c.model_set) != self.K for c in self.clients):
            raise RuntimeError('Require current-round N x K complete uploads and fixed groups')
        total = sum(c.train_samples for c in self.clients)
        samples = {c.id: c.train_samples/total for c in self.clients}
        order = [c.id for c in self.selected_clients]
        if sorted(order) != list(range(self.num_clients)):
            raise RuntimeError('All clients must upload once')
        self.uploaded_ids = order
        next_parameters, diagnostics = [], []
        for column, server in enumerate(self.global_model_set):
            tick = time.perf_counter()
            def uploads():
                for cid in order:
                    client = self.clients[cid]
                    post = (client.full_upload_parameters(column, self.device) if svd.enabled(self.args) else
                            dict(client.model_set[column].named_parameters()))
                    yield cid, samples[cid], post
            result, metrics, rows, matrices = aggregate_group_updates(dict(server.named_parameters()), uploads,
                samples, list(range(self.num_clients)) if phase=='warmup' else self.groups[column],
                'group_avg_all' if phase=='warmup' else self.args.multi_group_mode,
                self.args.multi_group_weight if phase=='grouped' and self.args.multi_group_mode == 'group_projection_softmax' else None,
                softmax_tau=self.args.group_softmax_tau)
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
            timing = dict(loop_round=self.loop_round, completed_train_count=self.loop_round+1, model_id=column,
                          phase=phase, aggregation_mode='group_avg_all' if phase=='warmup' else self.args.multi_group_mode,
                          aggregation_seconds=time.perf_counter()-tick)
            self.aggregation_records.append(dict(**timing, **metrics, matrices=matrices))
            self.weight_records.extend(dict(**timing, **row) for row in rows)
            next_parameters.append(result)
            diagnostics.append(dict(metrics=metrics, rows=rows, matrices=matrices))
            print(f'Model {column}, phase={phase}, G={None if phase=="warmup" else self.groups[column]}: {metrics}')
        with torch.no_grad():
            for server, result in zip(self.global_model_set, next_parameters):
                for name, parameter in server.named_parameters():
                    parameter.copy_(result[name])
        self.last_aggregation = diagnostics
        return diagnostics

    def train(self):
        for loop in range(self.global_rounds+1):
            self.loop_round = loop
            phase = phase_for_loop(self.args, loop)
            started = time.perf_counter()
            print(f'\n{"="*76}\nFedMultiGroupProj phase={phase}, loop_round={loop}, completed_train_count={loop+1}; N={self.num_clients} x K={self.K}\n{"="*76}')
            self.selected_clients = self.select_clients()
            for client in self.clients:
                client.set_parameters(self.global_model_set, loop)
                from utils.multi_group_kd_options import requested
                if requested(self.args):
                    from utils.multi_group_kd import bind_assignment
                    bind_assignment(client,self)
            if svd.enabled(self.args) and loop == 0:
                write_json(Path(self.report_dir())/'capacity_manifest.json', dict(protocol=svd.capacity_protocol(self.args),
                    initialization=svd.initialization_protocol(self.args),
                    clients={str(c.id): c.capacity_metadata for c in self.clients}))
            evaluated = evaluation_due(self.args, loop)
            before = [r for c in self.clients for r in c.observe_models()] if evaluated and not self.args.multi_no_pre_eval else None
            for client in self.selected_clients:
                client.train()
                self.training_records.extend(dict(loop_round=loop, completed_train_count=loop+1, phase=phase, **r) for r in client.train_stats)
                if requested(self.args):
                    from utils.multi_group_kd import persist_client
                    persist_client(self,client)
                print(f'Client {client.id} trained {self.K} models x {self.local_epochs} epochs; '
                      f'visits={sum(r["processed_train_samples"] for r in client.train_stats)}; '
                      f'batches={sum(r["train_batches"] for r in client.train_stats)}')
            # Always BEFORE aggregation, independent of eval_gap; servers still
            # equal the exact parameters downloaded at this round's start.
            self.group_evidence.collect(loop, self.global_model_set, self.clients)
            if phase=='grouped' and loop == self.group_protocol['formation_loop_round']:
                self._initialize_groups()
            if evaluated:
                after = [r for c in self.clients for r in c.observe_models()]
                if phase == 'warmup':
                    record_warmup_evaluation(self, before, after)
                else:
                    record_evaluation(self, before, after)
            self.aggregate_parameters()
            if phase=='warmup' and loop == self.group_protocol['formation_loop_round']:
                # The last warmup round ALSO finishes its all-client Avg.
                # Evidence was captured against the pre-Avg download above.
                # Waiting/publication therefore never skips warmup aggregation.
                self._initialize_groups()
            model_geometry(self)
            self.Budget.append(time.perf_counter()-started)
            records = [r for r in self.training_records if r['loop_round'] == loop]
            round_record = dict(loop_round=loop, completed_train_count=loop+1, evaluated=evaluated,
                phase=phase, actual_warmup_train_count=min(loop+1, self.args.multi_warmup_rounds),
                actual_grouped_train_count=max(0,loop+1-self.args.multi_warmup_rounds),
                round_seconds=self.Budget[-1], training_tasks=len(records),
                actual_sample_visits=sum(r['processed_train_samples'] for r in records),
                actual_optimizer_steps=sum(r['train_batches'] for r in records))
            self.round_records.append(round_record)
            # Durable, lightweight per-round records; no model tensors in logs.
            from common.experiment import jsonable
            with (Path(self.report_dir())/'rounds.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(jsonable(dict(budget=round_record, training=records,
                    primary=self.primary_records[-1] if evaluated and phase=='grouped' else None,
                    warmup_observations=[r for r in self.warmup_records if r['loop_round']==loop],
                    aggregation=[r for r in self.aggregation_records if r['loop_round']==loop])), allow_nan=False)+'\n')
        self.save_global_model()
        self.save_results()

    def save_global_model(self):
        if svd.enabled(self.args):
            return svd.save_models(self)
        root = Path(self.save_folder_name)
        metadata = dict(model_family=self.args.model_family, model_class='common.models.CNN_512',
                        model_kwargs=dict(in_channels=3, n_kernels=16, out_dim=self.num_classes, input_size=32),
                        completed_train_count=self.loop_round+1, schema=self.initial_schema,
                        initial_parameter_sha256=self.initial_hash,
                        group_fingerprint=self.group_payload['group_fingerprint'])
        metadata['grouping_protocol'] = self.group_protocol
        for column, model in enumerate(self.global_model_set):
            torch.save(dict(**metadata, state_dict=model.state_dict(), model_id=column,
                            role='post_aggregation_server'), root/f'Server_model_{column}.pt')
        for client in self.clients:
            for column, (model, optimizer) in enumerate(zip(client.model_set, client.optimizer_set)):
                torch.save(dict(**metadata, state_dict=model.state_dict(), model_id=column, client_id=client.id,
                    assigned_model_id=int(self.group_payload['client_to_model'][str(client.id)]),
                    role='ordinary_post_local', optimizer=optimizer.state_dict()), root/f'Client_{client.id}_model_{column}.pt')
        torch.save(dict(rng=rng_state(), training_start_rng=self.training_start_rng), root/'MultiGroup_rng.pt')
        write_json(root/'fixed_groups.json', self.group_payload)

    def save_results(self):
        save_multi_reports(self)
        self.export_final_models()
        target = Path(self.final_model_dir())
        write_json(target/'fixed_groups.json', self.group_payload)
        write_multi_manifest(self, target/'manifest.json')
