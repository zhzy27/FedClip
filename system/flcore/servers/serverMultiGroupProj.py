"""Fixed K full models, fixed training-update-derived groups, independent columns."""
import copy
import json
from pathlib import Path
import time

import torch

from common.client_data import audit_image_clients
from common.experiment import rng_state, sha256, write_json
from common.models import initial_full_model
from flcore.clients.clientMultiGroupProj import clientMultiGroupProj
from flcore.servers.serverbase import Server
from utils.cnn512_control import evaluation_due, seed_control
from utils.cnn512_reporting import model_digest
from utils.group_projection import aggregate_group_updates
from utils.multi_group_clustering import (update_similarity, balanced_cluster, grouping_context,
                                         digest_json, validate_group_file)
from utils.multi_group_reporting import record_evaluation, save_multi_reports, write_multi_manifest, model_geometry


class FedMultiGroupProj(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.K = args.num_server_models
        self.data_manifest = audit_image_clients(args)
        self.args.control_effective_config['actual_data_alpha'] = self.data_manifest['actual_alpha']
        write_json(Path(self.report_dir())/'data_manifest.json', self.data_manifest)
        (Path(self.report_dir())/'data_config.json').write_bytes(Path(self.data_manifest['config_path']).read_bytes())
        initial = initial_full_model(args).to(self.device)
        self.initial_hash = model_digest(initial)
        self.initial_schema = {name: list(p.shape) for name, p in initial.state_dict().items()}
        self.global_model_set = [initial] + [copy.deepcopy(initial) for _ in range(self.K-1)]
        self.global_model = initial
        self.set_slow_clients()
        self.set_clients(clientMultiGroupProj)
        if any(model_digest(m) != self.initial_hash for c in self.clients for m in c.model_set):
            raise RuntimeError('Common K model initialization mismatch')
        self.context = grouping_context(args, self.data_manifest, self.initial_hash)
        self.groups, self.group_payload = None, None
        if args.multi_group_file:
            self.group_payload = json.loads(Path(args.multi_group_file).read_text(encoding='utf-8'))
            self.groups = validate_group_file(self.group_payload, self.context)
            self.group_source_sha256 = sha256(args.multi_group_file)
        self.training_records, self.primary_records, self.endpoint_records = [], [], []
        self.group_records, self.aggregation_records, self.weight_records, self.geometry_records = [], [], [], []
        self.loop_round, self.Budget = -1, []
        self.round_records = []
        seed_control(args)  # AFTER base Client's historical manual_seed(0).
        self.training_start_rng = rng_state()
        write_multi_manifest(self)
        print(f'FedMultiGroupProj: N={self.num_clients}, K={self.K}; sizes={args.multi_capacities}; '
              f'{args.multi_group_mode}; beta={args.multi_group_weight}, tau={args.group_softmax_tau}')
        print(f'CE only; full CNN including classifier; SGD lr={self.learning_rate}, momentum=0, weight_decay=0; '
              f'clip={args.local_grad_clip}; each client K x epochs={self.K*self.local_epochs} data epochs/round')
        print('Buffers are client-local; only named_parameters uploaded/aggregated. Fixed membership chooses deployment, never test/CE selection.')

    def client_init_kwargs(self, client_id):
        return {'initial_model': self.global_model}

    def _initialize_groups(self):
        if self.loop_round != 0:
            raise RuntimeError('Groups must be fixed at first completed training round')
        similarity, diagnostics = update_similarity(self.global_model_set, self.clients)
        if self.groups is None:
            self.groups, algorithm = balanced_cluster(similarity, self.args.multi_capacities,
                seed=self.args.multi_cluster_seed, restarts=self.args.multi_cluster_restarts,
                iterations=self.args.multi_cluster_iterations)
            self.group_payload = dict(schema_version=1, groups=self.groups, context=self.context,
                context_sha256=digest_json(self.context), similarity_matrix=similarity.tolist(), algorithm=algorithm,
                first_loop_round=0, completed_train_count=1, diagnostics=diagnostics,
                client_to_model={str(cid): k for k, g in enumerate(self.groups) for cid in g},
                group_fingerprint=digest_json(dict(context=self.context, groups=self.groups)))
        else:
            original = self.group_payload['diagnostics']['first_upload_delta_sha256']
            # The supplied artifact fixes membership, including across devices.
            # Audit current uploads, but never recluster/choose a new group when
            # GPU floating point differs from the source run's first uploads.
            write_json(Path(self.report_dir())/'group_reuse_verification.json', dict(
                grouping_source_sha256=self.group_source_sha256,
                data_initialization_protocol_match=True,
                first_upload_bitwise_match=original == diagnostics['first_upload_delta_sha256'],
                similarity_bitwise_match=self.group_payload['similarity_matrix'] == similarity.tolist(),
                membership_reused_unchanged=True))
        validate_group_file(self.group_payload, self.context)
        write_json(Path(self.report_dir())/'fixed_groups.json', self.group_payload)
        write_json(Path(self.report_dir())/'group_similarity.json', dict(matrix=similarity, diagnostics=diagnostics))
        print(f'FIXED groups={self.groups}; fingerprint={self.group_payload["group_fingerprint"]}')

    def aggregate_parameters(self):
        # All references stay unchanged until ALL K kernels have completed.
        if self.groups is None or any(c.upload_round != self.loop_round or len(c.model_set) != self.K for c in self.clients):
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
                    yield cid, samples[cid], dict(self.clients[cid].model_set[column].named_parameters())
            result, metrics, rows, matrices = aggregate_group_updates(dict(server.named_parameters()), uploads,
                samples, self.groups[column], self.args.multi_group_mode,
                self.args.multi_group_weight if self.args.multi_group_mode == 'group_projection_softmax' else None,
                softmax_tau=self.args.group_softmax_tau)
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
            timing = dict(loop_round=self.loop_round, completed_train_count=self.loop_round+1, model_id=column,
                          aggregation_seconds=time.perf_counter()-tick)
            self.aggregation_records.append(dict(**timing, **metrics, matrices=matrices))
            self.weight_records.extend(dict(**timing, **row) for row in rows)
            next_parameters.append(result)
            diagnostics.append(dict(metrics=metrics, rows=rows, matrices=matrices))
            print(f'Model {column}, G={self.groups[column]}: {metrics}')
        with torch.no_grad():
            for server, result in zip(self.global_model_set, next_parameters):
                for name, parameter in server.named_parameters():
                    parameter.copy_(result[name])
        self.last_aggregation = diagnostics
        return diagnostics

    def train(self):
        for loop in range(self.global_rounds+1):
            self.loop_round = loop
            started = time.perf_counter()
            print(f'\n{"="*76}\nFedMultiGroupProj loop_round={loop}, completed_train_count={loop+1}; N={self.num_clients} x K={self.K}\n{"="*76}')
            self.selected_clients = self.select_clients()
            for client in self.clients:
                client.set_parameters(self.global_model_set, loop)
            evaluated = evaluation_due(self.args, loop)
            before = [r for c in self.clients for r in c.observe_models()] if evaluated and not self.args.multi_no_pre_eval else None
            for client in self.selected_clients:
                client.train()
                self.training_records.extend(dict(loop_round=loop, completed_train_count=loop+1, **r) for r in client.train_stats)
                print(f'Client {client.id} trained {self.K} models x {self.local_epochs} epochs; '
                      f'visits={sum(r["processed_train_samples"] for r in client.train_stats)}; '
                      f'batches={sum(r["train_batches"] for r in client.train_stats)}')
            if loop == 0:
                self._initialize_groups()
            if evaluated:
                after = [r for c in self.clients for r in c.observe_models()]
                record_evaluation(self, before, after)
            self.aggregate_parameters()
            model_geometry(self)
            self.Budget.append(time.perf_counter()-started)
            records = [r for r in self.training_records if r['loop_round'] == loop]
            round_record = dict(loop_round=loop, completed_train_count=loop+1, evaluated=evaluated,
                round_seconds=self.Budget[-1], training_tasks=len(records),
                actual_sample_visits=sum(r['processed_train_samples'] for r in records),
                actual_optimizer_steps=sum(r['train_batches'] for r in records))
            self.round_records.append(round_record)
            # Durable, lightweight per-round records; no model tensors in logs.
            from common.experiment import jsonable
            with (Path(self.report_dir())/'rounds.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(jsonable(dict(budget=round_record, training=records,
                    primary=self.primary_records[-1] if evaluated else None,
                    aggregation=[r for r in self.aggregation_records if r['loop_round']==loop])), allow_nan=False)+'\n')
        self.save_global_model()
        self.save_results()

    def save_global_model(self):
        root = Path(self.save_folder_name)
        metadata = dict(model_family=self.args.model_family, model_class='common.models.CNN_512',
                        model_kwargs=dict(in_channels=3, n_kernels=16, out_dim=self.num_classes, input_size=32),
                        completed_train_count=self.loop_round+1, schema=self.initial_schema,
                        initial_parameter_sha256=self.initial_hash,
                        group_fingerprint=self.group_payload['group_fingerprint'])
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
