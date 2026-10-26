"""K real low-rank tasks per client; recovered uploads are separate CPU tensors."""
import copy
from pathlib import Path
import time

import torch

from flcore.clients.clientMultiGroupProj import clientMultiGroupProj
from flcore.clients.clientTargetProj import clientTargetProj
from utils.multi_group_svd import recover_parameters, distance_squared, model_capacity, client_payload


class clientMultiGroupSVD(clientMultiGroupProj):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity_level = self.args.multi_svd_effective_assignment[self.id]
        self.rank_rate = self.args.multi_svd_effective_ratios[self.capacity_level]
        self.full_uploads, self.download_starts, self.capacity_metadata = [], [], []

    @torch.no_grad()
    def set_parameters(self, servers, loop_round):
        if len(servers) != self.K:
            raise ValueError('Incomplete K-model SVD download')
        from utils.cnn512_reporting import model_digest
        models, optimizers, starts, metadata = [], [], [], []
        tick = time.perf_counter()
        for column, server in enumerate(servers):
            # One real legacy decomposition of this column's actual mother.
            model = copy.deepcopy(server)
            model.decom_larger_model(self.rank_rate)
            # Legacy decom_larger_model rebuilds modules but does not set this
            # flag. F and recover_larger_model rely on the ACTUAL local rate.
            model.ratio_LR = self.rank_rate
            model = model.to(self.device)  # Legacy decomposition constructors allocate new CPU modules.
            for name, buffer in self.model_set[column].named_buffers():
                parent, _, leaf = name.rpartition('.')
                module = model.get_submodule(parent) if parent else model
                if leaf in module._buffers:
                    if module._buffers[leaf].shape != buffer.shape:
                        raise ValueError(f'Local buffer shape changed at SVD download: {name}')
                    module._buffers[leaf].copy_(buffer)
                else:
                    module.register_buffer(leaf, buffer.detach().clone())
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                        lr=self.learning_rate, momentum=0., weight_decay=0.)
            # Optional observations restore RNG: their constructor/recovery
            # randomness never changes real download or subsequent SGD streams.
            start = recover_parameters(model) if self.args.multi_svd_diagnostics else None
            full = dict(server.named_parameters())
            models.append(model); optimizers.append(optimizer); starts.append(start)
            metadata.append(dict(capacity_level=self.capacity_level, **model_capacity(model, full)))
        self.model_set, self.optimizer_set = models, optimizers
        self.model = models[0]
        self.download_starts, self.capacity_metadata = starts, metadata
        self.download_servers = servers
        self.download_hashes = [model_digest(m) for m in servers]
        self.full_uploads = [None] * self.K
        self.download_round, self.upload_round = loop_round, None
        from utils.multi_group_kd_options import requested
        if requested(self.args):
            self.kd_download_local_states=None
            if self.args.multi_kd_lambda>0 and self.K>1 and loop_round>0:
                from utils.head_snapshot import tensor_fingerprint
                self.kd_download_local_states=[tensor_fingerprint(m.state_dict()) for m in models]
        self.download_seconds = time.perf_counter() - tick

    def train(self):
        if self.download_round is None or self.upload_round == self.download_round:
            raise RuntimeError('Require one fresh K-column SVD download; duplicate training refused')
        if len(self.model_set) != self.K or len(self.optimizer_set) != self.K:
            raise RuntimeError('Require exactly K low-rank models and K current optimizers')
        from utils.cnn512_reporting import model_digest
        if self.download_hashes != [model_digest(m) for m in self.download_servers]:
            raise RuntimeError('Full downloaded mother changed before local training')
        loader, started, records = self.load_train_data(), time.perf_counter(), []
        from utils.multi_group_kd_options import requested
        kd=None
        if requested(self.args):
            from utils.multi_group_kd import Session
            kd=Session(self,loader)
            loader=kd.indexed_loader(loader)
        for column, (model, optimizer) in enumerate(zip(self.model_set, self.optimizer_set)):
            parameters = [p for p in model.parameters() if p.requires_grad]
            if [id(p) for p in parameters] != [id(p) for g in optimizer.param_groups for p in g['params']]:
                raise RuntimeError('SVD optimizer is bound to stale parameters')
            model.train()
            tick, ce_sum, f_sum, visits, steps = time.perf_counter(), 0., 0., 0, 0
            if kd is not None:raw_sum=kd_sum=total_sum=0.
            for _ in range(self.local_epochs):
                for batch in loader:
                    if kd is not None and kd.active:x,y,indices=batch
                    else:x,y=batch
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    if kd is not None and kd.active and column==kd.assigned:
                        ce,weighted_f,raw,weighted_kd=kd.objective(model,x,y,indices)
                        loss=ce+weighted_f+weighted_kd
                        raw_sum+=float(raw.detach())*len(y);kd_sum+=float(weighted_kd.detach())*len(y)
                    else:
                        ce, weighted_f = clientTargetProj._objective(self, model, x, y)
                        loss = ce + weighted_f
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite heterogeneous multi-group CE/F')
                    loss.backward()
                    cap = self.args.local_grad_clip
                    if cap is not None and cap > 0:
                        torch.nn.utils.clip_grad_norm_(parameters, cap, error_if_nonfinite=True)
                    optimizer.step()
                    ce_sum += ce.item() * len(y); f_sum += float(weighted_f.detach()) * len(y)
                    if kd is not None:total_sum+=float(loss.detach())*len(y)
                    visits += len(y); steps += 1
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
            train_seconds = time.perf_counter() - tick
            # Recover ONCE for upload, with RNG restored and no modification of
            # the saved ordinary low-rank prediction model or its buffers/modes.
            tick = time.perf_counter()
            self.full_uploads[column] = recover_parameters(model)
            full = self.full_uploads[column]
            base = dict(self.download_servers[column].named_parameters())
            # This check applies even when optional distance logging is off.
            if full.keys() != base.keys() or any(full[n].shape != base[n].shape for n in base):
                raise ValueError('Recovered upload is incompatible with the full mother')
            distances = {}
            if self.args.multi_svd_diagnostics:
                start = self.download_starts[column]
                for key, left, right in (('svd_truncation', start, base),
                    ('local_training_recovered', full, start), ('global_delta', full, base)):
                    squared = distance_squared(left, right)
                    distances[key + '_squared_norm'] = squared
                    distances[key + '_norm'] = squared ** .5
            self.download_starts[column] = None  # No per-round history of full anchors.
            torch.save(client_payload(self, column), Path(self.save_folder_name)/f'Client_{self.id}_model_{column}.pt')
            records.append(dict(client_id=self.id, model_id=column, local_epochs=self.local_epochs,
                train_batches=steps, processed_train_samples=visits, ce_loss=ce_sum/visits,
                regularization_loss=f_sum/visits, proximal_loss=0., train_seconds=train_seconds,
                upload_recovery_checkpoint_seconds=time.perf_counter()-tick,
                capacity_level=self.capacity_level, rank_rate=self.rank_rate,
                trainable_parameter_count=self.capacity_metadata[column]['trainable_parameter_count'],
                recovered_parameter_count=self.capacity_metadata[column]['recovered_parameter_count'],
                **(kd.fields(column,raw_sum,kd_sum,total_sum,visits) if kd is not None else {}),
                **distances))
            print(f'[MultiSVDLocal] client={self.id} model={column} rate={self.rank_rate} '
                  f'CE={ce_sum/visits:.8g} weighted_F={f_sum/visits:.8g} steps={steps} visits={visits} '
                  f'distances={distances}')
            if kd is not None:
                print(f'[OnlineKD] client={self.id} model={column} assigned={kd.assigned} '
                      f'mode={records[-1]["kd_effective_mode"]} reason={records[-1]["kd_disabled_reason"]} '
                      f'raw_KL={raw_sum/visits:.8g} weighted_KD={kd_sum/visits:.8g}')
        if kd is not None:kd.finish()
        self.train_stats, self.upload_round = records, self.download_round
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.perf_counter()-started

    def full_upload_parameters(self, column, device=None):
        if self.upload_round != self.download_round or not self.full_uploads or self.full_uploads[column] is None:
            raise RuntimeError('Recovered upload is absent or stale')
        values = self.full_uploads[column]
        return values if device is None else {n: p.to(device) for n, p in values.items()}
