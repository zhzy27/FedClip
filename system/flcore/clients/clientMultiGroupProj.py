"""K independent full-model SGD tasks; deployment is fixed membership, not CE."""
import copy
import time

import torch
from torch.utils.data import DataLoader

from common.data_loading import make_data_loader
from common.experiment import rng_state, set_rng
from common.models import copy_full_parameters
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class clientMultiGroupProj(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        if kwargs.get('initial_model') is None:
            raise ValueError('Multi-group clients require the common full initial model')
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.K = args.num_server_models
        self.model_set = [self.model] + [copy.deepcopy(self.model) for _ in range(self.K-1)]
        self.optimizer_set = [torch.optim.SGD(m.parameters(), lr=self.learning_rate, momentum=0., weight_decay=0.)
                              for m in self.model_set]
        self.train_stats = []
        self.download_round = None
        self.upload_round = None

    def load_train_data(self, batch_size=None):
        data = read_client_data(self.dataset, self.id, self.args, is_train=True, few_shot=0)
        return make_data_loader(data, self.batch_size if batch_size is None else batch_size,
                                shuffle=True, drop_last=False, num_workers=0, loader_cls=DataLoader)

    def load_test_data(self, batch_size=None):
        data = read_client_data(self.dataset, self.id, self.args, is_train=False, few_shot=0)
        return make_data_loader(data, self.args.test_batch_size if batch_size is None else batch_size,
                                shuffle=False, drop_last=False, num_workers=0, loader_cls=DataLoader)

    def set_parameters(self, servers, loop_round):
        if len(servers) != self.K:
            raise ValueError('Incomplete K-model download')
        for server, local in zip(servers, self.model_set):
            copy_full_parameters(server, local)  # Does not replace Parameter/optimizer bindings.
        self.download_round, self.upload_round = loop_round, None

    def train(self):
        if self.download_round is None or self.upload_round == self.download_round:
            raise RuntimeError('Require one fresh download per client train call; duplicate training refused')
        loader, started, records = self.load_train_data(), time.perf_counter(), []
        for column, (model, optimizer) in enumerate(zip(self.model_set, self.optimizer_set)):
            model.train()
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
            tick, ce_sum, visits, steps = time.perf_counter(), 0., 0, 0
            for _ in range(self.local_epochs):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    loss = self.loss(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite multi-group local CE')
                    loss.backward()
                    cap = self.args.local_grad_clip
                    if cap is not None and cap > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cap, error_if_nonfinite=True)
                    optimizer.step()
                    ce_sum += loss.item()*len(y)
                    visits += len(y)
                    steps += 1
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
            records.append(dict(client_id=self.id, model_id=column, local_epochs=self.local_epochs,
                train_batches=steps, processed_train_samples=visits, ce_loss=ce_sum/visits,
                regularization_loss=0., proximal_loss=0., train_seconds=time.perf_counter()-tick))
        self.train_stats, self.upload_round = records, self.download_round
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.perf_counter()-started

    @torch.no_grad()
    def observe_models(self):
        """True endpoint tests; restore all parameters/buffers/modes and RNG even on error."""
        random_state, records = rng_state(), []
        try:
            loader = self.load_test_data()
            for column, model in enumerate(self.model_set):
                state = {n: p.detach().clone() for n, p in model.state_dict().items()}
                modes = [(module, module.training) for module in model.modules()]
                try:
                    model.eval()
                    correct, samples = 0, 0
                    for x, y in loader:
                        logits = model(x.to(self.device))
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError('Nonfinite endpoint output')
                        correct += int((logits.argmax(dim=1) == y.to(self.device)).sum())
                        samples += len(y)
                    if samples != self.test_samples or not samples:
                        raise ValueError('Endpoint test sample count differs from audited client count')
                    records.append(dict(client_id=self.id, model_id=column, correct=correct,
                                        test_samples=samples, accuracy=correct/samples))
                finally:
                    model.load_state_dict(state, strict=True)
                    for module, training in modes:
                        module.training = training
            return records
        finally:
            set_rng(random_state)
