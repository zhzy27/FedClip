"""Standard FedFewS, adapted from pgg3/FedFew 1b29dec (MIT).

Independent native implementation; only the FedClip Client base is inherited.
K optimizers remain bound to K in-memory models across all downloads.
"""
import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_loading import make_data_loader
from utils.data_utils import read_client_data
from flcore.clients.clientbase import Client


class clientFedFewS(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        if kwargs.get('initial_model') is None:
            raise ValueError('FedFewS clients require the common initial CNN, not Model_Distribe')
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.K, self.mu, self.use_rep_mode = args.num_server_models, args.smooth_mu, False
        self.test_batch_size = args.test_batch_size
        self.use_cache = args.use_cache
        self._train_data_cache = self._read(True) if self.use_cache else None
        self._test_data_cache = self._read(False) if self.use_cache else None
        self.model_set = [copy.deepcopy(self.model) for _ in range(self.K)]
        self.optimizer_set = [torch.optim.SGD(m.parameters(), lr=self.learning_rate) for m in self.model_set]
        self.learning_rate_scheduler_set = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=args.learning_rate_decay_gamma)
                                            for o in self.optimizer_set]
        self.learning_rate_decay = False
        self.best_model_idx, self.deployed_model, self.train_model_losses = 0, None, None
        self.deployment_ce = None

    def _read(self, train):
        return read_client_data(self.dataset, self.id, self.args, is_train=train, few_shot=0)

    def load_train_data(self, batch_size=None):
        data = self._train_data_cache if self.use_cache else self._read(True)
        return make_data_loader(data, self.batch_size if batch_size is None else batch_size,
                                shuffle=True, drop_last=False, num_workers=0, loader_cls=DataLoader)

    def load_test_data(self, batch_size=None):
        data = self._test_data_cache if self.use_cache else self._read(False)
        return make_data_loader(data, self.test_batch_size if batch_size is None else batch_size,
                                shuffle=not getattr(self.args, 'cnn512_control', False),
                                drop_last=False, num_workers=0, loader_cls=DataLoader)

    def set_parameters(self, server_model_set):
        if len(server_model_set) != self.K:
            raise ValueError('Incomplete K-server download')
        for server, local in zip(server_model_set, self.model_set):
            if self.args.model_family == 'CNN-512':
                from common.models import copy_full_parameters
                copy_full_parameters(server, local)
                continue
            for new, old in zip(server.parameters(), local.parameters()):
                old.data = new.data.clone()  # Keep Parameter objects/optimizer bindings.

    def _batch(self, x, y):
        return x.to(self.device), y.to(self.device)

    def train(self):
        loader = self.load_train_data()
        started = time.time()
        total_losses = torch.zeros(self.K, device=self.device)
        ce_sum, visits, steps = 0., 0, 0
        for k, model in enumerate(self.model_set):
            model.train()
            optimizer = self.optimizer_set[k]
            for epoch in range(self.local_epochs):
                total_losses[k] = 0.0
                for x, y in loader:
                    x, y = self._batch(x, y)
                    optimizer.zero_grad()
                    loss = self.loss(model(x), y)
                    loss.backward()
                    if getattr(self.args, 'local_grad_clip', None) is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.local_grad_clip)
                    optimizer.step()
                    total_losses[k] += loss.item() * y.shape[0]
                    ce_sum += loss.item() * y.shape[0]
                    visits += y.shape[0]
                    steps += 1
        self.train_model_losses = (total_losses / self.train_samples).cpu().numpy()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - started
        if getattr(self.args, 'cnn512_control', False):
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
            self.last_normal_train_stats = dict(ce_loss=ce_sum/visits, regularization_loss=0., proximal_loss=0.,
                train_seconds=time.time()-started, local_epochs=self.local_epochs,
                processed_train_samples=visits, train_batches=steps)

    def select_best_model(self):
        loader = self.load_train_data()
        for model in self.model_set:
            model.eval()
        total_losses = torch.zeros(self.K, device=self.device)
        count = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = self._batch(x, y)
                for k, model in enumerate(self.model_set):
                    total_losses[k] += self.loss(model(x), y).item() * y.shape[0]
                count += y.shape[0]
        losses = (total_losses / count).cpu().numpy()
        self.best_model_idx = int(np.argmin(losses))
        self.deployed_model = self.model_set[self.best_model_idx]
        self.deployment_ce = losses.copy()
        return self.best_model_idx, losses

    def test_metrics(self):
        if self.deployed_model is None:
            self.select_best_model()
        loader = self.load_test_data()
        self.deployed_model.eval()
        correct, count = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = self._batch(x, y)
                output = self.deployed_model(x)
                correct += torch.sum(torch.argmax(output, dim=1) == y).item()
                count += y.shape[0]
        self.last_test = (correct, count, 0)
        return self.last_test

    def train_metrics(self):
        if self.deployed_model is None:
            self.select_best_model()
        loader = self.load_train_data(batch_size=self.test_batch_size)
        self.deployed_model.eval()
        loss_sum, count = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = self._batch(x, y)
                loss_sum += self.loss(self.deployed_model(x), y).item() * y.shape[0]
                count += y.shape[0]
        self.last_train_metrics = (loss_sum, count)
        return self.last_train_metrics
