"""Native standard FedFewS. Source: pgg3/FedFew 1b29dec, MIT.

Reuse FedClip's client construction, selection, directories and save facilities.
K-model training/dual weighting are independent of the compatibility runtime.
"""
import copy
from pathlib import Path
import random
import time

import numpy as np
import torch
from scipy.special import logsumexp

from common.client_data import audit_image_clients
from common.experiment import rng_state, write_json
from common.models import initial_full_model, checked_parameters
from utils.cnn512_control import seed_control, population_control, effective_grad_clip
from utils.cnn512_reporting import initialize_control, record_control, record_control_training, save_control
from flcore.clients.clientbase import save_item
from flcore.clients.clientfedfews import clientFedFewS
from flcore.servers.serverbase import Server
from utils.fedfews_reporting import round_record, save_fedfews_results, write_manifest


class FedFewS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.K, self.mu, self.use_rep_mode = args.num_server_models, args.smooth_mu, False
        self.data_manifest = audit_image_clients(args)
        write_json(Path(self.report_dir()) / 'data_manifest.json', self.data_manifest)
        (Path(self.report_dir()) / 'data_config.json').write_bytes(Path(self.data_manifest['config_path']).read_bytes())
        save_item(rng_state(), 'FedFewS', 'rng_before_initialization', self.save_folder_name)
        self.global_model = initial_full_model(args).to(self.device)
        self.global_model_set = [copy.deepcopy(self.global_model) for _ in range(self.K)]
        self.set_slow_clients()
        self.set_clients(clientFedFewS)
        self.Budget, self.rows, self.evaluations = [], [], []
        self.rs_model_losses, self.rs_alpha_weights, self.rs_soft_weights = [], [], []
        self.rs_client_test_acc, self.rs_client_train_loss, self.rs_client_test_auc, self.rs_client_ids = [], [], [], []
        self.loop_round = -1
        initialize_control(self, self.global_model, (m for c in self.clients for m in c.model_set))
        if getattr(args, 'cnn512_control', False):
            seed_control(args)
        write_manifest(self)
        print(f'FedFewS native: K={self.K}; CE only; SGD lr={self.learning_rate}, momentum=0, weight_decay=0; '
              f'gradient_norm_cap={effective_grad_clip(args)}; no F/Prox/SVD')
        print(f'Per-client budget: K({self.K}) x local_epochs({self.local_epochs}) = {self.K * self.local_epochs} data epochs/round')
        print(f'Loaders: train/selection/train_metrics shuffle=True; test shuffle={not getattr(args, "cnn512_control", False)}; '
              f'test batch={args.test_batch_size}; drop_last=False; workers=0; global Torch RNG')

    def client_init_kwargs(self, client_id):
        return {'initial_model': self.global_model}

    def result_file_prefix(self):
        # Dataset suffix belongs in structured directories, never in a filename.
        return f'FedFewS_{self._safe_path_component(self.goal)}_{self.times}'

    def send_models(self):
        for client in self.clients:
            started = time.time()
            client.set_parameters(self.global_model_set)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - started)

    def receive_models(self):
        active = random.sample(self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        self.uploaded_ids, self.uploaded_weights, self.uploaded_models = [], [], []
        total = 0
        for client in active:
            cost = (client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] +
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds'])
            if cost <= self.time_threthold:
                total += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model_set)
        if len(self.uploaded_ids) != self.num_clients:
            raise RuntimeError('Standard FedFewS requires all client uploads; time/drop filtering is unsupported')
        for i, weight in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = weight / total

    def _compute_alpha_weights(self, all_loss_values):
        # Identical float64 NumPy/SciPy evaluation order to the accepted source.
        logits = -all_loss_values / self.mu
        log_S_i = logsumexp(logits, axis=1, keepdims=True)
        log_w_ik = logits - log_S_i
        w_ik = np.exp(log_w_ik)
        log_sum_inv_S = logsumexp(-log_S_i)
        alpha_weights = np.exp(-log_S_i - log_sum_inv_S)
        return alpha_weights, w_ik

    def aggregate_parameters(self):
        losses = np.zeros((len(self.uploaded_ids), self.K))
        for row, cid in enumerate(self.uploaded_ids):
            losses[row] = self.clients[cid].train_model_losses
        losses = losses * np.array(self.uploaded_weights)[:, np.newaxis]
        alpha, w = self._compute_alpha_weights(losses)
        self.last_alpha, self.last_w = alpha, w
        if not np.isfinite(alpha).all() or not np.isfinite(w).all():
            raise FloatingPointError('Nonfinite FedFewS dual weights')
        # All differences use OLD servers. Preserve receive order and per-tensor
        # add_ arithmetic, rather than recasting as a normalized model average.
        differences = []
        for models in self.uploaded_models:
            model_diffs = []
            for k in range(self.K):
                diff = copy.deepcopy(self.global_model_set[k])
                if getattr(getattr(self, 'args', None), 'model_family', None) == 'CNN-512':
                    src, dst = checked_parameters(models[k], diff)
                    for name, old in dst.items():
                        old.data = src[name].data - old.data
                    model_diffs.append(diff)
                    continue
                for old, uploaded in zip(diff.parameters(), models[k].parameters()):
                    old.data = uploaded.data - old.data
                model_diffs.append(diff)
            differences.append(model_diffs)
        client_weights = {}
        for row, cid in enumerate(self.uploaded_ids):
            for k in range(self.K):
                weight = (alpha[row] * w[row][k])[0]
                client_weights[cid] = (alpha[row], w[row])
                if getattr(getattr(self, 'args', None), 'model_family', None) == 'CNN-512':
                    src, dst = checked_parameters(differences[row][k], self.global_model_set[k])
                    for name, server in dst.items():
                        server.data.add_(src[name].data, alpha=weight)
                    continue
                for server, diff in zip(self.global_model_set[k].parameters(), differences[row][k].parameters()):
                    server.data.add_(diff.data, alpha=weight)
        return ([c.train_model_losses.tolist() for c in self.clients],
                [client_weights[c.id][0] for c in self.clients],
                [client_weights[c.id][1].tolist() for c in self.clients])

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()      # ALL tests first; then ALL train_metrics.
        training = self.train_metrics()  # Both consume original global shuffle RNG.
        accuracy = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(training[2]) * 1.0 / sum(training[1])
        self.rs_test_acc.append(accuracy)
        self.rs_train_loss.append(train_loss)
        self.rs_client_test_acc.append([a/n for a, n in zip(stats[2], stats[1])])
        self.rs_client_test_auc.append([a/n for a, n in zip(stats[3], stats[1])])
        self.rs_client_train_loss.append([a/n for a, n in zip(training[2], training[1])])
        self.rs_client_ids.append(stats[0])
        print(f'FedFewS selected post-local weighted test accuracy: {100 * accuracy:.6f}%')

    def train(self):
        for loop in range(self.global_rounds + 1):
            self.loop_round = loop
            started = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            print(f'\n{"=" * 70}\nFedFewS loop_round={loop}; completed_train_count={loop+1}\n{"=" * 70}')
            for client in self.selected_clients:
                client.train()
                print(f'Client {client.id}: aggregation last-epoch CE={client.train_model_losses.tolist()}')
            record_control_training(self, loop)
            evaluated = loop % self.eval_gap == 0 or loop == self.global_rounds
            if evaluated:
                for client in self.clients:
                    idx, losses = client.select_best_model()
                    print(f'Client {client.id}: deployment endpoint CE={losses.tolist()}; selected={idx}')
                self.evaluate()
                if getattr(self.args, 'cnn512_control', False):
                    correct, samples, _ = self.clients[self.args.target_client_id].last_test
                    counts = ([dict(client_id=c.id, correct=c.last_test[0], test_samples=c.last_test[1])
                               for c in self.clients] if population_control(self.args) else None)
                    record_control(self, loop, correct, samples, client_counts=counts)
            self.receive_models()
            losses, alphas, weights = self.aggregate_parameters()
            self.rs_model_losses.append(losses)
            self.rs_alpha_weights.append(alphas)
            self.rs_soft_weights.append(weights)
            round_record(self, evaluated)
            self.Budget.append(time.time() - started)
            print(f'alpha={np.array(alphas).squeeze().tolist()}; server received mass={(self.last_alpha * self.last_w).sum(axis=0).tolist()}')
            print(f'Round cost={self.Budget[-1]:.3f}s')
        self.save_global_model()
        self.save_results()

    def save_global_model(self):
        # Single bundle includes every ACTUAL trained model, optimizer and RNG.
        state = dict(server_models=[m.state_dict() for m in self.global_model_set],
                     clients=[dict(client_id=c.id, best_model_idx=c.best_model_idx, deployed_model_is_post_local=True,
                                   models=[m.state_dict() for m in c.model_set],
                                   optimizers=[o.state_dict() for o in c.optimizer_set],
                                   schedulers=[s.state_dict() for s in c.learning_rate_scheduler_set]) for c in self.clients],
                     schema={n: list(p.shape) for n, p in self.global_model.state_dict().items()},
                     model=f'{type(self.global_model).__module__}.{type(self.global_model).__name__}',
                     model_family=self.args.model_family, num_classes=self.num_classes,
                     completed_train_count=self.loop_round + 1, rng=rng_state())
        save_item(state, 'FedFewS', 'final', self.save_folder_name)

    def save_results(self):
        super().save_results()  # Structured H5 path, final-model export and mirror.
        save_fedfews_results(self)
        save_control(self)
