"""Low-rank local training: classification plus Frobenius regularization."""

import copy
import random
import time

import numpy as np
import torch

from flcore.clients.clientbase import Client, load_item, save_item
from utils.target_prox_options import effective_prox_mu
from utils.target_prox import make_prox_anchor, proximal_penalty, trainable_parameters
from utils.target_head import local_head, classifier_names, preserved_head_rng


class clientTargetProj(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        if kwargs.get('initial_model') is not None:
            save_item(self.model, self.role, 'model', self.save_folder_name)
            del self.model  # Keep the original sequential checkpoint memory policy.

    def load_test_data(self, batch_size=None):
        if getattr(self.args, 'cnn512_control', False) and batch_size is None:
            batch_size = self.args.test_batch_size
        return super().load_test_data(batch_size)

    def load_train_data(self, batch_size=None):
        if hasattr(self, "_meta_c0_train_data"):
            from torch.utils.data import DataLoader
            return DataLoader(self._meta_c0_train_data, batch_size or self.batch_size,
                              drop_last=False, shuffle=True)
        return super().load_train_data(batch_size)

    def _load_model(self, role=None):
        role = self.role if role is None else role
        model = load_item(role, "model", self.save_folder_name)
        if model is None:
            raise RuntimeError(f"Missing model checkpoint for {role}.")
        return model.to(self.device)

    def _move_batch(self, x, y):
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    def _objective(self, model, x, y):
        ce = self.loss(model(x), y)
        regularization = (
            self.args.regular_lamda * model.frobenius_decay()
            if self.args.is_regular == 1 else ce.new_zeros(())
        )
        return ce, regularization

    def train(self, current_round=0):
        head_lambda = getattr(self.args, 'target_head_l2_lambda', 0.)
        if head_lambda > 0:
            from utils.target_head_l2 import validate_head_l2
            validate_head_l2(self.args)
        if getattr(self.args, "target_local_schedule", "joint") == "fedrep":
            from utils.fedrep_training import train_fedrep
            return train_fedrep(self, current_round)
        if effective_prox_mu(self.args, self.id) > 0:
            return self._train_with_prox(current_round)
        model = self._load_model()
        if head_lambda > 0:
            from utils.target_head_l2 import l2_weight, head_l2_penalty, head_weight_norm
            head_weight = l2_weight(model)
            head_l2_sum, total_sum = 0., 0.
        trainloader = self.load_train_data()
        parameters = [p for p in model.parameters() if p.requires_grad]
        # One learning rate for U, V, classifier and all other trainable parameters.
        optimizer = torch.optim.SGD(parameters, lr=self.learning_rate)
        model.train()
        max_epochs = self.local_epochs
        if self.train_slow:
            max_epochs = np.random.randint(1, max(2, max_epochs // 2))
        start = time.perf_counter()
        ce_sum, reg_sum, samples, batches = 0.0, 0.0, 0, 0
        for _ in range(max_epochs):
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                optimizer.zero_grad()
                ce, regularization = self._objective(model, x, y)
                if head_lambda > 0:
                    weighted_head_l2 = head_l2_penalty(head_weight, head_lambda)
                    total = ce + regularization + weighted_head_l2
                    if not torch.isfinite(total):
                        raise ValueError('Non-finite local-head L2 training loss.')
                    total.backward()
                    head_l2_sum += float(weighted_head_l2.detach()) * y.shape[0]
                    total_sum += float(total.detach()) * y.shape[0]
                else:
                    # Omitted/zero coefficient uses the original backward path.
                    (ce + regularization).backward()
                # Retain the existing local gradient norm cap (unrelated to CLIP alignment).
                from utils.cnn512_control import effective_grad_clip
                clip = effective_grad_clip(self.args)
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, clip)
                optimizer.step()
                ce_sum += ce.item() * y.shape[0]
                reg_sum += float(regularization) * y.shape[0]
                samples += y.shape[0]
                batches += 1
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        save_item(model, self.role, "model", self.save_folder_name)
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += elapsed
        self.last_train_time_cost = elapsed
        self.last_normal_train_stats = dict(ce_loss=ce_sum / max(samples, 1),
            regularization_loss=reg_sum / max(samples, 1), proximal_loss=0., train_seconds=elapsed,
            local_epochs=max_epochs, processed_train_samples=samples, train_batches=batches)
        if head_lambda > 0:
            if not samples:
                raise ValueError('Head L2 training requires nonempty local data.')
            self.last_head_l2_train_stats = dict(round=current_round + 1, loop_round=current_round,
                client_id=self.id, target_head_l2_lambda=head_lambda, train_mean_ce=ce_sum / samples,
                train_mean_weighted_frobenius=reg_sum / samples, train_mean_weighted_head_l2=head_l2_sum / samples,
                train_mean_total_loss=total_sum / samples, head_weight_norm=head_weight_norm(model),
                train_epochs=max_epochs, train_batches=batches, train_sample_visits=samples,
                train_seconds=elapsed, target_local_schedule='joint', optimizer_weight_decay=0.)
            print('[HeadL2Local] ' + ' '.join(f'{k}={v}' for k, v in self.last_head_l2_train_stats.items()))
        label = 'FullLocal' if getattr(self.args, 'model_family', None) == 'CNN-512' else 'LowRankLocal'
        print(f"[{label}] round={current_round} client={self.id} "
              f"lr={self.learning_rate:g} ce_loss={ce_sum / max(samples, 1):.8g} "
              f"regularization_loss={reg_sum / max(samples, 1):.8g} time={elapsed:.3f}s")
        return elapsed

    def _train_with_prox(self, current_round):
        model = self._load_model()
        # The round's actual downloaded low-rank initialization, before any step.
        anchor = make_prox_anchor(model)
        named = trainable_parameters(model)
        parameters = list(named.values())
        mu = effective_prox_mu(self.args, self.id)
        trainloader = self.load_train_data()
        optimizer = torch.optim.SGD(parameters, lr=self.learning_rate)
        model.train()
        max_epochs = self.local_epochs
        if self.train_slow:
            max_epochs = np.random.randint(1, max(2, max_epochs // 2))
        start = time.perf_counter()
        ce_sum, reg_sum, prox_sum, samples, batches = 0., 0., 0., 0, 0
        for _ in range(max_epochs):
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                optimizer.zero_grad()
                ce, regularization = self._objective(model, x, y)
                prox = proximal_penalty(named, anchor, mu)
                total = ce + regularization + prox
                if not torch.isfinite(total):
                    raise ValueError("Non-finite CE/Frobenius/proximal training loss.")
                total.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 10.)
                optimizer.step()
                ce_sum += ce.item() * len(y)
                reg_sum += float(regularization) * len(y)
                prox_sum += prox.item() * len(y)
                samples += len(y)
                batches += 1
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        save_item(model, self.role, "model", self.save_folder_name)
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += elapsed
        self.last_train_time_cost = elapsed
        self.last_normal_train_stats = dict(ce_loss=ce_sum / max(samples, 1),
            regularization_loss=reg_sum / max(samples, 1), proximal_loss=prox_sum / max(samples, 1),
            train_seconds=elapsed, local_epochs=max_epochs, processed_train_samples=samples, train_batches=batches)
        print(f"[LowRankLocalProx] round={current_round} client={self.id} lr={self.learning_rate:g} "
              f"ce_loss={ce_sum / max(samples, 1):.8g} regularization_loss={reg_sum / max(samples, 1):.8g} "
              f"proximal_loss={prox_sum / max(samples, 1):.8g} mu={mu:g} "
              f"scope={getattr(self.args, 'target_prox_scope', 'helpers')} time={elapsed:.3f}s")
        return elapsed

    def build_dwa_guidance(self, current_round, post_local_round):
        """One full training epoch on a private post-local copy; no checkpoint write.

        Return an additional full-W parameter upload for guidance scoring only.
        Preserve every RNG stream even when training/recovery raises an exception.
        """
        if current_round != post_local_round:
            raise ValueError("DWA guidance must start from this round's ordinary post-local model.")
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng():
                model = copy.deepcopy(self._load_model())
                low_rank_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                model.train()
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.SGD(parameters, lr=self.learning_rate)
                ce_sum, reg_sum, samples, batches = 0., 0., 0, 0
                start = time.perf_counter()
                # Consume the complete normal train loader, with its existing batch rules.
                for x, y in self.load_train_data():
                    x, y = self._move_batch(x, y)
                    optimizer.zero_grad()
                    ce, regularization = self._objective(model, x, y)
                    (ce + regularization).backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 10.0)
                    optimizer.step()
                    ce_sum += ce.item() * y.shape[0]
                    reg_sum += float(regularization.detach()) * y.shape[0]
                    samples += y.shape[0]
                    batches += 1
                if samples <= 0:
                    raise RuntimeError("C0 guidance requires a non-empty training epoch.")
                if str(self.device).startswith("cuda"):
                    torch.cuda.synchronize(self.device)
                train_seconds = time.perf_counter() - start
                start = time.perf_counter()
                with torch.no_grad():
                    if any(name.endswith(("conv_v", "weight_v")) for name, _ in model.named_parameters()):
                        model.recover_larger_model()
                    full = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
                if any(not torch.isfinite(value).all() for value in full.values()):
                    raise ValueError("Non-finite DWA guidance parameter.")
                metadata = dict(guidance_epochs=1, guidance_lr=self.learning_rate,
                    guidance_loss_rule=("CrossEntropy + regular_lamda * frobenius_decay"
                                        if self.args.is_regular == 1 else "CrossEntropy"),
                    guidance_regularization_enabled=int(self.args.is_regular == 1),
                    guidance_regularization_lambda=self.args.regular_lamda, guidance_grad_clip_norm=10.,
                    guidance_source_post_local_round=post_local_round, guidance_loop_round=current_round,
                    guidance_train_samples=samples, guidance_train_batches=batches,
                    guidance_ce_loss=ce_sum / samples, guidance_regularization_loss=reg_sum / samples,
                    guidance_extra_train_seconds=train_seconds,
                    guidance_recovery_seconds=time.perf_counter() - start,
                    guidance_extra_upload_bytes=sum(p.numel() * p.element_size() for p in full.values()),
                    guidance_extra_upload_parameters=sum(p.numel() for p in full.values()),
                    ordinary_target_low_rank_parameter_bytes=low_rank_bytes)
                return dict(client_id=self.id, loop_round=current_round, source_post_local_round=post_local_round,
                            parameters=full, metadata=metadata)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

    @torch.no_grad()
    def initialize_local_head(self, global_model):
        """One public head initialization before round zero; never a new random head."""
        model = self._load_model()
        if getattr(model, "_target_local_head_initialized", False):
            raise RuntimeError("A persistent local classifier must be initialized exactly once.")
        names = classifier_names(model)
        classifier_names(global_model)
        source, destination = dict(global_model.named_parameters()), dict(model.named_parameters())
        for name in names:
            destination[name].copy_(source[name])
        model._target_local_head_initialized = True
        save_item(model, self.role, "model", self.save_folder_name)

    @torch.no_grad()
    def _download_local_backbone(self, model, global_model):
        if not getattr(model, "_target_local_head_initialized", False):
            raise RuntimeError("Local head was not initialized from the public server classifier before round zero.")
        excluded = classifier_names(model)
        classifier_names(global_model)
        global_model.decom_larger_model(model.ratio_LR)
        source, destination = dict(global_model.named_parameters()), dict(model.named_parameters())
        if source.keys() != destination.keys():
            raise RuntimeError("Downloaded low-rank parameter names do not match the client.")
        for name, parameter in destination.items():
            if source[name].shape != parameter.shape:
                raise RuntimeError(f"Downloaded parameter shape mismatch: {name}")
            if name not in excluded:
                parameter.copy_(source[name])
        return model

    @torch.no_grad()
    def set_parameters(self):
        model = self._load_model()
        global_model = self._load_model("Server")
        if getattr(self.args, 'model_family', None) == 'CNN-512':
            from common.models import copy_full_parameters
            copy_full_parameters(global_model, model)
            save_item(model, self.role, 'model', self.save_folder_name)
            return
        if local_head(getattr(self, "args", None)):
            self._download_local_backbone(model, global_model)
            save_item(model, self.role, "model", self.save_folder_name)
            return
        global_model.decom_larger_model(model.ratio_LR)
        source = dict(global_model.named_parameters())
        destination = dict(model.named_parameters())
        if source.keys() != destination.keys():
            raise RuntimeError("Downloaded low-rank parameter names do not match the client.")
        for name, parameter in destination.items():
            if source[name].shape != parameter.shape:
                raise RuntimeError(f"Downloaded parameter shape mismatch: {name}")
            parameter.copy_(source[name])
        # Keep local buffers, following the existing low-rank download convention.
        save_item(model, self.role, "model", self.save_folder_name)

    @torch.no_grad()
    def test_metrics(self):
        model = self._load_model()
        model.eval()
        correct, samples = 0, 0
        for x, y in self.load_test_data():
            x, y = self._move_batch(x, y)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            samples += y.shape[0]
        return correct, samples, 0

    @torch.no_grad()
    def train_metrics(self):
        model = self._load_model()
        model.eval()
        losses, samples = 0.0, 0
        for x, y in self.load_train_data():
            x, y = self._move_batch(x, y)
            ce, regularization = self._objective(model, x, y)
            losses += (ce + regularization).item() * y.shape[0]
            samples += y.shape[0]
        return losses, samples

    @torch.no_grad()
    def test_post_local(self):
        """Observe the saved local model without writing it or advancing RNG.

        The inherited test loader uses shuffle=False. Restore every module's mode,
        including mixed train/eval states, even when inference raises an exception.
        """
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng():
                model = self._load_model()
                states = [(module, module.training) for module in model.modules()]
                try:
                    model.eval()
                    correct, samples = 0, 0
                    for x, y in self.load_test_data():
                        x, y = self._move_batch(x, y)
                        correct += (model(x).argmax(dim=1) == y).sum().item()
                        samples += y.shape[0]
                    return correct, samples, 0
                finally:
                    for module, training in states:
                        module.training = training
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

    def test_downloaded_global(self):
        """Evaluate the current server model through the normal low-rank download.

        Restore the local checkpoint and RNG streams even if evaluation fails.
        This extra observation must not change the next round's optimization,
        local BatchNorm buffers, or the inherited local-model accuracy logs.
        """
        if local_head(getattr(self, "args", None)):
            # Read-only composition: no checkpoint writes, even on failure.
            with preserved_head_rng(), torch.no_grad():
                model = self._download_local_backbone(self._load_model(), self._load_model("Server"))
                model.eval()
                correct, samples = 0, 0
                for x, y in self.load_test_data():
                    x, y = self._move_batch(x, y)
                    correct += (model(x).argmax(dim=1) == y).sum().item()
                    samples += y.shape[0]
                return correct, samples, 0
        local_model = load_item(self.role, "model", self.save_folder_name)
        if local_model is None:
            raise RuntimeError(f"Missing local checkpoint for {self.role}.")
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        try:
            with torch.random.fork_rng():
                self.set_parameters()
                return self.test_metrics()
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            save_item(local_model, self.role, "model", self.save_folder_name)
