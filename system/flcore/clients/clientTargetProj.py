"""Low-rank local training: classification plus Frobenius regularization."""

import copy
import random
import time

import numpy as np
import torch

from flcore.clients.clientbase import Client, load_item, save_item


class clientTargetProj(Client):
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
        model = self._load_model()
        trainloader = self.load_train_data()
        parameters = [p for p in model.parameters() if p.requires_grad]
        # One learning rate for U, V, classifier and all other trainable parameters.
        optimizer = torch.optim.SGD(parameters, lr=self.learning_rate)
        model.train()
        max_epochs = self.local_epochs
        if self.train_slow:
            max_epochs = np.random.randint(1, max(2, max_epochs // 2))
        start = time.perf_counter()
        ce_sum, reg_sum, samples = 0.0, 0.0, 0
        for _ in range(max_epochs):
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                optimizer.zero_grad()
                ce, regularization = self._objective(model, x, y)
                (ce + regularization).backward()
                # Retain the existing local gradient norm cap (unrelated to CLIP alignment).
                torch.nn.utils.clip_grad_norm_(parameters, 10.0)
                optimizer.step()
                ce_sum += ce.item() * y.shape[0]
                reg_sum += float(regularization) * y.shape[0]
                samples += y.shape[0]
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        save_item(model, self.role, "model", self.save_folder_name)
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += elapsed
        self.last_train_time_cost = elapsed
        print(f"[LowRankLocal] round={current_round} client={self.id} "
              f"lr={self.learning_rate:g} ce_loss={ce_sum / max(samples, 1):.8g} "
              f"regularization_loss={reg_sum / max(samples, 1):.8g} time={elapsed:.3f}s")
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
    def set_parameters(self):
        model = self._load_model()
        global_model = self._load_model("Server")
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
