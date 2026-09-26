"""Low-rank local training: classification plus Frobenius regularization."""

import random
import time

import numpy as np
import torch

from flcore.clients.clientbase import Client, load_item, save_item


class clientTargetProj(Client):
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
