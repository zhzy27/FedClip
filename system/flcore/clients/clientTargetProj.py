"""Reuse FedCLIP local optimization without changing any training rule."""

import random

import numpy as np
import torch

from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientCLIP import clientCLIP


class clientTargetProj(clientCLIP):
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
