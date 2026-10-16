"""Explicit DataLoader construction, with call-site class injection for observers."""
from torch.utils.data import DataLoader


def make_data_loader(dataset, batch_size, *, shuffle, drop_last, num_workers, loader_cls=DataLoader, **kwargs):
    """No generator, seed reset, sampling, iteration or caching is added here."""
    return loader_cls(dataset, batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, **kwargs)


def loader_description(batch_size, *, shuffle, drop_last=False, num_workers=0):
    """Read-only description of the defaults explicitly used by these adapters."""
    return dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
                generator=None, pin_memory=False, persistent_workers=False,
                sampling_rng="global Torch RNG, no additional Generator")
