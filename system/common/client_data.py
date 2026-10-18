"""Explicit existing image shards, shared by baseline readers and data audits."""
import json
from pathlib import Path

import torch

from .data_io import select_client_shard, read_shard, decode_image_data, NUMPY_STACK_LIST, TENSOR_LIST
from .experiment import sha256, tensor_digest

FORMATS = {'npz_only': ('.npz',), 'pkl_first': ('.pkl', '.npz'), 'pkl_only': ('.pkl',)}


def data_directory(args):
    if not getattr(args, 'data_root', None):
        raise ValueError('Explicit data_root is required')
    dataset = Path(args.dataset)
    suffix = getattr(args, 'data_subdir', None)
    if suffix:
        if len(dataset.parts) != 1:
            raise ValueError('Supply dataset suffix either in -data or --data-subdir, not both')
        dataset /= suffix
    if dataset.is_absolute() or '..' in dataset.parts:
        raise ValueError('Dataset/subdirectory must be relative within the explicit data root')
    return Path(args.data_root).expanduser().resolve() / dataset


def image_policy(args):
    # The old reader remains untouched unless the caller explicitly opts in.
    few = args.algorithm == 'FedFewS'
    formats = getattr(args, 'data_formats', None) or ('pkl_first' if few else 'npz_only')
    if formats not in FORMATS:
        raise ValueError(f'Invalid shard policy: {formats}')
    return FORMATS[formats], NUMPY_STACK_LIST if few and not getattr(args, 'cnn512_control', False) else TENSOR_LIST


def client_file(args, client_id, train):
    formats, _ = image_policy(args)
    return select_client_shard(data_directory(args) / ('train' if train else 'test'), client_id, formats=formats)


def read_image_client(args, client_id, train=True):
    _, mode = image_policy(args)
    return decode_image_data(read_shard(client_file(args, client_id, train)), list_mode=mode)


def audit_image_clients(args):
    directory = data_directory(args)
    path = directory / 'config.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    comparison = getattr(args, 'cnn512_control', False)
    if comparison:
        if 'num_clients' not in config or config.get('partition') not in ('pat', 'dir'):
            raise ValueError('Comparison needs config.json num_clients and supported partition metadata; no directory-name inference')
        if config.get('num_classes') != args.num_classes:
            raise ValueError('Data config num_classes differs from the supported model')
    if config.get('num_clients', args.num_clients) != args.num_clients:
        raise ValueError('Data config num_clients differs from run')
    expected_alpha = getattr(args, 'data_expected_alpha', None)
    if expected_alpha is not None and config.get('alpha') != expected_alpha:
        raise ValueError(f"Actual config.json alpha={config.get('alpha')!r} != {expected_alpha}; directory name is not evidence")
    partition = getattr(args, 'data_expected_partition', None)
    if partition is not None and config.get('partition') != partition:
        raise ValueError(f"Actual partition={config.get('partition')!r} != {partition!r}")
    records = []
    for cid in range(args.num_clients):
        for train in [True, False]:
            selected = client_file(args, cid, train)
            data = read_image_client(args, cid, train)
            if not data:
                raise ValueError(f'Empty shard: {selected}')
            x, y = torch.stack([p[0] for p in data]), torch.stack([p[1] for p in data])
            if x.shape[1:] != (3, 32, 32) or x.dtype != torch.float32 or y.dtype != torch.int64:
                raise ValueError(f'Invalid CNN image/label shape/dtype: {selected}')
            if not torch.isfinite(x).all() or y.min() < 0 or y.max() >= args.num_classes:
                raise ValueError(f'Invalid images/labels: {selected}')
            labels, counts = y.unique(return_counts=True)
            records.append(dict(client_id=cid, split='train' if train else 'test', path=str(selected.resolve()),
                                format=selected.suffix, sha256=sha256(selected), count=len(data),
                                labels=dict(zip(labels.tolist(), counts.tolist())), x_dtype=str(x.dtype), y_dtype=str(y.dtype),
                                decoded_x_sha256=tensor_digest(x), decoded_y_sha256=tensor_digest(y),
                                npz_shadowed_by_pkl=selected.suffix == '.pkl' and selected.with_suffix('.npz').exists()))
    formats, mode = image_policy(args)
    result = dict(directory=str(directory), config_path=str(path), config_sha256=sha256(path), config=config,
                actual_alpha=config.get('alpha'), format_policy=formats, image_list_mode=mode, files=records)
    if comparison:
        result['partition_verification'] = verify_partition(args, config, records)
    return result


def verify_partition(args, config, records):
    """Audit actual labels without inferring a generator protocol from a folder."""
    partition = config['partition']
    if partition != getattr(args, 'data_expected_partition', None):
        raise ValueError('Actual config partition differs from the requested comparison protocol')
    if partition == 'dir':
        return dict(partition='dir', cpc_applicable=False, actual_alpha=config.get('alpha'),
                    cpc_check='not applicable: Dirichlet does not assign a fixed number of labels/client')
    requested = args.class_per_client
    declared = config.get('class_per_client')
    if declared is not None and declared != requested:
        raise ValueError(f'pat class_per_client metadata={declared} differs from -cpc {requested}')
    observed = {}
    counts = {}
    for cid in range(args.num_clients):
        combined = {}
        for row in records:
            if row['client_id'] == cid:
                for label, count in row['labels'].items():
                    combined[int(label)] = combined.get(int(label), 0) + count
        observed[cid] = len(combined)
        counts[cid] = combined
    if any(n > requested for n in observed.values()):
        raise ValueError(f'Observed train+test labels exceed pat -cpc {requested}: {observed}')
    allocation = config.get('Size of samples for labels in clients')
    if allocation is not None:
        if len(allocation) != args.num_clients:
            raise ValueError('Data allocation statistics client count differs')
        for cid in range(args.num_clients):
            expected = {int(label): int(count) for label, count in allocation[cid] if count > 0}
            if expected != counts[cid]:
                raise ValueError(f'Client {cid} actual train+test label counts differ from config allocation statistics')
    complete = all(n == requested for n in observed.values())
    return dict(partition='pat', cpc_applicable=True, requested_cpc=requested, declared_cpc=declared,
                observed_label_counts=observed, allocation_statistics_checked=allocation is not None,
                verification_level='observed_class_counts_match' if complete else 'partial',
                limitation=None if complete else 'Some observed counts are below cpc; absent allocation metadata cannot prove unobserved assigned labels. No partition-name inference.')
