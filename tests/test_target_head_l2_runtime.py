"""Frozen 9a9e4c1 zero regression and 20-client/two-round low-rank joint smoke."""

from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'system'))
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverTargetProj import FedTargetProj
from utils.head_snapshot import tensor_fingerprint, file_hash
from utils import target_head_l2
from test_meta_projection import rng_state


class HeadL2RuntimeTests(unittest.TestCase):
    def assert_same_rng(self, a, b):
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1][0], b[1][0]); np.testing.assert_array_equal(a[1][1], b[1][1])
        self.assertEqual(a[1][2:], b[1][2:])
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        self.assertEqual(len(a[3]), len(b[3]))
        for x, y in zip(a[3], b[3]): torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_frozen_zero_trajectory_and_nonzero_observation_isolation_exports(self):
        torch.set_num_threads(2)
        old_client, old_server = {}, {}
        for relative, namespace in (('clients/clientTargetProj.py', old_client), ('servers/serverTargetProj.py', old_server)):
            source = subprocess.check_output(['git', 'show', '9a9e4c1:system/flcore/' + relative], cwd=ROOT, text=True)
            exec(compile(source, '9a9e4c1/' + relative, 'exec'), namespace)
        old_server['clientTargetProj'] = old_client['clientTargetProj']
        factory = 'Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)'
        generator = torch.Generator().manual_seed(79)
        x = torch.randn(18, 3, 32, 32, generator=generator)
        # Unequal batch sizes test sample-visit weighting; unique pixel markers trace actual batches.
        x[:, 0, 0, 0] = torch.arange(18)
        old_cwd, trajectories, common_head = os.getcwd(), {}, None
        context = dict(training=False, rows=[])
        class TracedLoader(DataLoader):
            def __iter__(self):
                for images, labels in super().__iter__():
                    if context['training']:
                        context['rows'].append((context['cid'], tuple(images[:, 0, 0, 0].tolist())))
                    yield images, labels
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                os.chdir(root)
                def reader(dataset, cid, args, is_train=True, **kwargs):
                    count = (17 if cid % 2 else 18) if is_train else 3 + cid % 3
                    return [(x[i], torch.tensor((cid + i) % 100)) for i in range(count)]
                for label in ('legacy', 'omitted', 'zero', 'on', 'off'):
                    tag = 'avg_local_head_' + label
                    args = SimpleNamespace(device='cpu', dataset='Synthetic', num_classes=100, global_rounds=1,
                        local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                        random_join_ratio=False, few_shot=0, algorithm='FedTargetProj', time_select=False,
                        goal='smoke', time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                        client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                        model_family='Decom_CNN-5-512', models_folder_name='', niid=1, partition='pat', class_per_client=20,
                        exp_name=tag, save_file_paths=[], models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                        global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                        is_regular=1, regular_lamda=.001, target_proj_mode='avg', target_prox_mu=0.,
                        target_head_mode='local', target_local_schedule='joint', save_folder_name=str(root / 'c' / tag),
                        final_model_root=str(root / 'f' / tag), h5_result_root=str(root / 'h' / tag),
                        training_log_root=str(root / 'l' / tag), training_log_dir=str(root / 'l' / tag))
                    if label in ('zero', 'on', 'off'):
                        args.target_head_l2_lambda = 0. if label == 'zero' else .001
                        args.target_head_l2_diagnostics = int(label != 'off')
                    cls = old_server['FedTargetProj'] if label == 'legacy' else FedTargetProj
                    context.update(training=False, rows=[])
                    with redirect_stdout(io.StringIO()), patch('flcore.clients.clientbase.read_client_data', side_effect=reader), \
                            patch('flcore.servers.serverbase.read_client_data', side_effect=reader), \
                            patch('flcore.clients.clientbase.DataLoader', TracedLoader):
                        server = cls(args, 0)
                        initial_model = load_item('Server', 'model', server.save_folder_name)
                        if common_head is None:
                            common_head = {n: p.detach().clone() for n, p in initial_model.named_parameters() if n.startswith('fc3.')}
                        calls, records, previous_heads, client_records = [0] * 20, [], {}, []
                        for client in server.clients:
                            model = load_item(client.role, 'model', client.save_folder_name)
                            for n, p in model.named_parameters():
                                if n.startswith('fc3.'): torch.testing.assert_close(p, common_head[n], rtol=0, atol=0)
                            model.register_buffer('probe', torch.tensor([client.id + 1.]))
                            save_item(model, client.role, 'model', client.save_folder_name)
                            native_download = client.set_parameters
                            def download(client=client, native_download=native_download):
                                native_download()
                                model = load_item(client.role, 'model', client.save_folder_name)
                                for n, p in model.named_parameters():
                                    if n.startswith('fc3.'):
                                        torch.testing.assert_close(p, previous_heads.get(client.id, common_head)[n], rtol=0, atol=0)
                            client.set_parameters = download
                            native_train = client.train
                            def train(current_round, client=client, native_train=native_train):
                                calls[client.id] += 1
                                before = load_item(client.role, 'model', client.save_folder_name)
                                context.update(training=True, cid=client.id)
                                try: native_train(current_round)
                                finally: context['training'] = False
                                model = load_item(client.role, 'model', client.save_folder_name)
                                previous_heads[client.id] = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith('fc3.')}
                                self.assertFalse(torch.equal(before.fc3.weight, model.fc3.weight))
                                self.assertFalse(torch.equal(before.conv1.weight, model.conv1.weight))
                                torch.testing.assert_close(model.probe, torch.tensor([client.id + 1.]), rtol=0, atol=0)
                                self.assertEqual(client.last_normal_train_stats['train_batches'], 10)
                                self.assertEqual(client.last_normal_train_stats['local_epochs'], 5)
                                client_records.append((tensor_fingerprint(model.state_dict()), rng_state()))
                            client.train = train
                        native_aggregate = server.aggregate_parameters_avg
                        def aggregate():
                            uploads = [tensor_fingerprint(load_item(c.role, 'model', c.save_folder_name).state_dict()) for c in server.clients]
                            native_aggregate()
                            records.append(dict(uploads=uploads, server=tensor_fingerprint(load_item('Server', 'model', server.save_folder_name).state_dict()),
                                batches=list(context['rows']), rng=rng_state(), order=list(server.uploaded_ids)))
                        server.aggregate_parameters_avg = aggregate
                        native_observe = target_head_l2.observe_target
                        def observation(s):
                            if label != 'on': return native_observe(s)
                            paths = list(Path(s.save_folder_name).glob('*.pt'))
                            hashes, rng = [file_hash(p) for p in paths], rng_state()
                            native_observe(s)
                            self.assertEqual([file_hash(p) for p in paths], hashes)
                            self.assert_same_rng(rng_state(), rng)
                        with patch.object(target_head_l2, 'observe_target', side_effect=observation): server.train()
                        self.assertEqual(calls, [2] * 20)
                        trajectories[label] = (records, client_records)
                        if label != 'on': continue
                        logs = Path(server.report_dir())
                        for suffix, count in (('clients', 40), ('target', 2)):
                            with (logs / f'target_head_l2_{suffix}.csv').open() as stream:
                                rows = list(csv.DictReader(stream))
                            history = json.loads((logs / f'target_head_l2_{suffix}.json').read_text())['history']
                            self.assertEqual(len(rows), count); self.assertEqual(len(history), count)
                            for csv_row, json_row in zip(rows, history):
                                for key, value in json_row.items():
                                    if isinstance(value, (float, int)): self.assertEqual(float(csv_row[key]), value)
                        clients = json.loads((logs / 'target_head_l2_clients.json').read_text())
                        self.assertEqual(sorted({r['client_id'] for r in clients['history']}), list(range(20)))
                        for row in clients['history']:
                            self.assertGreater(row['train_mean_weighted_head_l2'], 0.)
                            self.assertGreater(row['train_mean_weighted_frobenius'], 0.)
                            self.assertEqual(row['train_sample_visits'], (17 if row['client_id'] % 2 else 18) * 5)
                            self.assertAlmostEqual(row['train_mean_total_loss'], row['train_mean_ce'] + row['train_mean_weighted_frobenius'] + row['train_mean_weighted_head_l2'], places=5)
                        targets = json.loads((logs / 'target_head_l2_target.json').read_text())
                        self.assertEqual(targets['primary_summary'], server._local_accuracy_summary())
                        for row, primary in zip(targets['history'], server.target_proj_history):
                            self.assertEqual(row['endpoint_test_accuracy'], primary['target_post_local_acc'])
                        with h5py.File(args.save_file_paths[-1]) as h5:
                            h = h5['target_projection/target_head_l2']
                            self.assertEqual(h['clients/train_mean_weighted_head_l2'].shape, (40,))
                            np.testing.assert_array_equal(h['target/endpoint_test_accuracy'][:], [r['endpoint_test_accuracy'] for r in targets['history']])
                            np.testing.assert_array_equal(h['clients/train_mean_weighted_head_l2'][:], [r['train_mean_weighted_head_l2'] for r in clients['history']])
                        final = Path(server.final_model_dir())
                        manifest = json.loads((final / 'manifest.json').read_text())
                        self.assertEqual(manifest['target_head_l2']['policy']['target_head_l2_lambda'], .001)
                        self.assertEqual(len(manifest['target_head_l2']['final_round_training']), 20)
                        for artifact in ('Client_0_model.pt', 'C0_aggregated_backbone_local_head.pt'):
                            model = torch.load(final / artifact, weights_only=False)
                            model.eval()
                            self.assertEqual(tuple(model(x[:2]).shape), (2, 100))
                        backbone = torch.load(final / 'Server_backbone_only.pt', weights_only=True)
                        self.assertFalse(any(n.startswith('fc3.') for n in backbone['parameters']))
                for reference, candidate in (('legacy', 'omitted'), ('legacy', 'zero'), ('on', 'off')):
                    a, b = trajectories[reference], trajectories[candidate]
                    for old, new in zip(a[0], b[0]):
                        for key in ('uploads', 'server', 'batches', 'order'): self.assertEqual(old[key], new[key])
                        self.assert_same_rng(old['rng'], new['rng'])
                    for old, new in zip(a[1], b[1]):
                        self.assertEqual(old[0], new[0]); self.assert_same_rng(old[1], new[1])
            finally:
                os.chdir(old_cwd)


if __name__ == '__main__': unittest.main()
