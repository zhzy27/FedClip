"""Actual main.py entry, original-width heterogeneous CNN, temporary synthetic NPZ only."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'system'))


class HeadL2EntryTests(unittest.TestCase):
    def test_main_two_round_twenty_client_original_width_CNN(self):
        with tempfile.TemporaryDirectory(prefix='hl2_') as directory:
            root = Path(directory)
            system = root / 's'; system.mkdir()
            rng = np.random.default_rng(17)
            for split, samples in (('train', 2), ('test', 3)):
                folder = root / 'dataset/Cifar100/pat_20' / split
                folder.mkdir(parents=True)
                for cid in range(20):
                    np.savez(folder / f'{cid}.npz', data=dict(x=rng.normal(size=(samples, 3, 32, 32)).astype('float32'),
                        y=np.arange(samples, dtype='int64') + cid))
            tag = 'avg_local_head_head_l2_0p001'
            command = [sys.executable, '-u', str(ROOT / 'system/main.py'), '-algo', 'FedTargetProj',
                '-data', 'Cifar100', '-m', 'Decom_CNN-5-512', '-dev', 'cpu', '-nc', '20', '-ncl', '100',
                '-t', '1', '-lr', '.005', '-jr', '1.', '-lbs', '16', '-gr', '1', '-ls', '5', '-eg', '1',
                '-niid', '1', '-pt', 'pat', '-cpc', '20', '-is_regular', '1', '-regular_lamda', '.001',
                '--target_client_id', '0', '--seed', '0', '--target_proj_mode', 'avg', '--target_head_mode', 'local',
                '--target_local_schedule', 'joint', '--target_head_l2_lambda', '.001', '--target_prox_mu', '0',
                '-exp_name', tag, '-sfn', str(root / 'c' / tag), '--h5_result_root', str(root / 'h' / tag),
                '--final-model-root', str(root / 'f' / tag), '--run_log_dir', str(root / 'l' / tag), '--clip_cpu_threads', '2']
            env = dict(os.environ, PYTHONIOENCODING='utf-8')
            with (root / 'output.txt').open('w', encoding='utf-8') as out:
                result = subprocess.run(command, cwd=system, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=600)
            output = (root / 'output.txt').read_text(encoding='utf-8')
            self.assertEqual(result.returncode, 0, output[-12000:])
            metrics = list((root / 'l').rglob('target_head_l2_clients.json'))
            self.assertEqual(len(metrics), 1)
            document = json.loads(metrics[0].read_text())
            self.assertEqual(len(document['history']), 40)
            self.assertEqual(document['policy']['parameter_names'], ['fc3.weight'])
            for row in document['history']:
                self.assertEqual(row['target_head_l2_lambda'], .001)
                self.assertEqual(row['train_epochs'], 5)
                self.assertEqual(row['train_batches'], 5)
                self.assertEqual(row['train_sample_visits'], 10)
            targets = json.loads((metrics[0].parent / 'target_head_l2_target.json').read_text())
            self.assertEqual(len(targets['history']), 2)
            for row in targets['history']:
                self.assertEqual(row['endpoint_train_samples'], 2)
                self.assertEqual(row['endpoint_test_samples'], 3)
            h5_paths = list((root / 'h').rglob('*.h5'))
            self.assertEqual(len(h5_paths), 1)
            with h5py.File(h5_paths[0]) as h:
                self.assertEqual(h['target_projection/target_head_l2/clients/train_epochs'].shape, (40,))
                self.assertEqual(h['target_projection'].attrs['final_target_local_acc'], targets['primary_summary']['final_target_local_acc'])
            manifests = list((root / 'f').rglob('manifest.json'))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text())
            self.assertEqual(manifest['args']['target_head_l2_lambda'], .001)
            self.assertEqual(manifest['target_head_l2']['policy'], document['policy'])
            model = torch.load(manifests[0].parent / 'Client_0_model.pt', map_location='cpu', weights_only=False)
            self.assertEqual(model.conv1.out_channels, 16)
            self.assertTrue(hasattr(model.fc1, 'weight_u'))
            self.assertEqual(tuple(model.eval()(torch.zeros(1, 3, 32, 32)).shape), (1, 100))
            archives = list((root / 'l').rglob('*.zip'))
            self.assertTrue(archives)
            with zipfile.ZipFile(archives[0]) as archive:
                self.assertTrue(any(n.endswith('target_head_l2_clients.csv') for n in archive.namelist()))
                self.assertFalse(any(n.endswith('.pt') for n in archive.namelist()))

    def test_main_parser_accepts_and_rejects_combinations_without_training(self):
        # Exercise actual main parser and naming; replace only run(), never execute training.
        script = r'''
import ast, contextlib, io, json, pathlib, sys
from unittest.mock import patch
path = pathlib.Path(sys.argv[1]); additions = json.loads(sys.argv[2]); sys.path.insert(0, str(path.parent))
tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
node.body = ast.parse('capture_args(args)').body; ast.fix_missing_locations(tree)
captured = []
sys.argv = [str(path), '-algo', 'FedTargetProj', '-m', 'Decom_CNN-5-512', '-ncl', '100', '-nc', '20', '-gr', '30', '-dev', 'cpu', '--target_proj_mode', 'avg', '--target_head_mode', 'local'] + additions
with contextlib.redirect_stdout(io.StringIO()), patch('utils.training_logs.training_log_session', side_effect=lambda a: contextlib.nullcontext()):
    exec(compile(tree, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path), 'capture_args': captured.append})
print(json.dumps(vars(captured[0])))
'''
        for additions, success in (([], True), (['--target_head_l2_lambda', '0'], True),
            (['--target_head_l2_lambda', '.01'], True), (['--target_head_l2_lambda', '-.1'], False),
            (['--target_head_l2_lambda', '.01', '-exp_name', 'requested_avg_local_head_head_l2_0p01'], True),
            (['--target_head_l2_lambda', 'nan'], False), (['--target_head_l2_lambda', 'inf'], False),
            (['--target_head_l2_lambda', '.01', '--target_head_mode', 'shared'], False),
            (['--target_head_l2_lambda', '.01', '--target_local_schedule', 'fedrep'], False)):
            with self.subTest(additions=additions):
                process = subprocess.run([sys.executable, '-c', script, str(ROOT / 'system/main.py'), json.dumps(additions)],
                    capture_output=True, text=True, env=dict(os.environ, PYTHONIOENCODING='utf-8'), encoding='utf-8', timeout=90)
                self.assertEqual(process.returncode == 0, success, process.stderr[-3000:])
                if success:
                    args = json.loads(process.stdout)
                    expected = float(additions[1]) if additions else 0.
                    self.assertEqual(args['target_head_l2_lambda'], expected)
                    self.assertEqual('head_l2_' in args['save_folder_name'], expected > 0)
                    if '-exp_name' in additions:
                        self.assertEqual(args['exp_name'], additions[additions.index('-exp_name') + 1])


if __name__ == '__main__': unittest.main()
