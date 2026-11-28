from contextlib import redirect_stdout
from itertools import combinations
import io
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
import unittest

import torch

from test_single_avg_proto import options
from flcore.trainmodel.models import FactorizedConv, FactorizedLinear, Hyper_CNN_512
from utils import multi_fc1_rank as fc1, multi_group_svd as svd, multi_svd_spectral as spectral


class SelectiveSVDTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_seven_subsets_construct_recover_and_reload(self):
        all_layers = svd.DEFAULT_FACTORIZED_LAYERS
        types = {'conv2': FactorizedConv, 'fc1': FactorizedLinear, 'fc2': FactorizedLinear}
        for count in range(1, 4):
            for selected in combinations(all_layers, count):
                with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temp:
                    configured = ', '.join(reversed(selected))
                    a = options(multi_svd_factorized_layers=configured, cnn_hidden_width=32,
                                multi_proto_mode='none', multi_proto_lambda=0., multi_svd_init_rank_rate=.9)
                    self.assertEqual(svd.factorized_layers(a), selected)
                    mother = svd.initial_model(a)
                    dense = {name: p.clone() for name, p in mother.named_parameters()}
                    for name, kind in types.items():
                        self.assertNotIsInstance(getattr(mother, name), kind)
                    with redirect_stdout(io.StringIO()):
                        mother.decom_larger_model(.37)
                    mother.ratio_LR = .37
                    for name, kind in types.items():
                        self.assertEqual(isinstance(getattr(mother, name), kind), name in selected)
                        if name not in selected:
                            self.assertTrue(torch.equal(getattr(mother, name).weight, dense[name+'.weight']))
                    for name in ('conv1', 'fc3'):
                        self.assertTrue(torch.equal(getattr(mother, name).weight, dense[name+'.weight']))
                    f = sum(getattr(mother, name).reconstruct_full_weight().square().sum() for name in selected)
                    torch.testing.assert_close(mother.frobenius_decay(), f)
                    upload = svd.recover_parameters(mother)
                    self.assertEqual(upload.keys(), dense.keys())
                    self.assertTrue(all(upload[n].shape == v.shape for n, v in dense.items()))
                    file = Path(temp)/'client.pt'
                    torch.save(dict(svd.model_spec(a,.37,mother), state_dict=mother.state_dict()), file)
                    restored, _ = svd.load_checkpoint(file)
                    self.assertEqual(restored.factorized_layers, selected)
                    self.assertTrue(all(torch.equal(v, restored.state_dict()[n]) for n,v in mother.state_dict().items()))

    def test_bad_layers_and_independent_schedule_and_download(self):
        self.assertEqual(svd.factorized_layers(SimpleNamespace()), ('conv2','fc1','fc2'))
        for value in ('', 'fc1,', 'fc1,,fc2', 'conv1', 'fc3', 'fc1,fc1', 'fc1, fc1', 'fc1,unknown'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'nonempty subset'):
                svd.factorized_layers(SimpleNamespace(multi_svd_factorized_layers=value))
        for selected in ('fc2','conv2,fc2'):
            for method in ('full_avg','rank_partition'):
                a = options(multi_svd_factorized_layers=selected,multi_svd_aggregation=method,
                            multi_proto_mode='none',multi_proto_lambda=0.)
                self.assertFalse(fc1.uses_local_schedule(a))
                self.assertFalse(fc1.uses_shared_download(a))
            with self.assertRaisesRegex(ValueError,'requires low-rank fc1'):
                options(multi_svd_factorized_layers=selected,multi_fc1_local_mode='staged_equal_frozen',
                        multi_proto_mode='none',multi_proto_lambda=0.)
        for selected in ('fc1','fc1,fc2','conv2,fc1,fc2'):
            a = options(multi_svd_factorized_layers=selected,multi_fc1_local_mode='staged_equal_frozen',
                        multi_svd_aggregation='rank_partition',multi_proto_mode='none',multi_proto_lambda=0.)
            self.assertTrue(fc1.uses_local_schedule(a))
            self.assertEqual(fc1.uses_shared_download(a), selected=='fc1')

    def test_real_width_936_parameter_counts(self):
        for rate, expected in ((.15,386612),(.9,1984244)):
            model = Hyper_CNN_512(num_classes=100,ratio_LR=rate,hidden_width=936,factorized_layers=('fc1','fc2'))
            self.assertEqual(sum(p.numel() for p in model.parameters()),expected)

    def test_partial_spectral_boundaries_use_actual_factorized_layers(self):
        a = options(multi_svd_factorized_layers='fc2,conv2',multi_svd_spectral_mode='single_point',
                    multi_svd_spectral_layers='fc2',multi_proto_mode='none',multi_proto_lambda=0.)
        mother = svd.initial_model(a)
        self.assertEqual(list(spectral.boundary_ranks(mother,a)),['conv2','fc2'])
        a.multi_svd_spectral_layers = 'fc1'
        with self.assertRaisesRegex(ValueError,'only factorized layers'):
            spectral.validate(a)

    def test_command_sextet_keeps_same_clip_controls(self):
        path = Path(__file__).resolve().parents[1]/'system/selective_svd_clip_commands.sh'
        cmds = [shlex.split(l.strip()[1:-1]) for l in path.read_text().splitlines() if l.lstrip().startswith('"')]
        self.assertEqual(len(cmds),6)
        def flag(cmd, name): return cmd[cmd.index(name)+1]
        combinations_used = {(flag(c,'--multi_svd_factorized_layers'),flag(c,'--multi_svd_aggregation'),flag(c,'--multi_fc1_local_mode')) for c in cmds}
        self.assertEqual(combinations_used,{(layer,agg,train) for layer in ('fc1,fc2','conv2,fc1,fc2')
            for agg,train in [('full_avg','full'),('rank_partition','full'),('rank_partition','staged_equal_frozen')]})
        for c in cmds:
            for key,value in (('--cnn_hidden_width','936'),('-gr','99'),('--multi_clip_mse_lambda','1.0'),
                 ('--use_asymmetric_lr','0'),('--use_loss_specific_u_scaling','0'),('--multi_proto_mode','none')):
                self.assertEqual(flag(c,key),value)
        for key in ('-sfn','--run_log_dir','--h5_result_root','--final-model-root'):
            self.assertEqual(len({flag(c,key) for c in cmds}),6)
