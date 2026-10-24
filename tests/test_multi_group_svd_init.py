"""Mother initialization only: frozen dense parity and original low-rank recovery."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import shlex
import subprocess
import sys
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'system'))
from test_multi_group_svd import options
from test_multi_group import args as multi_args
from test_control_population import assert_rng
from common.experiment import rng_state
from utils.multi_group_svd import initial_model, initialization_protocol, recover_parameters, DEFAULT_RATIOS
from utils.multi_group_clustering import grouping_context, digest_json, validate_group_file
from flcore.trainmodel.models import Hyper_CNN_512


class SVDInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_available(): torch.cuda.get_rng_state_all()
        source = subprocess.check_output(['git','show','a8ceabf:system/utils/multi_group_svd.py'],cwd=ROOT)
        cls.frozen = types.ModuleType('frozen_a8ceabf_svd')
        exec(compile(source,'frozen_a8ceabf_svd.py','exec'),cls.frozen.__dict__)

    def test_dense_default_and_explicit_parameters_and_every_rng_stream_are_exact(self):
        for seed in (0,19):
            for explicit in (False,True):
                a = options(seed=seed,**({'multi_svd_init':'dense','multi_svd_init_rank_rate':.15} if explicit else {}))
                before = rng_state()
                expected = self.frozen.initial_model(a); assert_rng(self,before)
                actual = initial_model(a); assert_rng(self,before)
                self.assertEqual(actual.ratio_LR,1.)
                self.assertEqual(expected.state_dict().keys(),actual.state_dict().keys())
                for name,p in actual.state_dict().items(): self.assertTrue(torch.equal(p,expected.state_dict()[name]),name)

    def test_low_rank_recovered_equals_independent_reference_and_copies(self):
        for seed,rate in [(0,.15),(19,.37),(0,1.)]:
            a = options(seed=seed,multi_svd_init='low_rank_recovered',multi_svd_init_rank_rate=rate)
            before = rng_state(); actual = initial_model(a); assert_rng(self,before)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                low = Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=16,ratio_LR=rate,input_size=32)
                products = {n: (getattr(low,n).reconstruct_full_weight().detach().clone()
                               if rate < 1 else getattr(low,n).weight.detach().clone()) for n in ('conv2','fc1','fc2')}
                low.recover_larger_model()
            assert_rng(self,before)
            for name,p in actual.named_parameters(): self.assertTrue(torch.equal(p,dict(low.named_parameters())[name]),name)
            for name,w in products.items(): self.assertTrue(torch.equal(getattr(actual,name).weight,w),name)
            self.assertEqual(actual.ratio_LR,1.)
            self.assertTrue(all(isinstance(getattr(actual,n),(torch.nn.Conv2d,torch.nn.Linear)) for n in ('conv2','fc1','fc2')))
            copies = [actual]+[copy.deepcopy(actual) for _ in range(2)]
            for name,p in copies[0].named_parameters():
                for model in copies[1:]:
                    self.assertTrue(torch.equal(p,dict(model.named_parameters())[name]))
                    self.assertIsNot(p,dict(model.named_parameters())[name])

    def test_original_five_downloads_reconstruct_low_rank_mother_with_roundoff_only(self):
        mother = initial_model(options(multi_svd_init='low_rank_recovered'))
        with redirect_stdout(io.StringIO()):
            for rate in DEFAULT_RATIOS:
                local = copy.deepcopy(mother); local.decom_larger_model(rate); local.ratio_LR=rate
                recovered = recover_parameters(local)
                self.assertEqual(local.ratio_LR,rate)
                for name,p in mother.named_parameters():
                    error = (p.detach()-recovered[name]).double()
                    relative = error.norm()/p.detach().double().norm().clamp_min(1e-30)
                    self.assertLessEqual(float(relative),3e-6,(rate,name))
                    self.assertLessEqual(float(error.abs().max()),1e-5,(rate,name))

    def test_protocol_rejects_mode_or_initial_rank_mismatch_even_with_same_hash(self):
        data = dict(config_sha256='config',files=[])
        dense = grouping_context(options(),data,'same_hash')
        self.assertNotIn('svd_initialization',dense)
        for rate in (.15,.37,1.):
            a = options(multi_svd_init='low_rank_recovered',multi_svd_init_rank_rate=rate)
            context = grouping_context(a,data,'same_hash')
            self.assertEqual(context['svd_initialization']['effective_low_rank_rate'],rate)
            with self.assertRaises(ValueError):
                validate_group_file(dict(schema_version=2,context=dense,context_sha256=digest_json(dense)),context)
        changed = grouping_context(options(multi_svd_init='low_rank_recovered',multi_svd_init_rank_rate=.37),data,'same_hash')
        base = grouping_context(options(multi_svd_init='low_rank_recovered'),data,'same_hash')
        with self.assertRaises(ValueError):
            validate_group_file(dict(schema_version=2,context=base,context_sha256=digest_json(base)),changed)

    def test_options_rank_not_inferred_from_client_minimum_and_illegal_inputs(self):
        a = options(multi_svd_init='low_rank_recovered',multi_svd_ratios='.4,.3')
        self.assertEqual(initialization_protocol(a)['effective_low_rank_rate'],.15)
        for extra in [dict(multi_svd_init='other'),dict(multi_svd_init_rank_rate=.2)]+[
            dict(multi_svd_init='low_rank_recovered',multi_svd_init_rank_rate=r) for r in (0.,-.1,1.1,float('nan'),float('inf'))]:
            with self.subTest(extra=extra),self.assertRaises(ValueError): options(**extra)
        for extra in [dict(multi_svd_init='low_rank_recovered'),dict(multi_svd_init_rank_rate=.3)]:
            with self.assertRaises(ValueError): multi_args(**extra)

    def test_only_one_formal_K1_original_capacities_command_and_separate_paths(self):
        text = (ROOT/'system/multi_group_svd_init_commands.sh').read_text()
        commands = [shlex.split(line.strip('"')) for line in text.splitlines() if line.startswith('"python -u main.py')]
        self.assertEqual(len(commands),1)
        c = commands[0]
        for key,value in [('-nsm','1'),('-nc','20'),('-gr','100'),('-ls','5'),('-lr','0.005'),('-lbs','16'),
            ('--multi_svd_ratios','0.9,0.37,0.35,0.25,0.15'),('--multi_svd_init','low_rank_recovered'),
            ('--multi_svd_init_rank_rate','0.15'),('--multi_group_sizes','20'),('--multi_group_mode','group_avg_only'),
            ('--multi_group_weight','1.0'),('-regular_lamda','1e-3'),('--local_grad_clip','10')]:
            self.assertEqual(c[c.index(key)+1],value)
        self.assertNotIn('--multi_group_file',c)
        self.assertIn('multi_svd_init_runs/',text)


if __name__ == '__main__': unittest.main()
