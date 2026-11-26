import io
from contextlib import redirect_stdout
from pathlib import Path
import shlex
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'system'))
from flcore.trainmodel.models import Hyper_CNN_512
from utils import multi_fc1_rank as fc1
from utils.multi_rank_partition import partition_product
from utils.single_avg_proto import loader_seed


class Fc1RankTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_reorder_moves_the_vectors_stably_and_keeps_zeros(self):
        u = torch.eye(6, dtype=torch.float64).roll(2, 0)
        vh = torch.eye(6, dtype=torch.float64).roll(1, 1)
        s = torch.tensor([10., 8., 6., 4., 3., 0.], dtype=torch.float64)
        left, values, right, amplified, coverage, order = fc1.coverage_reorder(u, s, vh, [2, 3, 5], [2, 3, 5])
        torch.testing.assert_close(coverage, s.new_tensor([1, 1, .8, .5, .5]))
        torch.testing.assert_close(amplified, s.new_tensor([10, 8, 7.5, 8, 6, 0]))
        self.assertEqual(order.tolist(), [0, 1, 3, 2, 4, 5])  # equal 8 keeps the original ordering
        torch.testing.assert_close((left*values)@right, (u*amplified)@vh)
        left, values, right, _, _, order = fc1.coverage_reorder(u, s, vh, [2, 5], [8, 2])
        self.assertEqual(order[:2].tolist(), [2, 3])
        torch.testing.assert_close((left[:, :2]*values[:2])@right[:2], 30*u[:, 2:3]@vh[2:3] + 20*u[:, 3:4]@vh[3:4])
        self.assertEqual(values[-1].item(), 0.)

    def test_partition_uses_products_and_original_unequal_counts(self):
        entries = []
        for cid, rank in enumerate((1, 2, 3)):
            u = torch.arange(1., 13.).reshape(4, 3)[:, :rank] + cid
            v = torch.arange(1., 10.).reshape(3, 3)[:rank] - cid
            entries.append(dict(client_id=cid, U=u, V=v, rank=rank))
        a,b,c = [entry['U']@entry['V'] for entry in entries]
        b_first = entries[1]['U'][:, :1]@entries[1]['V'][:1]
        c_first = entries[2]['U'][:, :1]@entries[2]['V'][:1]
        c_last = entries[2]['U'][:, 2:]@entries[2]['V'][2:]
        expected = (2*a+3*b_first+7*c_first)/12 + (3*(b-b_first)+7*(c-c_first-c_last))/10 + c_last
        result, rows = partition_product(entries, {0:2, 1:3, 2:7})
        torch.testing.assert_close(result, expected)
        self.assertEqual([row['contributor_train_samples_total'] for row in rows], [12, 10, 7])

    def test_quota_pairing_zero_skips_and_private_label_shuffle(self):
        ranks = [120, 200, 280, 296, 720]
        expected = {'random_weighted':[115,76,76,15,423], 'staged_weighted':[118,78,78,16,415],
                    'staged_equal':[141]*5, 'random_equal':[141]*5}
        for mode, counts in expected.items():
            plan, actual = fc1.batch_plan(mode, ranks, 705, 1)
            self.assertEqual(counts, actual)
            if mode.startswith('staged'):
                frozen, frozen_counts = fc1.batch_plan(mode+'_frozen', ranks, 705, 2)
                self.assertEqual(counts, frozen_counts)
                self.assertEqual([a for a,_ in plan], [a for a,_ in frozen])
        self.assertEqual(fc1.batch_plan('staged_weighted_frozen', ranks, 3, 0),
                         ([(120, 0), (720, 120), (720, 120)], [1,0,0,0,2]))
        from types import SimpleNamespace
        dataset = torch.arange(42)
        def batches(mode):
            loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                generator=torch.Generator().manual_seed(loader_seed(SimpleNamespace(seed=0), 5, 1)))
            fc1.batch_plan(mode, ranks, len(loader)*5, 117)
            return [batch.tolist() for _ in range(5) for batch in loader]
        rng = torch.get_rng_state().clone()
        self.assertEqual(batches('random_equal'), batches('full'))
        self.assertEqual(batches('random_weighted'), batches('full'))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_step_masks_CE_F_and_optimizer_binding(self):
        model = Hyper_CNN_512(num_classes=100,n_kernels=2,ratio_LR=.7,factorized_layers=('fc1',),hidden_width=64)
        u,v = model.fc1.weight_u,model.fc1.weight_v
        x,y = torch.rand(3,3,32,32), torch.tensor([1,2,4])
        reference, regular = model(x), model.frobenius_decay()
        with fc1.use_prefix(model.fc1,model.fc1.rank,0):
            self.assertTrue(torch.equal(reference,model(x)))
            self.assertTrue(torch.equal(regular,model.frobenius_decay()))
        optimizer = torch.optim.SGD(model.parameters(),lr=.005,momentum=0,weight_decay=0)
        for active,frozen in [(5,0),(18,5),(24,18),(18,0)]:
            before = {name:p.detach().clone() for name,p in model.named_parameters()}
            optimizer.zero_grad()
            with fc1.use_prefix(model.fc1,active,frozen):
                ce,f = F.cross_entropy(model(x),y),model.frobenius_decay()
                torch.testing.assert_close(f,(u[:, :active]@v[:active]).square().sum())
            for objective in [ce,f]:
                gradients = torch.autograd.grad(objective,(u,v),retain_graph=True)
                for gradient in (gradients[0].T,gradients[1]):
                    self.assertEqual(gradient[:frozen].count_nonzero().item(),0)
                    self.assertEqual(gradient[active:].count_nonzero().item(),0)
                    self.assertGreater(gradient[frozen:active].count_nonzero().item(),0)
            (ce+.001*f).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            for old,new in [(before['fc1.weight_u'].T,u.T),(before['fc1.weight_v'],v)]:
                self.assertTrue(torch.equal(old[:frozen],new[:frozen]))
                self.assertTrue(torch.equal(old[active:],new[active:]))
                self.assertFalse(torch.equal(old[frozen:active],new[frozen:active]))
            for name in ('conv1.weight','conv2.weight','fc1.bias','fc2.weight','fc3.weight'):
                self.assertFalse(torch.equal(before[name],dict(model.named_parameters())[name]))
            self.assertFalse(hasattr(model.fc1,'_fc1_training_prefix'))
        self.assertIs(model.fc1.weight_u,u)
        self.assertIs(model.fc1.weight_v,v)

    def test_commands_cover_each_method_schedule_pair_and_real_ranks(self):
        lines = (ROOT/'system/fc1_rank_schedule_commands.sh').read_text().splitlines()
        jobs = [shlex.split(l.strip()[1:-1]) for l in lines if l.strip().startswith('"')]
        self.assertEqual(len(jobs),14)
        def value(job,key):return job[job.index(key)+1]
        self.assertEqual({(value(j,'--multi_svd_aggregation'),value(j,'--multi_fc1_local_mode')) for j in jobs},
                         {(a,m) for a in ('full_avg','rank_partition') for m in fc1.MODES})
        for flag in ('-sfn','--run_log_dir','--h5_result_root','--final-model-root'):
            self.assertEqual(len({value(job,flag) for job in jobs}),14)
        per_job = ('-did','-sfn','--run_log_dir','--h5_result_root','--final-model-root','-exp_name',
                   '--multi_svd_aggregation','--multi_svd_spectral_mode','--multi_fc1_local_mode')
        def shared(job):
            job = job.copy()
            for flag in per_job:
                index = job.index(flag)
                del job[index:index+2]
            return job
        for job in jobs:
            self.assertEqual(shared(job), shared(jobs[0]))
            expected_spectrum = 'coverage_reorder' if value(job,'--multi_svd_aggregation')=='full_avg' else 'none'
            self.assertEqual(value(job,'--multi_svd_spectral_mode'), expected_spectrum)
            for flag,expected in [('-gr','99'),('-ls','5'),('-nsm','1'),('-nc','20'),('--multi_svd_factorized_layers','fc1'),
                    ('--multi_svd_ratios','0.90,0.37,0.35,0.25,0.15'),('--multi_proto_mode','none'),('--cnn_hidden_width','2000')]:
                self.assertEqual(value(job,flag),expected)
        for rate,rank in zip([.9,.37,.35,.25,.15],[720,296,280,200,120]):
            model = Hyper_CNN_512(num_classes=100,ratio_LR=rate,factorized_layers=('fc1',))
            self.assertEqual(model.fc1.rank,rank)
            self.assertIsInstance(model.conv2,torch.nn.Conv2d)
            self.assertIsInstance(model.fc2,torch.nn.Linear)
            with redirect_stdout(io.StringIO()),torch.no_grad():
                model.recover_larger_model();model.ratio_LR=1.
                model.decom_larger_model(rate)
            self.assertEqual(model.fc1.rank,rank)
            self.assertIsInstance(model.conv2,torch.nn.Conv2d)
            self.assertIsInstance(model.fc2,torch.nn.Linear)
