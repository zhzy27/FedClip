"""Online downloaded-teacher semantics, exact disabled paths and KD arithmetic."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from common.data_loading import make_data_loader
from common.experiment import rng_state,set_rng,tensor_digest
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_kd import distillation_loss,remove_true,Session,bind_assignment,_forward_private
from utils.multi_group_kd_options import validate_kd,protocol,ensure_kd_name
from utils.multi_group_clustering import grouping_context
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Hyper_CNN_512
from test_multi_group_svd import options
from test_control_population import assert_rng
from utils.cnn512_reporting import model_digest


def make_client(folder,mode='full',coefficient=.1,K=3,probes=0):
    args=options(num_server_models=K,multi_group_mode='group_avg_only' if K==1 else 'group_projection_softmax',
        multi_group_weight=1. if K==1 else .5,multi_kd_mode=mode,multi_kd_lambda=coefficient,
        multi_kd_probe_samples=probes,local_epochs=1,batch_size=2)
    args.device='cpu';args.save_folder_name_full=str(folder);args.few_shot=0
    model=Hyper_CNN_512(num_classes=100,n_kernels=2,ratio_LR=1.)
    c=clientMultiGroupSVD(args,1,3,3,initial_model=model,train_slow=False,send_slow=False)
    servers=[copy.deepcopy(model) for _ in range(K)]
    # Different columns make cached pre-training targets distinguishable.
    with torch.no_grad():
        for k,m in enumerate(servers):m.fc3.bias.add_(torch.linspace(-.1,.1,100)*k)
    generator=torch.Generator().manual_seed(89)
    data=[(torch.randn(3,32,32,generator=generator),torch.tensor(y)) for y in (3,8,12)]
    c.load_train_data=lambda:make_data_loader(data,2,shuffle=True,drop_last=False,num_workers=0)
    return c,servers,data


def download(c,servers,loop,assigned=2):
    c.set_parameters(servers,loop)
    groups=[[] for _ in range(c.K)];groups[assigned]=[c.id]
    data=c.load_train_data().dataset
    fake=SimpleNamespace(loop_round=loop,groups=None if loop==0 else groups,
        group_payload=None if loop==0 else dict(client_to_model={str(c.id):assigned},group_fingerprint='fixed'),
        data_manifest=dict(files=[dict(client_id=c.id,split='train',sha256='file',
            decoded_x_sha256=tensor_digest(torch.stack([x for x,y in data])),decoded_y_sha256=tensor_digest(torch.stack([y for x,y in data])))]))
    bind_assignment(c,fake)


class OnlineKDTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_full_averages_probabilities_KL_direction_and_T_squared(self):
        student=torch.tensor([[1.,2.,-1.],[2.,-1.,1.]],dtype=torch.float64,requires_grad=True)
        teachers=torch.tensor([[[9.,0.,0.],[0.,2.,0.]],[[0.,4.,1.],[4.,1.,0.]]],dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([0,2]);T=2.;q=(teachers.detach()/T).softmax(-1).mean(1)
        raw=distillation_loss(student,teachers,labels,'full',T)
        expected=(q*(q.log()-(student/T).log_softmax(-1))).sum(-1).mean()
        torch.testing.assert_close(raw,expected,atol=1e-15,rtol=1e-14)
        self.assertGreater(float((q-(teachers.mean(1)/T).softmax(-1)).abs().max()),.01)
        (.1*T*T*raw).backward();self.assertIsNone(teachers.grad);self.assertTrue(torch.isfinite(student.grad).all())

    def test_not_true_per_teacher_delete_then_normalize_and_zero_direct_true_gradient(self):
        student=torch.tensor([[10.,1.,3.,2.],[2.,3.,-8.,1.]],dtype=torch.float64,requires_grad=True)
        teachers=torch.tensor([[[20.,6.,1.,0.],[-7.,1.,2.,4.]],[[3.,2.,12.,4.],[5.,0.,-15.,1.]]],dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([0,2]);T=2.
        raw=distillation_loss(student,teachers,labels,'not_true',T)
        teacher_parts=torch.stack([torch.stack([torch.cat((row[:y],row[y+1:])) for row,y in zip(teachers[:,k],labels)]) for k in range(2)],dim=1)
        q=(teacher_parts.detach()/T).softmax(-1).mean(1)
        student_parts=torch.stack([torch.cat((row[:y],row[y+1:])) for row,y in zip(student,labels)])
        expected=(q*(q.log()-(student_parts/T).log_softmax(-1))).sum(-1).mean()
        torch.testing.assert_close(raw,expected,atol=1e-15,rtol=1e-14)
        wrong=remove_true((teachers/T).softmax(-1).mean(1),labels);wrong=wrong/wrong.sum(-1,keepdim=True)
        self.assertGreater(float((q-wrong).abs().max()),.01)
        raw.backward();self.assertTrue(torch.equal(student.grad.gather(1,labels[:,None]),torch.zeros(2,1,dtype=torch.float64)))
        self.assertGreater(float(student.grad.abs().sum()),0.);self.assertIsNone(teachers.grad)
        for K in (1,3):
            value=distillation_loss(student.detach(),teachers.detach()[:,:1].expand(-1,K,-1),labels,'not_true',T)
            self.assertTrue(torch.isfinite(value))

    def test_options_and_group_fingerprint_exclude_KD(self):
        base=options();data=dict(config_sha256='c',files=[])
        expected=grouping_context(base,data,'init')
        for mode in ('full','not_true'):
            a=options(multi_kd_mode=mode,multi_kd_lambda=.1)
            self.assertEqual(expected,grouping_context(a,data,'init'))
            a.exp_name='experiment';ensure_kd_name(a);self.assertIn('onlineKD_'+mode,a.exp_name)
            self.assertEqual(protocol(a)['first_eligible_loop_round'],1)
        for change in [dict(multi_kd_lambda=-1),dict(multi_kd_lambda=float('nan')),dict(multi_kd_temperature=0),
                       dict(multi_kd_temperature=float('inf')),dict(multi_kd_teacher_batch_size=0),dict(multi_kd_observe_gap=0),
                       dict(multi_kd_probe_samples=-1),dict(multi_svd=False,model_family='CNN-512'),dict(target_prox_mu=.1)]:
            with self.subTest(change=change),self.assertRaises(ValueError):options(multi_kd_mode='full',**change)
        a=copy.deepcopy(base);a.algorithm='FedTargetProj';a.multi_kd_mode='full'
        with self.assertRaises(ValueError):validate_kd(a)

    def test_download_cache_immutable_when_earlier_external_columns_train_and_stale_round_refused(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            c,servers,data=make_client(folder);download(c,servers,1,assigned=2)
            before=[tensor_fingerprint(m.state_dict()) for m in c.model_set];modes=[[m.training for m in model.modules()] for model in c.model_set]
            state=rng_state();session=Session(c,c.load_train_data());assert_rng(self,state)
            expected,_,_=_forward_private(c.model_set[:2],data,list(range(3)),'cpu',16)
            self.assertTrue(torch.equal(session.cache,expected));self.assertEqual(session.teacher_ids,[0,1]);self.assertEqual(session.assigned,2)
            self.assertEqual(before,[tensor_fingerprint(m.state_dict()) for m in c.model_set])
            self.assertEqual(modes,[[m.training for m in model.modules()] for model in c.model_set])
            frozen=session.cache.clone()
            x=torch.stack([p[0] for p in data]);y=torch.stack([p[1] for p in data])
            for col in (0,1):
                opt=c.optimizer_set[col];opt.zero_grad();c.loss(c.model_set[col](x),y).backward();opt.step()
            self.assertTrue(torch.equal(frozen,session.cache));self.assertTrue(all(p.requires_grad for m in c.model_set for p in m.parameters()))
            with self.assertRaisesRegex(RuntimeError,'modified before teacher caching'):Session(c,c.load_train_data())
            ce,frob,raw,weighted=session.objective(c.model_set[2],x,y,torch.arange(3))
            torch.testing.assert_close(weighted,.1*4*raw)
            c.download_round=2
            with self.assertRaisesRegex(RuntimeError,'Stale'):session.objective(c.model_set[2],x,y,torch.arange(3))

    def test_zero_and_first_round_exact_legacy_training_uploads_batches_and_rng(self):
        # Exact source of the original class method is a fixed, independent reference.
        text=subprocess.run(['git','show','e1e7f61:system/flcore/clients/clientMultiGroupSVD.py'],cwd=ROOT,
                            text=True,capture_output=True,check=True).stdout
        scope={'__name__':'legacy_online_KD_reference'};exec(compile(text,'e1e7f61/clientMultiGroupSVD.py','exec'),scope)
        legacy_train=scope['clientMultiGroupSVD'].train
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            c,servers,data=make_client(folder,mode='none');download(c,servers,1,2)
            saved=copy.deepcopy(c);state=rng_state();legacy_train(saved);expected_rng=rng_state()
            expected=[{n:p.clone() for n,p in m.state_dict().items()} for m in saved.model_set]
            for mode,coefficient,loop in [('none',.1,1),('full',0.,1),('not_true',0.,1),('full',.1,0),('not_true',.1,0)]:
                trial=copy.deepcopy(c);trial.args.multi_kd_mode=mode;trial.args.multi_kd_lambda=coefficient
                trial.download_round=loop;trial.kd_binding['loop_round']=loop
                if loop==0:trial.kd_binding['assigned_model_id']=None
                set_rng(state)
                with patch('utils.multi_group_kd._forward_private',side_effect=AssertionError('No extra forward on disabled path')):trial.train()
                assert_rng(self,expected_rng)
                for k,m in enumerate(trial.model_set):
                    self.assertTrue(all(torch.equal(p,expected[k][n]) for n,p in m.state_dict().items()))
                    self.assertTrue(all(torch.equal(p,saved.full_uploads[k][n]) for n,p in trial.full_uploads[k].items()))

    def test_only_real_assigned_column_K2_K3_and_probe_toggle_preserves_training(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            for K in (2,3):
                assigned=0 if K==2 else 2
                c,servers,data=make_client(folder,K=K);download(c,servers,1,assigned)
                state=rng_state();watched=copy.deepcopy(c);reference=copy.deepcopy(c);reference.args.multi_kd_mode='none'
                plain=copy.deepcopy(c);plain.args.multi_kd_probe_samples=0;watched.args.multi_kd_probe_samples=2
                reference.train();reference_parameters=[tensor_fingerprint(m.state_dict()) for m in reference.model_set]
                set_rng(state);plain.train();plain_rng=rng_state()
                set_rng(state);watched.train();assert_rng(self,plain_rng)
                self.assertEqual([r['kd_enabled'] for r in plain.train_stats],[k==assigned for k in range(K)])
                self.assertEqual([r['kd_effective_temperature'] for r in plain.train_stats],[2. if k==assigned else None for k in range(K)])
                for k in range(K):
                    self.assertEqual(tensor_fingerprint(plain.model_set[k].state_dict()),tensor_fingerprint(watched.model_set[k].state_dict()))
                    if k!=assigned:self.assertEqual(tensor_fingerprint(plain.model_set[k].state_dict()),reference_parameters[k])
                self.assertEqual(len(watched.kd_probe_records),2);self.assertEqual(watched.kd_probe_records[0]['probe_sample_ids'],watched.kd_probe_records[1]['probe_sample_ids'])
                self.assertTrue(all(r['train_batches']==2 and r['processed_train_samples']==3 for r in plain.train_stats))

    def test_K1_zero_effective_no_teacher_cache(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            c,servers,data=make_client(folder,K=1);download(c,servers,1,0)
            with patch('utils.multi_group_kd._forward_private',side_effect=AssertionError('K1 must not forward teachers')):c.train()
            row=c.train_stats[0];self.assertFalse(row['kd_enabled']);self.assertEqual(row['kd_disabled_reason'],'K1_no_external_teachers')
            self.assertEqual(row['teacher_cache_bytes'],0);self.assertEqual(row['weighted_kd_loss'],0.)

    def test_teacher_copy_preserves_nontrivial_buffers_modes_flags_and_backend(self):
        from utils.head_snapshot import backend_settings
        class Buffered(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bn=torch.nn.BatchNorm1d(3);self.head=torch.nn.Linear(3,100)
            def forward(self,x):return self.head(self.bn(x.mean((2,3))))
        source=Buffered().train()
        source.bn.running_mean.copy_(torch.tensor([1.,2.,3.]));source.bn.num_batches_tracked.fill_(19)
        data=[(torch.ones(3,32,32)*i,torch.tensor(0)) for i in range(3)]
        before=tensor_fingerprint(source.state_dict());state=rng_state();backend=backend_settings()
        logits,states,_=_forward_private([source],data,[0,1,2],'cpu',2)
        self.assertEqual(logits.shape,(3,1,100));self.assertFalse(logits.requires_grad)
        self.assertEqual(before,tensor_fingerprint(source.state_dict()));self.assertEqual(states,[before])
        self.assertTrue(all(m.training for m in source.modules()))
        self.assertTrue(all(p.requires_grad and p.grad is None for p in source.parameters()))
        self.assertEqual(backend,backend_settings());assert_rng(self,state)

    def test_command_arrays_have_two_jobs_and_independent_optional_baseline(self):
        lines=(ROOT/'system/multi_group_kd_commands.sh').read_text().splitlines()
        commands=[shlex.split(row[1:-1]) for row in lines if row.startswith('"python ')]
        self.assertEqual(len(commands),3)
        for command,mode,gpu in zip(commands,('full','not_true','none'),('1','2','0')):
            for flag,value in [('--multi_kd_mode',mode),('-did',gpu),('-nsm','3'),('-nc','20'),('-gr','100'),('-ls','5'),
                ('--multi_svd_ratios','0.9,0.37,0.35,0.25,0.15'),('--multi_svd_init','low_rank_recovered'),
                ('--multi_svd_init_rank_rate','0.15'),('--multi_kd_temperature','2'),('--group_softmax_tau','0.20'),
                ('--multi_group_weight','0.50'),('--multi_group_mode','group_projection_softmax')]:
                self.assertEqual(command[command.index(flag)+1],value)
        self.assertEqual(commands[0][commands[0].index('--multi_group_publish')+1],commands[1][commands[1].index('--multi_group_file')+1])
        self.assertIn('declare -a OPTIONAL_BASELINE_COMMANDS=(',lines)


if __name__=='__main__':unittest.main()
