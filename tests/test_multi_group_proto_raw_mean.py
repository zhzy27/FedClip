"""Natural-scale sum/count targets, typed transport, and fixed-path parity."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from common.experiment import rng_state,set_rng
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_proto import (Features,prototype_loss,merge_statistics,collect_client,decode_packet,
    prepare_round,Session,validate_representation,save_latest)
from utils.multi_group_proto_options import representations,protocol
from test_multi_group_proto import fixture
from test_multi_group_svd import options
from test_control_population import assert_rng


def raw_statistics():
    # Samples in group 0: [100,0] once, [0,2] three times.
    # Equal-client means and per-sample unit means BOTH give different answers.
    values={}
    for cid,counts in enumerate(([1,1,0,0],[3,1,0,0],[2,0,2,0])):
        sums=torch.zeros(2,4,2,dtype=torch.float64)
        sums[0,0]=torch.tensor([100.,0.]) if cid==0 else torch.tensor([0.,6.]) if cid==1 else torch.tensor([2000.,0.])
        sums[1,0]=sums[0,0].flip(0)*2
        if cid==0:sums[0,1]=torch.tensor([3.,-4.]);sums[1,1]=torch.tensor([-3.,4.])
        if cid==1:sums[0,1]=torch.tensor([-3.,4.]);sums[1,1]=torch.tensor([3.,-4.])
        if cid==2:sums[0,2]=torch.tensor([-8.,6.]);sums[1,2]=torch.tensor([20.,0.])
        values[cid]=dict(sums=sums,counts=torch.tensor(counts),capacity_level=cid,rank_rate=.9-cid*.2,
            **representations('raw_mean'))
    return values


def raw_fixture(folder,coefficient=1.,diagnostics=1):
    server=fixture(folder,coefficient,diagnostics)
    server.args.multi_proto_loss='mse';server.args.multi_proto_feature='base_output';server.args.multi_proto_target_type='raw_mean'
    return server


class NaturalPrototypeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_sum_count_preserves_natural_scale_direction_and_columns(self):
        st=raw_statistics();p,v,rows,_,_=merge_statistics(st,[[0,1],[2]],1e-12,1,'g',prototype_target_type='raw_mean')
        self.assertTrue(torch.equal(p[0,0],torch.tensor([25.,1.5],dtype=torch.float64)))
        self.assertTrue(torch.equal(p[1,0],torch.tensor([0.,2000.],dtype=torch.float64)))
        self.assertTrue(torch.equal(p[0,2],torch.tensor([-4.,3.],dtype=torch.float64)))
        self.assertEqual(float(p[0,2].norm()),5.)
        self.assertGreater(float(p[0,0].norm()),25.)
        normalized_mean=torch.tensor([1.,3.],dtype=torch.float64)/4
        self.assertFalse(torch.allclose(F.normalize(p[0,0],dim=0),F.normalize(normalized_mean,dim=0)))
        self.assertEqual(rows[0]['source'],'group');self.assertEqual(rows[2]['source'],'outside')
        changed=copy.deepcopy(st);changed[2]['sums'][0,0].fill_(1e9)
        p2,*_=merge_statistics(changed,[[0,1],[2]],1e-12,1,'g',prototype_target_type='raw_mean')
        self.assertTrue(torch.equal(p2[0,0],p[0,0]))

    def test_zero_and_near_zero_mean_are_valid_no_outside_replacement(self):
        st=raw_statistics();p,v,rows,directions,_=merge_statistics(st,[[0,1],[2]],1e-12,1,'g',prototype_target_type='raw_mean')
        self.assertTrue(v[0,1]);self.assertTrue(torch.equal(p[0,1],torch.zeros(2,dtype=torch.float64)))
        row=rows[1];self.assertEqual(row['source'],'group');self.assertIsNone(row['invalid_reason'])
        self.assertTrue(row['mean_is_exact_zero']);self.assertEqual(row['source_client_ids'],[0,1])
        self.assertFalse(v[:,3].any());self.assertEqual(rows[3]['invalid_reason'],'no_training_samples')
        st[1]['sums'][0,1,0]+=1e-14
        p,v,rows,*_=merge_statistics(st,[[0,1],[2]],1e-12,1,'g',prototype_target_type='raw_mean')
        self.assertTrue(v[0,1]);self.assertTrue(rows[1]['mean_is_near_zero']);self.assertGreater(float(p[0,1].norm()),0)

    def test_nonfinite_statistics_and_type_mismatch_fail(self):
        st=raw_statistics();st[0]['sums'][0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):merge_statistics(st,[[0,1],[2]],1e-12,1,'g',prototype_target_type='raw_mean')
        st=raw_statistics();st[0].pop('statistics_representation')
        with self.assertRaisesRegex(ValueError,'representation'):merge_statistics(st,[[0,1],[2]],1e-12,1,'g',prototype_target_type='raw_mean')
        with self.assertRaisesRegex(ValueError,'type mismatch'):validate_representation({},'raw_mean','test')
        with self.assertRaisesRegex(ValueError,'type mismatch'):validate_representation(representations('raw_mean'),'fixed_norm','test')

    def test_raw_loss_and_gradient_equal_direct_MSE_zero_target_valid(self):
        z=torch.randn(4,7,dtype=torch.float64,requires_grad=True);p=torch.randn(3,7,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():p[1].zero_()
        y=torch.tensor([0,1,2,0]);valid=torch.tensor([True,True,False]);selection=valid[y]
        loss,n=prototype_loss(z,y,p,valid,1e-12,'mse',1.,'raw_mean')
        reference=z.detach().clone().requires_grad_(True)
        expected=F.mse_loss(reference[selection],p.detach()[y[selection]])
        self.assertTrue(torch.equal(loss,expected));loss.backward();expected.backward()
        self.assertTrue(torch.equal(z.grad,reference.grad));self.assertIsNone(p.grad);self.assertEqual(n,3)
        zero,_=prototype_loss(z,y,p,torch.zeros(3,dtype=torch.bool),1e-12,'mse',1.,'raw_mean')
        self.assertEqual(float(zero),0.)
        self.assertTrue(torch.equal(torch.autograd.grad(zero,z)[0],torch.zeros_like(z)))
        with self.assertRaises(ValueError):prototype_loss(z,y,p,valid,1e-12,'mse',2.,'raw_mean')
        bad=p.detach().clone();bad[0,0]=float('inf')
        with self.assertRaises(FloatingPointError):prototype_loss(z,y,bad,valid,1e-12,'mse',1.,'raw_mean')
        with self.assertRaises(FloatingPointError):prototype_loss(torch.full_like(z,1e200),y,p,valid,1e-12,'mse',1.,'raw_mean')

    def test_collection_matches_actual_raw_forward_and_preserves_state_RNG_modes(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=raw_fixture(folder);client=server.clients[0]
            prepare_binding=dict(loop_round=1,assigned_model_id=0,group_fingerprint='fixed',
                **{k:server.data_manifest['files'][0][k] for k in ('decoded_x_sha256','decoded_y_sha256')})
            before=[tensor_fingerprint(m.state_dict()) for m in client.model_set];modes=[[m.training for m in model.modules()] for model in client.model_set]
            state=rng_state();packet,rows,upload=collect_client(client,prepare_binding);assert_rng(self,state)
            statistics,meta=decode_packet(packet,'statistics');validate_representation(meta,'raw_mean','test')
            data=client.load_train_data().dataset
            with torch.no_grad():
                for column,source in enumerate(client.model_set):
                    model=copy.deepcopy(source).eval();sums=torch.zeros_like(statistics['sums'][column])
                    for begin in range(0,len(data),server.args.multi_proto_batch_size):
                        batch=data[begin:begin+server.args.multi_proto_batch_size]
                        features=model.base(torch.stack([x for x,y in batch])).double()
                        for feature,(_,y) in zip(features,batch):sums[int(y)].add_(feature)
                    self.assertTrue(torch.equal(statistics['sums'][column],sums))
            self.assertEqual(before,[tensor_fingerprint(m.state_dict()) for m in client.model_set])
            self.assertEqual(modes,[[m.training for m in model.modules()] for model in client.model_set])
            self.assertTrue(all(r['statistics_representation']=='raw_feature_class_sum' for r in rows))

    def test_collective_training_types_target_norm_and_backbone_gradient(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=raw_fixture(folder);prepare_round(server);client=server.clients[0]
            self.assertIsNone(protocol(server.args)['target_norm'])
            values,meta=decode_packet(server.proto_latest_packet,'library');validate_representation(meta,'raw_mean','library')
            self.assertGreater(float(values['prototypes'].norm(dim=2).max()),1.)
            session=Session(client);session.begin_column();model=client.model_set[0]
            data=client.load_train_data().dataset;x=torch.stack([a for a,b in data]);y=torch.stack([b for a,b in data])
            before=tensor_fingerprint(model.state_dict());ce,reg,raw,weighted,n=session.objective(0,model,x,y);raw.backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.base.parameters()))
            self.assertIsNone(model.fc3.weight.grad);self.assertEqual(before,tensor_fingerprint(model.state_dict()))
            fields=session.fields(0,float(raw)*len(y),float(weighted)*len(y),float(ce+reg+weighted)*len(y),n,len(y))
            self.assertIsNone(fields['aux_target_norm']);self.assertIsNone(fields['actual_fixed_target_norm'])
            self.assertGreater(fields['training_actual_target_l2_mean'],1.)
            client.proto_receipt.pop('prototype_target_type')
            with self.assertRaisesRegex(ValueError,'type mismatch'):Session(client)
            client.proto_receipt['prototype_target_type']='raw_mean';client.args.multi_proto_feature='classifier_input'
            with self.assertRaisesRegex(ValueError,'feature position'):Session(client)

    def test_wrong_statistics_receiver_and_saved_library_type_rejected(self):
        import utils.multi_group_proto as runtime
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=raw_fixture(folder);native=runtime.collect_client
            def wrong(*args):
                packet,rows,upload=native(*args);values,meta=decode_packet(packet,'statistics')
                meta.update(representations('fixed_norm'))
                return runtime.encode_packet('statistics',values,meta),rows,upload
            with patch.object(runtime,'collect_client',wrong),self.assertRaisesRegex(ValueError,'type mismatch'):prepare_round(server)
            prepare_round(server);server.args.multi_proto_target_type='fixed_norm'
            with self.assertRaisesRegex(ValueError,'type mismatch'):save_latest(server)

    def test_fixed_norm_packet_and_active_train_match_43eb806(self):
        old={};source=subprocess.run(['git','show','43eb806:system/utils/multi_group_proto.py'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
        exec(compile(source,'43eb806/multi_group_proto.py','exec'),old)
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=fixture(folder);server.args.multi_proto_loss='mse';server.args.multi_proto_feature='base_output'
            other=copy.deepcopy(server);state=rng_state();old['prepare_round'](other);set_rng(state);prepare_round(server)
            self.assertEqual(server.proto_latest_packet,other.proto_latest_packet)
            start=rng_state()
            with patch('utils.multi_group_proto.Session',old['Session']):other.clients[0].train()
            expected=rng_state();set_rng(start);server.clients[0].train();assert_rng(self,expected)
            for a,b in zip(server.clients[0].model_set,other.clients[0].model_set):self.assertEqual(tensor_fingerprint(a.state_dict()),tensor_fingerprint(b.state_dict()))

    def test_diagnostics_do_not_change_raw_libraries_training_uploads_or_RNG(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=raw_fixture(folder);other=copy.deepcopy(server);other.args.multi_proto_diagnostics=0
            state=rng_state();prepare_round(server);set_rng(state);prepare_round(other)
            self.assertEqual(server.proto_latest_packet,other.proto_latest_packet)
            start=rng_state();server.clients[0].train();expected=rng_state();set_rng(start);other.clients[0].train();assert_rng(self,expected)
            for a,b in zip(server.clients[0].model_set,other.clients[0].model_set):self.assertEqual(tensor_fingerprint(a.state_dict()),tensor_fingerprint(b.state_dict()))

    def test_zero_and_first_round_skip_collection_and_feature_work(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=raw_fixture(folder,coefficient=0)
            with patch('utils.multi_group_proto.Features',side_effect=AssertionError('No features')),\
                 patch('utils.multi_group_proto.collect_client',side_effect=AssertionError('No collection')):
                prepare_round(server);server.clients[0].train()
            self.assertFalse(server.proto_round_records)
            server=raw_fixture(folder);server.loop_round=0
            for c in server.clients:c.download_round=0
            with patch('utils.multi_group_proto.Features',side_effect=AssertionError('No first-round features')),\
                 patch('utils.multi_group_proto.collect_client',side_effect=AssertionError('No first-round collection')):
                prepare_round(server);server.clients[0].train()
            self.assertFalse(server.proto_round_records)

    def test_validation_raw_only_MSE_and_no_nondefault_fixed_target_norm(self):
        for changes in [dict(multi_proto_mode='none'),dict(multi_proto_loss='cosine'),dict(multi_proto_loss='z2'),
                        dict(multi_proto_target_norm=2),dict(multi_proto_target_type='bad')]:
            params=dict(multi_proto_mode='group_priority',multi_proto_loss='mse',multi_proto_target_type='raw_mean');params.update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):options(**params)
        a=options(multi_proto_mode='group_priority',multi_proto_loss='mse',multi_proto_target_type='raw_mean',multi_proto_lambda=0)
        self.assertIsNone(protocol(a)['target_norm'])

    def test_two_new_commands_use_existing_groups_and_no_fixed_norm_argument(self):
        rows=(ROOT/'system/multi_group_proto_raw_mean_commands.sh').read_text().splitlines()
        commands=[shlex.split(row[1:-1]) for row in rows if row.startswith('"python ')]
        self.assertEqual(len(commands),2)
        for command,coefficient in zip(commands,('0.1','1')):
            for key,value in [('--multi_proto_target_type','raw_mean'),('--multi_proto_loss','mse'),('--multi_proto_feature','base_output'),
                              ('--multi_proto_lambda',coefficient),('-gr','100'),('--multi_proto_start_round','2')]:
                self.assertEqual(command[command.index(key)+1],value)
            self.assertNotIn('--multi_group_publish',command);self.assertNotIn('--multi_proto_target_norm',command)
        self.assertEqual(commands[0][commands[0].index('--multi_group_file')+1],commands[1][commands[1].index('--multi_group_file')+1])


if __name__=='__main__':unittest.main()
