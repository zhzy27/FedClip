"""Configuration ranges, real shard audit capability and controlled initialization."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'system'))
from common.client_data import audit_image_clients
from common.models import initial_full_model
from utils.cnn512_reporting import model_digest, scope_ids, primary_point
from utils.cnn512_control import validate_control, effective_grad_clip, seed_control, evaluation_due, preview_control, resolve_data_settings
from utils.projection_variants import projection_similarity_weights
from utils.group_projection import aggregate_group_updates
from test_control_population import options


def configured(**kw):
    values=dict(partition='pat',class_per_client=3,global_rounds=3,num_clients=4,seed=7,
        target_client_id=2,local_learning_rate=.012,batch_size=4,local_epochs=2,eval_gap=2,
        control_test_batch_size=3,num_server_models=2,smooth_mu=.025,local_grad_clip=0,
        data_subdir='arbitrary_existing_name',data_expected_partition=None,data_expected_alpha=None,
        projection_softmax_tau=.2,group_softmax_tau=.2,fedfews_rng='original',fedfews_control_seed=0)
    values.update(kw)
    args=options(**values);validate_control(args);return args


class ConfigurableTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_nondefaults_requested_effective_and_clipping_default_rules(self):
        a=configured(algorithm='FedFewS')
        self.assertEqual((a.seed,a.num_clients,a.num_server_models,a.local_epochs,a.batch_size),(7,4,2,2,4))
        self.assertEqual(a.control_requested_config['fedfews_rng'],'original')
        self.assertEqual(a.control_effective_config['fedfews_rng'],'controlled')
        self.assertEqual(a.fedfews_control_seed,7)
        self.assertIsNone(effective_grad_clip(a))
        self.assertEqual(configured(local_grad_clip=None).local_grad_clip,10.)
        b=copy.deepcopy(a);b.cnn512_control=False;b.local_grad_clip=None
        self.assertIsNone(effective_grad_clip(b))
        b.algorithm='FedTargetProj';self.assertEqual(effective_grad_clip(b),10.)
        with self.assertRaises(ValueError):configured(fedfews_test_batch_size=8,algorithm='FedFewS')

    def test_pat10_formal_commands_have_matching_public_protocol(self):
        import re,shlex
        text=(ROOT/'system/cnn512_pat10_commands.sh').read_text(encoding='utf-8')
        body=text.split('declare -a COMMANDS=(',1)[1].split('\n)',1)[0]
        commands=[shlex.split(x) for x in re.findall(r'^"([^\n]+)"$',body,re.MULTILINE)]
        self.assertEqual(len(commands),2)
        for args in commands:
            for key,value in [('-m','CNN-512'),('-data','Cifar100'),('-nc','20'),('-cpc','10'),
                ('--data-subdir','pat_10'),('-pt','pat'),('--control_eval_scope','all'),('--seed','0'),
                ('-lr','0.005'),('-lbs','16'),('-ls','5'),('-gr','100'),('-eg','1'),
                ('--local_grad_clip','10'),('--control_test_batch_size','16'),('-is_regular','0')]:
                self.assertEqual(args[args.index(key)+1],value)
        few=commands[1]
        self.assertEqual(few[few.index('-nsm')+1],'3');self.assertEqual(few[few.index('-smu')+1],'0.01')

    def test_range_validation_and_no_fixed_protocol_values(self):
        for change in [dict(global_rounds=0,eval_gap=10),dict(seed=99,target_client_id=3),
            dict(local_learning_rate=.1,batch_size=1,local_epochs=1),dict(local_grad_clip=2.5),
            dict(target_proj_mode='projection_softmax',projection_self_weight=.63,projection_softmax_tau=.6),
            dict(target_proj_mode='group_projection_softmax',control_eval_scope='group',target_group_ids=[0,3],target_group_weight=None,group_softmax_tau=.7)]:
            configured(**change)
        for change in [dict(global_rounds=-1),dict(eval_gap=0),dict(local_epochs=0),dict(num_clients=0),
            dict(target_client_id=4),dict(seed=-1),dict(seed=2**32),dict(local_grad_clip=-1),
            dict(local_grad_clip=float('nan')),dict(local_learning_rate=float('inf')),dict(smooth_mu=0),
            dict(batch_size=0),dict(group_softmax_tau=0),dict(projection_softmax_tau=-1),
            dict(partition='exdir'),dict(class_per_client=0),dict(data_expected_alpha=-1),
            dict(target_proj_mode='group_projection_softmax',control_eval_scope='group',target_group_ids=[0,0]),
            dict(target_proj_mode='group_projection_softmax',control_eval_scope='group',target_group_ids=[0,5])]:
            with self.subTest(change=change),self.assertRaises(ValueError):configured(**change)

    def test_seed_changes_initialization_and_all_controlled_streams(self):
        hashes=[];streams=[]
        for seed in [0,7,7]:
            args=configured(seed=seed)
            hashes.append(model_digest(initial_full_model(args)))
            seed_control(args)
            streams.append((random.random(),np.random.rand(),torch.rand(1).item()))
        self.assertNotEqual(hashes[0],hashes[1]);self.assertEqual(hashes[1],hashes[2])
        self.assertNotEqual(streams[0],streams[1]);self.assertEqual(streams[1],streams[2])

    def test_target_ids_sparse_schedule_and_temperature_weights(self):
        args=configured(control_eval_scope='c0')
        self.assertEqual(scope_ids(args),[2])
        point=primary_point(args,3,[dict(client_id=2,correct=2,test_samples=3)])
        self.assertEqual(point['member_ids'],'2')
        self.assertEqual([r for r in range(4) if evaluation_due(args,r)],[0,2,3])
        args.global_rounds=0;self.assertTrue(evaluation_due(args,0))
        rows=[dict(client_id=i,sample_weight=1/3,cosine_before_projection=c) for i,c in enumerate([1.,-.7,.5])]
        a=projection_similarity_weights(rows,0,'projection_softmax',.3,.2)
        b=projection_similarity_weights(rows,0,'projection_softmax',.3,.7)
        self.assertEqual(a[2]['temperature'],.2);self.assertEqual(b[2]['temperature'],.7)
        self.assertNotEqual(a[0][1],b[0][1]);self.assertEqual(a[0][0],.3)
        initial={'head.weight':torch.zeros(2,dtype=torch.float64)}
        posts=[{'head.weight':torch.tensor(v,dtype=torch.float64)} for v in [(1.,0.),(-1.,1.),(.5,1.)]]
        factory=lambda:iter((i,w,posts[i]) for i,w in enumerate([.2,.3,.5]))
        g=aggregate_group_updates(initial,factory,dict(enumerate([.2,.3,.5])),[0],'group_projection_softmax',None,.7)
        self.assertEqual(g[3]['softmax_temperature'],.7);self.assertAlmostEqual(g[1]['actual_group_mass'],.2)

    def test_data_dependent_mass_defaults_are_recorded_without_changing_arguments(self):
        args=configured(target_proj_mode='group_projection_softmax',control_eval_scope='group',
                        target_group_ids=[0,3],target_group_weight=None)
        audit=dict(actual_alpha=.5,config=dict(partition='pat'),files=[dict(client_id=i,count=n,split='train')
            for i,n in enumerate([2,3,4,5])])
        actual=resolve_data_settings(args,audit)
        self.assertEqual(actual['target_group_weight'],.5)
        self.assertIsNone(args.target_group_weight)
        self.assertIsNone(args.control_requested_config['target_group_weight'])
        args=configured(target_proj_mode='projection_softmax',projection_self_weight=None)
        self.assertEqual(resolve_data_settings(args,audit)['projection_self_weight'],4/14)
        from flcore.servers.serverbase import Server
        s=Server.__new__(Server);s.args=configured(partition='dir',dir_alpha=.1)
        resolve_data_settings(s.args,dict(audit,config=dict(partition='dir')))
        self.assertEqual(s._partition_path_component(),'dir_alpha0p5')
        s.args.cnn512_control=False
        self.assertEqual(s._partition_path_component(),'dir_alpha0p1')

    def test_explicit_zero_clip_matches_unclipped_sgd_in_both_clients(self):
        from flcore.clients.clientTargetProj import clientTargetProj
        from flcore.clients.clientfedfews import clientFedFewS
        from flcore.clients.clientbase import save_item, load_item
        from contextlib import redirect_stdout
        import io
        args=configured(local_epochs=1)
        model=initial_full_model(args)
        x,y=torch.randn(4,3,32,32)*20,torch.tensor([0,1,2,3])
        expected=copy.deepcopy(model);opt=torch.optim.SGD(expected.parameters(),lr=args.local_learning_rate)
        loss=torch.nn.functional.cross_entropy(expected(x),y);loss.backward();opt.step()
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            c=clientTargetProj.__new__(clientTargetProj)
            c.args=args;c.device='cpu';c.id=2;c.role='Client_2';c.save_folder_name=directory
            c.learning_rate=args.local_learning_rate;c.local_epochs=1;c.train_slow=False
            c.loss=torch.nn.CrossEntropyLoss();c.train_time_cost=dict(num_rounds=0,total_cost=0.)
            c.load_train_data=lambda:[(x,y)];save_item(model,c.role,'model',directory);c.train()
            actual=load_item(c.role,'model',directory)
            f=clientFedFewS.__new__(clientFedFewS);f.args=args;f.K=1;f.device='cpu';f.local_epochs=1;f.train_samples=4
            f.model_set=[copy.deepcopy(model)];f.optimizer_set=[torch.optim.SGD(f.model_set[0].parameters(),lr=args.local_learning_rate)]
            f.loss=torch.nn.CrossEntropyLoss();f.load_train_data=lambda:[(x,y)];f.train_time_cost=dict(num_rounds=0,total_cost=0.)
            f.train()
            for result in [actual,f.model_set[0]]:
                for name,p in expected.named_parameters():self.assertTrue(torch.equal(p,dict(result.named_parameters())[name]),name)

    def test_actual_shards_pat_dir_metadata_mismatch_fingerprints_no_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'Cifar100/arbitrary_existing_name'
            for split in ['train','test']:
                (directory/split).mkdir(parents=True)
                for cid in range(4):
                    x=np.arange(3*3*32*32,dtype='float32').reshape(3,3,32,32)
                    np.savez(directory/split/f'{cid}.npz',data=dict(x=x,y=np.array([cid,cid+1,cid+2])))
            config=dict(num_clients=4,num_classes=100,partition='pat',class_per_client=3,alpha=.5)
            file=directory/'config.json';file.write_text(json.dumps(config));original=file.read_bytes()
            a=configured(data_root=str(root));b=configured(data_root=str(root),algorithm='FedFewS')
            self.assertEqual(audit_image_clients(a),audit_image_clients(b))
            self.assertEqual(audit_image_clients(a)['partition_verification']['verification_level'],'observed_class_counts_match')
            self.assertEqual(file.read_bytes(),original)
            preview=preview_control(a);self.assertFalse(preview['training_started'])
            self.assertEqual(preview['data_epochs_per_client'],2)
            for field,value in [('num_clients',5),('num_classes',10),('class_per_client',10),('partition','dir')]:
                bad=dict(config);bad[field]=value;file.write_text(json.dumps(bad))
                with self.subTest(field=field),self.assertRaises(ValueError):audit_image_clients(a)
            file.write_text(json.dumps(dict(config,partition='dir',class_per_client=1)))
            d=configured(data_root=str(root),partition='dir',class_per_client=99,data_expected_alpha=.5)
            self.assertFalse(audit_image_clients(d)['partition_verification']['cpc_applicable'])
            d.data_expected_alpha=.1
            with self.assertRaises(ValueError):audit_image_clients(d)
            # Incomplete old metadata is reported as partial, never inferred from a folder.
            file.write_text(json.dumps({k:v for k,v in config.items() if k!='class_per_client'}))
            a.class_per_client=10
            self.assertEqual(audit_image_clients(a)['partition_verification']['verification_level'],'partial')
            bad=dict(config);bad['Size of samples for labels in clients']=[[[0,99]]]*4
            file.write_text(json.dumps(bad));a.class_per_client=3
            with self.assertRaises(ValueError):audit_image_clients(a)


if __name__=='__main__':unittest.main()
