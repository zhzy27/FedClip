"""Independent budget batch: real counts, alias handling and read-only audit."""
import copy
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'system'))
from common.experiment import rng_state
from utils.multi_group_budget import read_config, build_audit, inspect_client, parameter_count, markdown_report
from utils.multi_group_svd import DEFAULT_RATIOS
from flcore.trainmodel.models import Hyper_CNN_512
from test_control_population import assert_rng


class ParameterBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_available(): torch.cuda.get_rng_state_all()
        cls.config = read_config()
        cls.report = build_audit()

    def test_all_actual_models_match_table_and_20_clients_fit(self):
        expected_ranks = [(24,239,153),(13,129,83),(9,91,58),(7,71,46),(4,41,27)]
        self.assertEqual(len(self.report['clients']),20)
        for cid,c in enumerate(self.report['clients']):
            level = cid%5
            self.assertEqual(c['baseline_parameters'],self.config['expected_baseline_parameters'][level])
            self.assertEqual(c['local_parameter_total'],self.config['expected_total_parameters'][level])
            self.assertLessEqual(c['local_parameter_total'],c['baseline_parameters'])
            self.assertEqual(len(c['local_models']),3)
            for m in c['local_models']:
                self.assertEqual(m['parameter_count'],self.config['expected_single_parameters'][level])
                self.assertEqual(tuple(m['layers'][n]['rank'] for n in ('conv2','fc1','fc2')),expected_ranks[level])
        self.assertFalse(self.report['communication_budget_aligned'])
        self.assertFalse(self.report['peak_memory_measured'])
        self.assertEqual(self.report['server_full_parameter_total'],3*2691860)
        self.assertEqual(self.report['previous_default_K1_reference']['single_model_parameters'],
                         [3246372,1365828,1295428,941396,587364])
        self.assertIn('|19|5|715,932|722,360|PASS|',markdown_report(self.report))

    def test_read_only_actual_models_aliases_modes_buffers_and_rng(self):
        models = [Hyper_CNN_512(**self.config['model_kwargs'],ratio_LR=.299) for _ in range(3)]
        models[0].eval()
        for m in models: m.register_buffer('audit_probe',torch.tensor([2.]))
        before = copy.deepcopy([m.state_dict() for m in models])
        modes = [[n.training for n in m.modules()] for m in models]
        state = rng_state()
        row = inspect_client(models,0,self.config,3348360)
        self.assertEqual(row,self.report['clients'][0])
        assert_rng(self,state)
        self.assertEqual(modes,[[n.training for n in m.modules()] for m in models])
        for old,m in zip(before,models):
            for name,p in m.state_dict().items(): self.assertTrue(torch.equal(old[name],p))
        m = models[0]
        self.assertIs(m.fc3.weight,m.head[1].weight)
        self.assertGreater(sum(v.numel() for v in m.state_dict().values()),parameter_count(m))
        self.assertEqual(parameter_count(m),1114356)
        # The standalone constructor audit also restores every RNG stream.
        build_audit(); assert_rng(self,state)

    def test_manifest_actual_ranks_counts_and_mismatch_fail(self):
        config = self.config
        manifest = dict(protocol=dict(rank_rates=config['rank_rates'],
            client_level_ids=[i%5 for i in range(20)],client_rank_rates=[config['rank_rates'][i%5] for i in range(20)]),clients={})
        for c in self.report['clients']:
            manifest['clients'][str(c['client_id'])] = [dict(rank_rate=c['rank_rate'],layers=m['layers'],
                trainable_parameter_count=m['parameter_count'],recovered_parameter_count=2691860) for m in c['local_models']]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'capacity_manifest.json'
            path.write_text(json.dumps(manifest))
            self.assertTrue(build_audit(capacity_manifest=path)['capacity_manifest_checked'])
            for change in ('rank','count','K','old_capacity'):
                wrong = copy.deepcopy(manifest)
                if change == 'rank': wrong['clients']['0'][0]['layers']['fc1']['rank'] += 1
                if change == 'count': wrong['clients']['0'][0]['trainable_parameter_count'] += 1
                if change == 'K': wrong['clients']['0'].pop()
                if change == 'old_capacity': wrong['protocol']['rank_rates'] = list(DEFAULT_RATIOS)
                path.write_text(json.dumps(wrong))
                with self.subTest(change=change),self.assertRaises(ValueError): build_audit(capacity_manifest=path)

    def test_changed_K_shape_rate_or_budget_requires_recalculation(self):
        models = [Hyper_CNN_512(**self.config['model_kwargs'],ratio_LR=.299) for _ in range(3)]
        for k,baseline in [(2,3348360),(3,3348359)]:
            with self.assertRaises(ValueError): inspect_client(models[:k],0,self.config,baseline)
        with self.assertRaisesRegex(ValueError,'independent Parameter'):
            inspect_client([models[0]]*3,0,self.config,3348360)
        wrong = [Hyper_CNN_512(num_classes=10,ratio_LR=.299) for _ in range(3)]
        with self.assertRaisesRegex(ValueError,'width/classes'): inspect_client(wrong,0,self.config,3348360)
        wrong = [Hyper_CNN_512(num_classes=100,ratio_LR=.9) for _ in range(3)]
        with self.assertRaisesRegex(ValueError,'Rank rate'): inspect_client(wrong,0,self.config,3348360)
        config = copy.deepcopy(self.config); config['expected_single_parameters'][0] += 1
        with self.assertRaises(ValueError): inspect_client(models,0,config,3348360)

    def test_new_groups_reject_old_capacity_and_keep_both_comparison_lists(self):
        from utils.multi_group_clustering import digest_json
        def payload(rates):
            context = dict(num_clients=20,K=3,capacities=[7,7,6],model_family='Decom_CNN-5-512',
                svd_capacity_protocol=dict(rank_rates=list(rates),client_level_ids=[i%5 for i in range(20)]))
            groups = [list(range(7)),list(range(7,14)),list(range(14,20))]
            return dict(schema_version=1,context=context,context_sha256=digest_json(context),groups=groups,
                client_to_model={str(cid):k for k,g in enumerate(groups) for cid in g},
                group_fingerprint=digest_json(dict(context=context,groups=groups)))
        with tempfile.TemporaryDirectory() as d:
            new,old = Path(d)/'new.json',Path(d)/'old.json'
            new.write_text(json.dumps(payload(self.config['rank_rates'])))
            old.write_text(json.dumps(payload(DEFAULT_RATIOS)))
            report = build_audit(group_file=new,previous_group_file=old)
            self.assertEqual(report['groups']['members'],payload(self.config['rank_rates'])['groups'])
            self.assertEqual(report['previous_groups']['context']['svd_capacity_protocol']['rank_rates'],list(DEFAULT_RATIOS))
            with self.assertRaisesRegex(ValueError,'new capacity batch'): build_audit(group_file=old)

    def test_separate_two_commands_and_no_default_capacity_change(self):
        self.assertEqual(DEFAULT_RATIOS,(.9,.37,.35,.25,.15))
        text = (ROOT/'system/multi_group_budget_commands.sh').read_text()
        commands = [shlex.split(line.strip('"')) for line in text.splitlines() if line.startswith('"python -u main.py')]
        self.assertEqual(len(commands),2)
        for c,gpu,mode in zip(commands,['0','1'],['group_avg_only','group_projection_softmax']):
            for key,value in [('-did',gpu),('-nsm','3'),('-nc','20'),('-ls','5'),('-gr','100'),('-eg','1'),
                ('--multi_group_mode',mode),('--multi_svd_ratios',','.join(map(str,self.config['rank_rates']))),
                ('--multi_group_sizes','7,7,6'),('--multi_group_basis','full_weights'),('--multi_warmup_rounds','0'),
                ('--multi_group_weight','1.0' if mode=='group_avg_only' else '0.50'),('--group_softmax_tau','0.20'),('--multi_svd_diagnostics','1'),
                ('--target_prox_mu','0'),('--target_head_l2_lambda','0'),('-regular_lamda','1e-3')]:
                self.assertEqual(c[c.index(key)+1],value)
        self.assertEqual(commands[0][commands[0].index('--multi_group_file')+1],
                         commands[1][commands[1].index('--multi_group_publish')+1])
        self.assertIn('multi_svd_budget_runs/',text)
        self.assertNotEqual(commands[0][commands[0].index('-sfn')+1],commands[1][commands[1].index('-sfn')+1])


if __name__ == '__main__': unittest.main()
