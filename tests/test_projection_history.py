"""Independent active-set oracle, finite cache, legacy compatibility and scope."""
import copy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'system'))
from utils import projection_history as ph
from utils.projection_variants import aggregate_projection_weighting
from fedfews_compat.observe import rng_digest


def params(v):
    # Include a head; all geometry uses the complete parameter set exactly once.
    return dict(backbone=torch.tensor(v[:2], dtype=torch.float64),
                **{'fc3.weight': torch.tensor(v[2:], dtype=torch.float64)})


class HistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():
            torch.cuda.get_rng_state_all()

    def test_active_set_against_independent_constrained_optimizer(self):
        gen = np.random.default_rng(763)
        directions = [(np.array([1.,0.,0.]), np.array(q)) for q in
            ([1.,0.,0.], [-1.,0.,0.], [0.,1.,0.], [1.,1e-7,0.], [-1.,1e-7,0.])]
        directions += [(gen.normal(size=3), gen.normal(size=3)) for _ in range(50)]
        for u, q in directions:
            u=u/np.linalg.norm(u);q=q/np.linalg.norm(q)
            for strength in (0., 1., 10., 1e4):
                d=gen.normal(size=3)
                A,B,flags=ph.solve_coefficients(u@d,q@d,u@q,strength)
                v=d+A*u+B*q
                def fun(x):return .5*np.sum((x-d)**2)+strength/2*min(0.,q@x)**2
                def grad(x):return x-d+strength*min(0.,q@x)*q
                oracle=minimize(fun, np.zeros(3), jac=grad, method='SLSQP',
                    constraints=[dict(type='ineq',fun=lambda x:u@x,jac=lambda x:u)],
                    options=dict(ftol=1e-12,maxiter=1000))
                self.assertTrue(oracle.success,oracle.message)
                self.assertGreaterEqual(u@v,-1e-10)
                self.assertAlmostEqual(fun(v),oracle.fun,delta=2e-8*max(1.,oracle.fun))
                self.assertLess(np.linalg.norm(v-oracle.x),2e-4)
                self.assertGreaterEqual(A,-1e-9)  # hard-constraint multiplier

    def test_not_sequential_projection_and_history_is_soft(self):
        u=np.array([1.,0.]);q=np.array([-.8,.6]);d=np.array([-1.,-1.])
        A,B,_=ph.solve_coefficients(u@d,q@d,u@q,1.)
        v=d+A*u+B*q
        self.assertGreaterEqual(u@v,-1e-14)
        self.assertLess(q@v,0.)  # allowed historical residual
        sequential=d-min(0.,u@d)*u
        sequential-=.5*min(0.,q@sequential)*q
        self.assertLess(u@sequential,0.)  # the prohibited shortcut breaks the hard constraint
        np.testing.assert_allclose(v,[0.,-1/1.36],atol=1e-14)

    def test_tiny_angle_large_strength_uses_stable_orthogonal_system(self):
        base=params([0.,0.,0.]);target=params([1.,0.,0.]);helper=params([-1.,-1.,0.])
        for sign in (-1.,):
            cache=ph.History(10,.8);cache.commit(0,params([sign,1e-10,0.]))
            factory=lambda:iter([(0,.5,target),(1,.5,helper)])
            result=ph.aggregate(base,target,factory,0,cache,1,1e20,.125)
            # On u.v=0, penalty strength*(1e-10*v_y)^2 / 2:
            # minimize .5*(v_y+1)^2 + .5*v_y^2 -> v_y=-.5.
            torch.testing.assert_close(torch.cat(list(result[0].values())),
                torch.tensor([.125,-.4375,0.],dtype=torch.float64),rtol=0,atol=1e-13)

    def test_cache_weights_eviction_copy_schema_no_current_leak(self):
        cache=ph.History(2,.5);value=params([1.,2.,3.]);cache.commit(0,value)
        value['backbone'].fill_(999.)
        cache.commit(1,params([3.,4.,5.]))
        mean,rounds=cache.mean(2,params([0.,0.,0.]))
        self.assertEqual(rounds,[1,0]);self.assertEqual(cache.bytes,48)
        torch.testing.assert_close(mean['backbone'],torch.tensor([7/3,10/3],dtype=torch.float64))
        mean['backbone'].zero_()
        cache.commit(2,params([7.,8.,9.]))
        self.assertEqual([r for r,_ in cache.entries],[1,2])
        with self.assertRaises(ValueError):cache.mean(2,params([0.,0.,0.]))
        with self.assertRaises(ValueError):cache.commit(2,params([0.,0.,0.]))
        with self.assertRaises(ValueError):cache.mean(3,dict(other=torch.zeros(1)))
        latest=ph.History(1,1.);latest.commit(0,params([7.,8.,9.]))
        self.assertEqual(latest.mean(1,params([0.,0.,0.]))[1],[0])

    def case(self,reference=None,strength=1.,target=None,diagnostics=True):
        base=params([0.,0.,0.]);target=params(target or [1.,0.,0.])
        posts={2:target,0:params([-1.,-2.,1.]),1:params([1.,-1.,-2.])}
        factory=lambda:iter((cid,1/3,posts[cid]) for cid in (1,2,0))
        cache=ph.History(10,.8) if strength else None
        if reference is not None:cache.commit(0,params(reference))
        inputs=copy.deepcopy((base,posts,cache.state() if cache else None))
        old=aggregate_projection_weighting(base,target,factory,2,'projection_softmax',.125,.2)
        rng=rng_digest()
        result=ph.aggregate(base,target,factory,2,cache,1 if reference is not None else 0,strength,.125,.2,diagnostics)
        self.assertEqual(rng,rng_digest())
        for actual, expected in zip((base,posts),inputs[:2]):
            if actual is base:
                self.assertTrue(all(torch.equal(actual[n],expected[n]) for n in actual))
            else:
                self.assertTrue(all(torch.equal(actual[i][n],expected[i][n]) for i in actual for n in actual[i]))
        self.assertEqual([r['aggregation_weight'] for r in old[2]],[r['aggregation_weight'] for r in result[2]])
        self.assertEqual([r['cosine_before_projection'] for r in old[2]],[r['cosine_before_projection'] for r in result[2]])
        if cache:
            self.assertEqual([r for r,_ in cache.entries],[0] if reference is not None else [])
            for (_, actual), (_, expected) in zip(cache.entries, inputs[2]['entries']):
                self.assertTrue(all(torch.equal(actual[n], expected[n]) for n in actual))
        return old,result,posts,cache

    def test_no_history_zero_strength_zero_nearzero_and_aligned_legacy_exact(self):
        for kw in ({},{'strength':0.},{'reference':[0.,0.,0.]},{'reference':[1e-15,0.,0.]},
                   {'reference':[1.,2.,0.],'target':[0.,0.,0.]},
                   {'reference':[1.,2.,0.],'target':[1e-15,0.,0.]},{'reference':[2.,0.,0.]}):
            old,result,_,_=self.case(**kw)
            self.assertTrue(result[1]['history_fallback_reason'])
            self.assertTrue(all(torch.equal(result[0][n],old[0][n]) for n in old[0]))
            self.assertEqual(result[1]['history_applied'],0)

    def test_full_parameter_solver_and_observation_readonly(self):
        old,result,posts,cache=self.case(reference=[-.8,.6,0.])
        self.assertEqual(result[1]['history_applied'],1)
        self.assertGreater(max(r['correction_from_legacy_norm'] for r in result[2]),0)
        expected=torch.zeros(3,dtype=torch.float64)
        q=torch.tensor([-.8,.6,0.],dtype=torch.float64)
        for row in result[2]:
            d=torch.cat(list(posts[row['client_id']].values()))
            if row['client_id']==2:v=d
            else:
                A,B,_=ph.solve_coefficients(float(d[0]),float(q@d),-.8,1.)
                v=d+torch.tensor([A,0.,0.],dtype=torch.float64)+B*q
            expected+=row['aggregation_weight']*v
        torch.testing.assert_close(torch.cat(list(result[0].values())),expected,rtol=0,atol=1e-14)
        without=self.case(reference=[-.8,.6,0.],diagnostics=False)[1]
        self.assertTrue(all(torch.equal(result[0][n],without[0][n]) for n in result[0]))
        with tempfile.TemporaryDirectory() as tmp:
            server=SimpleNamespace(args=SimpleNamespace(),cur_ground=1,projection_history_cache=cache,save_folder_name=tmp)
            before=cache.state()
            with patch.object(torch,'save',side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):ph.commit_success(server,result[-1],{})
            self.assertEqual(cache.state()['source_loop_rounds'],before['source_loop_rounds'])
            ph.commit_success(server,result[-1],{})
            loaded=torch.load(Path(tmp)/'projection_history_cache.pt',weights_only=False)
            self.assertEqual(loaded['source_loop_rounds'],[0,1])

    def test_validation(self):
        good=dict(algorithm='FedTargetProj',target_proj_mode=ph.MODE,model_family='Decom_CNN-5-512')
        ph.validate(SimpleNamespace(**good))
        for key,value in [('projection_history_window',0),('projection_history_window',1.5),
            ('projection_history_decay',-1.),('projection_history_decay',float('nan')),
            ('projection_history_strength',-1.),('projection_history_strength',float('inf')),
            ('target_head_mode','local'),('target_local_schedule','fedrep'),('target_prox_mu',.1),
            ('target_z2_lambda',1.),('target_head_l2_lambda',1.),('target_c0_replay_mode','batch_pair'),
            ('join_ratio',.5),('meta_collect_snapshots',True),('multi_svd',True),('cnn512_control',True)]:
            with self.assertRaises(ValueError,msg=str((key,value))):ph.validate(SimpleNamespace(**dict(good,**{key:value})))
        for decay in (0.,.8,1.):ph.validate(SimpleNamespace(**good,projection_history_decay=decay))
        with self.assertRaises(ValueError):ph.validate(SimpleNamespace(target_proj_mode='projection_softmax',projection_history_strength=0.))


if __name__=='__main__':unittest.main()
