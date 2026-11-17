"""Isolated N20 actual-main R1/R2 smoke; no real data or formal training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker');p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve()
    if a.worker:
        from verify_svd_five_round import worker
        return worker(out,a.worker)
    jobs={name:dict(case=name,rounds=2,diagnostics=1,epoch_diagnostics=1,command_file='svd_first_band_commands.sh') for name in ('S120','S200','H','H_first_band')}
    jobs.update(H_epoch_off=dict(jobs['H'],epoch_diagnostics=0),
        frozen_H=dict(jobs['H'],epoch_diagnostics=0,frozen=True,reference='b49c54e',freeze_trainer=True),
        all_tail=dict(jobs['H_first_band'],epoch_diagnostics=0,override_spectral_mode='all_tail'))
    if a.resume:
        if json.loads((out/'jobs.json').read_text())!=jobs:raise ValueError('Wrong owned fixture workspace')
    else:
        out.mkdir(parents=True,exist_ok=False)
        import numpy as np
        gen=np.random.default_rng(617);data=out/'data/Cifar100/fixture'
        for split in ('train','test'):
            (data/split).mkdir(parents=True)
            for cid in range(20):
                n=2+cid%3 if split=='train' else 1+cid%4
                x=(gen.normal(size=(n,3,32,32))*.1).astype('float32');x[:,0,0,0]=np.arange(n)/10
                np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)+cid)%6))
        (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
        (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    for label in jobs:
        if a.resume and (out/label/'verification.json').is_file():continue
        print('RUN',label,flush=True)
        with (out/(label+'.log')).open('w',encoding='utf-8') as stream:
            r=subprocess.run([sys.executable,'-u',__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{label} exit={r.returncode}, see {out}/{label}.log')
        print('PASS',label,flush=True)
    reports={name:json.loads((out/name/'verification.json').read_text()) for name in jobs}
    for left,right in [('H','H_epoch_off'),('H_epoch_off','frozen_H')]:
        for field in ('events','batches','primary','final_rng'):assert reports[left][field]==reports[right][field],(left,right,field)
    prefix=lambda rows:[r for r in rows if (r.get('loop',-1) if isinstance(r,dict) else r[0])==0]
    assert prefix(reports['H']['events'])==prefix(reports['H_first_band']['events'])
    for cid in (4,9,14,19):
        left=next(r for r in reports['H']['events'] if r.get('kind')=='download' and r.get('loop')==1 and r.get('cid')==cid)
        right=next(r for r in reports['H_first_band']['events'] if r.get('kind')=='download' and r.get('loop')==1 and r.get('cid')==cid)
        assert left==right  # R1 Avg is identical, hence the SAME actual R2 mother.
    for case,expected in [('S120',dict(conv2=12,fc1=120,fc2=77)),('S200',dict(conv2=20,fc1=200,fc2=128))]:
        m=json.loads((Path(reports[case]['report_dir'])/'manifest.json').read_text())
        for capacities in m['client_capacity_metadata'].values():
            assert {n:r['rank'] for n,r in capacities[0]['layers'].items()}==expected
    import struct
    root=Path(reports['H_first_band']['report_dir']);m=json.loads((root/'manifest.json').read_text())
    counts={r['client_id']:r['count'] for r in m['data_manifest']['files'] if r['split']=='train'}
    spectra=json.loads((root/'five_layer_spectra.json').read_text());expected={}
    for s in spectra:
        if s['stage']!='server_original_download' or s['loop_round']!=1:continue
        name=s['layer'];ranks={int(i):c[0]['layers'][name]['rank'] for i,c in m['client_capacity_metadata'].items()};k,b=sorted(set(ranks.values()))[:2]
        raw=s['singular_values'];values=list(raw)
        for position in range(k,b):
            owners=[i for i,r in ranks.items() if r>=position+1];q=sum(counts[i] for i in owners)/sum(counts.values())
            candidate=raw[position]/q;clamped=candidate>values[position-1]
            values[position]=min(candidate,values[position-1])
            corrected=struct.unpack('f',struct.pack('f',values[position]))[0]
            expected[name,position+1]=(raw[position],corrected,q,owners,clamped)
    rows=json.loads((root/'svd_first_band_ranks.json').read_text())
    assert len(rows)==20*(8+80+51)
    for row in rows:
        raw,corrected,q,owners,clamped=expected[row['layer'],row['rank']]
        assert row['original']==raw and row['corrected']==corrected and row['contributor_mass']==q
        assert row['contributor_ids']==owners and row['contributor_count']==len(owners) and row['nonincreasing_limited']==clamped
    assert len({r['initialization_sha256'] for r in reports.values()})==1 and len({r['data_fingerprint'] for r in reports.values()})==1
    assert all(reports[name]['batches']==reports['H']['batches'] for name in jobs)
    from compare_svd_first_band import comparisons
    contrasts=comparisons({name:reports[name]['report_dir'] for name in ('S120','S200','H','H_first_band')})
    (out/'comparisons.json').write_text(json.dumps(contrasts,indent=2))
    result=dict(status='PASS_CPU_SYNTHETIC_ONLY',N=20,K=1,rounds=2,local_epochs=5,formal_cases=4,
        case_names=['S120','S200','H','H_first_band'],verification_jobs=len(jobs),SGD_steps_per_job=200,native_download_SVD_calls_per_job=120,
        same_data_and_initialization=True,all_batches_paired=True,H_H_first_band_R1_download_upload_Avg_RNG_exact=True,
        epoch_diagnostics_vs_off_parameters_upload_Avg_batches_RNG_exact=True,none_vs_frozen_b49c54e_exact=True,
        old_all_tail_actual_main_regression=True,weighted_metrics_CSV_JSON_H5_and_21_model_reloads_per_job=True,
        complete_epoch_0_to_5_for_8_clients=True,extra_SGD_steps=0,comparison_rows=len(contrasts),
        actual_S120_S200_integer_ranks_verified=True,R2_same_mother_minimum_clients_exact=True,
        R2_all_layers_rank_rows_independently_recomputed_from_actual_counts=True,
        real_accuracy_verified=False,CUDA_tested=False)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':main()
