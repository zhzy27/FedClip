"""Small actual-main width/save/reload and actual-sweep CLI synthetic acceptance."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def fixture(out):
    import numpy as np
    gen=np.random.default_rng(625);base=out/'data/Cifar100/fixture'
    for split in ('train','test'):
        (base/split).mkdir(parents=True)
        for cid in range(2):
            n=3+cid if split=='train' else 2+cid
            np.savez(base/split/f'{cid}.npz',data=dict(x=(gen.normal(size=(n,3,32,32))*.1).astype('float32'),y=(np.arange(n)+cid)%3))
    (base/'config.json').write_text(json.dumps(dict(num_clients=2,num_classes=100,partition='pat',class_per_client=20)))


def train(out,width):
    text=(ROOT/'system/svd_five_round_commands.sh').read_text()
    line=next(l.strip()[1:-1] for l in text.splitlines() if l.strip().startswith('"') and '/H/checkpoints' in l)
    words=shlex.split(line)[3:];base=out/str(width)
    for flag,value in [('-did',0),('-dev','cpu'),('-nc',2),('-nsm',1),('-gr',0),('-ls',1),('-lbs',2),
        ('--multi_group_sizes','2'),('--multi_svd_ratios','.90,.15'),('--multi_svd_five_diagnostics',0),
        ('--data-root',out/'data'),('--data-subdir','fixture'),('-clip_cpu_threads',2),('-exp_name',f'width_{width}_synthetic'),
        ('-sfn',base/'checkpoints'),('--run_log_dir',base/'logs'),('--h5_result_root',base/'h5'),('--final-model-root',base/'final_models')]:
        words[words.index(flag)+1]=str(value)
    words+=['--cnn_hidden_width',str(width)];sys.argv=['main.py']+words
    runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')


def sweep(out,width,device='cpu',source_fixture=None):
    import torch
    from utils import svd_capacity_sweep as d
    source=out if source_fixture is None else source_fixture
    calls=[];native=d.shared_download;original=torch.linalg.svd
    def shared(*a,**kw):
        before_dtype=torch.get_default_dtype()
        with patch.object(torch.linalg,'svd',wraps=original) as observed:result=native(*a,**kw)
        assert torch.get_default_dtype()==before_dtype
        assert all(p.dtype==torch.float32 for p in result[0].parameters())
        calls.append(observed.call_count)
        assert all(call.args[0].device.type==device for call in observed.call_args_list)
        assert all(call.args[0].dtype==(torch.float64 if device=='cuda' else torch.float32) for call in observed.call_args_list)
        assert all(call.kwargs.get('driver')==('gesvd' if device=='cuda' else None) for call in observed.call_args_list)
        if device=='cpu':assert all('driver' not in call.kwargs for call in observed.call_args_list)
        return result
    def forbidden(*a,**kw):raise AssertionError('Offline sweep must never optimize or backward')
    sys.argv=['svd_capacity_sweep.py','-did','0','--device',device,'--source-name',str(width),'--source-run',str(source/str(width)),
        '--completed-round','1','--client-ids','0','--data-root',str(source/'data'),'--data-subdir','fixture',
        '--threads','2','--output-dir',str(out/f'sweep_{width}')]
    with patch.object(d,'shared_download',side_effect=shared),patch.object(torch.optim.SGD,'step',side_effect=forbidden),patch.object(torch.Tensor,'backward',side_effect=forbidden):
        runpy.run_path(str(ROOT/'system/svd_capacity_sweep.py'),run_name='__main__')
    assert calls==[3],calls


def compare(out):
    sys.argv=['svd_capacity_sweep.py','--device','cpu','--compare-results',str(out/'sweep_2000'),str(out/'sweep_1536'),
        '--output-dir',str(out/'paired_complete')]
    runpy.run_path(str(ROOT/'system/svd_capacity_sweep.py'),run_name='__main__')


def verify(out,device='cpu'):
    import csv
    from PIL import Image
    for width in (2000,1536):
        root=out/f'sweep_{width}';m=json.loads((root/'manifest.json').read_text());rows=json.loads((root/'client_results.json').read_text())
        assert m['status']=='complete' and m['input_files_unchanged'] and m['hidden_width']==width
        assert m['svd_driver']==('gesvd' if device=='cuda' else None)
        evidence=json.loads((root/'shared_svd.json').read_text())
        assert len(evidence)==1 and evidence[0]['native_download_SVD_calls']==3
        assert evidence[0]['svd_driver']==m['svd_driver']
        assert evidence[0]['svd_compute_dtype']==('torch.float64' if device=='cuda' else 'torch.float32')
        assert evidence[0]['prediction_dtype']=='torch.float32'
        assert len(evidence[0]['layers'])==3 and all(r['svd_driver']==m['svd_driver'] for r in evidence[0]['layers'])
        assert len(rows)==7 and rows[0]['variant']=='native' and rows[1]['variant']=='resvd' and rows[1]['rank_rate']==.9
        assert rows[0]['parameter_count']==(3246372 if width==2000 else 2697924)
        assert len({r['sample_order_sha256'] for r in rows})==1
        for file in root.glob('*.csv'):
            with file.open(encoding='utf-8',newline='') as f:assert len(list(csv.DictReader(f)))==len(json.loads(file.with_suffix('.json').read_text()))
        checks=json.loads((root/'roundtrip_checks.json').read_text());assert checks[0]['passed'] and checks[0]['source_model_and_RNG_unchanged']
        with Image.open(root/'parameters_vs_drop.png') as image:assert image.size==(1500,900)
    paired=out/'paired_complete';contrasts=json.loads((paired/'paired_comparisons.json').read_text())
    assert len(contrasts)==6 and all(r['client_ids']==[0] for r in contrasts)
    with Image.open(paired/'parameters_vs_drop.png') as image:assert image.size==(1500,900)
    r=dict(status='PASS_'+device.upper()+'_SYNTHETIC_ONLY',actual_main_widths=[2000,1536],clients_per_main=2,rounds_per_main=1,local_epochs=1,
        actual_sweep_sources=2,native_reference_separate_from_roundtrip=True,each_checkpoint_three_shared_download_SVDs=True,
        source_models_RNG_and_input_files_unchanged=True,offline_backward_or_SGD_forbidden=True,
        default_legacy_checkpoint_missing_width_and_narrow_saved_width_verified=True,plots_CSV_JSON_verified=True,paired_compare_CLI_verified=True,
        CUDA_tested=device=='cuda',svd_driver='gesvd' if device=='cuda' else None,
        recovery_SVD_dtype='torch.float64' if device=='cuda' else 'torch.float32',prediction_dtype='torch.float32',default_dtype_restored=True,
        real_L_H_checkpoints_evaluated=False,accuracy_benefit_claimed=False)
    (out/'acceptance_report.json').write_text(json.dumps(r,indent=2));print('PASS',out/'acceptance_report.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--phase',choices=['fixture','train2000','train1536','sweep2000','sweep1536','compare','verify'])
    p.add_argument('--sweep-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--source-fixture',help='Reuse an existing actual-main synthetic fixture; skip fixture creation/training')
    a=p.parse_args();out=Path(a.output).resolve()
    if a.phase:
        if a.phase.startswith('train'):return train(out,int(a.phase[5:]))
        if a.phase.startswith('sweep'):return sweep(out,int(a.phase[5:]),a.sweep_device,Path(a.source_fixture).resolve() if a.source_fixture else None)
        if a.phase=='verify':return verify(out,a.sweep_device)
        return globals()[a.phase](out)
    out.mkdir(parents=True,exist_ok=False)
    phases=('sweep2000','sweep1536','compare','verify') if a.source_fixture else ('fixture','train2000','train1536','sweep2000','sweep1536','compare','verify')
    for phase in phases:
        print('RUN',phase,flush=True)
        with (out/(phase+'.log')).open('w',encoding='utf-8') as f:
            command=[sys.executable,'-u',__file__,'--output',str(out),'--phase',phase,'--sweep-device',a.sweep_device]
            if a.source_fixture:command+=['--source-fixture',str(Path(a.source_fixture).resolve())]
            r=subprocess.run(command,cwd=ROOT/'system',
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if a.sweep_device=='cuda' and phase.startswith('sweep') else '-1',
                    PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{phase} exit={r.returncode}, see {out}/{phase}.log')
        print('PASS',phase,flush=True)


if __name__=='__main__':main()
