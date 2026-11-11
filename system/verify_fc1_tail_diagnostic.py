"""CPU synthetic acceptance: real N20/K1 FA02 main -> paired offline A/B.

Uses ONLY a newly-created output fixture. Source has two real synthetic rounds,
so labels are offline R3 from R2; it is never misreported as actual FA02 R101.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def execute(command,log,expect_success=True):
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=ROOT/'system',env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
    if (result.returncode==0)!=expect_success:raise RuntimeError(f'Unexpected acceptance exit status: {log}')


def reference(out,server):
    import torch
    from fc1_tail_diagnostic import parser
    from utils import fc1_tail_diagnostic as d
    from utils import single_avg_proto as p
    from utils.cnn512_reporting import model_digest
    from flcore.clients.clientMultiGroupProj import clientMultiGroupProj
    from utils.head_snapshot import tensor_fingerprint
    from common.client_data import read_image_client
    import numpy as np
    import random
    cli=parser().parse_args(['--server-file',str(server),'--source-round','2','--data-root',str(out/'data'),
        '--data-subdir','fixture','--client-ids','0,2','--device','cpu','--threads','2','--output-dir',str(out/'unused_reference')])
    torch.set_num_threads(2)
    mother,payload,args,bank,receipt,manifest,source=d.load_source(cli)
    default_cli=copy.copy(cli);default_cli.client_ids=None
    default_values=d.load_source(default_cli)
    assert default_values[-1]['selected_client_ids']==[0,2,5,7,10,12,15,17]
    assert default_values[-1]['minimum_actual_fc1_rank']==120
    del default_values
    source_code=subprocess.check_output(['git','show','97dc436:system/utils/single_avg_proto.py'],cwd=ROOT,text=True)
    namespace={'__name__':'frozen_97dc436_single_avg_proto'}
    exec(compile(source_code,'97dc436/single_avg_proto.py','exec'),namespace)
    results=[]
    random.seed(args.seed);np.random.seed(args.seed);torch.random.default_generator.manual_seed(args.seed)
    for cid in (0,2):
        data=read_image_client(args,cid,True);rate=args.multi_svd_effective_ratios[args.multi_svd_effective_assignment[cid]]
        local,evidence=d.download_once(mother,rate)
        with tempfile.TemporaryDirectory(dir=out,prefix='frozen_reference_') as folder:
            c=d.local_client(local,mother,args,cid,bank,receipt,data,folder)
            # The actual old public client loader, without indexing or observer.
            c.load_train_data=lambda c=c:clientMultiGroupProj.load_train_data(c)
            namespace['train'](c)
            row=dict(client_id=cid,final_model_sha256=tensor_fingerprint(local.state_dict()),
                final_rng_sha256=d.rng_fingerprint(d.rng_snapshot(local)),
                local_proto_packet_sha256=d.packet_sha(c.avg_proto_local),training=c.train_stats[0])
            results.append(row)
    (out/'frozen_reference.json').write_text(json.dumps(results,indent=2))


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--reference-worker',action='store_true')
    p.add_argument('--resume-owned-output',action='store_true',help='Finish checks in this verifier\'s complete synthetic source workspace only')
    args=p.parse_args();out=Path(args.output).resolve()
    if args.reference_worker:
        server=next((out/'full1/f').rglob('Server_model_0.pt'));return reference(out,server)
    if args.resume_owned_output:
        jobs=json.loads((out/'jobs.json').read_text())
        if jobs!=dict(full1=dict(method='full_avg',proto=True)) or not (out/'full1/verification.json').is_file():
            raise ValueError('Not a complete owned synthetic FA02 source workspace')
    else:out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    rng=np.random.default_rng(94);data=out/'data/Cifar100/fixture'
    if not args.resume_owned_output:
        for split in ('train','test'):
            (data/split).mkdir(parents=True)
            for cid in range(20):
                n=2+cid%3 if split=='train' else 1+cid%4
                x=(rng.normal(size=(n,3,32,32))*.2).astype('float32');x[:,0,0,0]=np.arange(n)/10
                np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)//2+cid)%8))
        (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
        (out/'jobs.json').write_text(json.dumps(dict(full1=dict(method='full_avg',proto=True))))
        print('RUN real main N20/K1/2 rounds source',flush=True)
        execute([sys.executable,str(ROOT/'system/verify_rank_partition.py'),'--output',str(out),'--worker','full1'],out/'source_main.log')
    server=next((out/'full1/f').rglob('Server_model_0.pt'))
    common=[sys.executable,str(ROOT/'system/fc1_tail_diagnostic.py'),'--server-file',str(server),'--source-round','2',
        '--data-root',str(out/'data'),'--data-subdir','fixture','--client-ids','0,2','--device','cpu','--threads','2']
    for label,extra in [('paired',['--save-models']),('quiet',['--diagnostics','0'])]:
        if args.resume_owned_output and (out/label/'manifest.json').is_file():
            if json.loads((out/label/'manifest.json').read_text())['status']!='complete':raise ValueError('Existing partial diagnostic run; use a new output workspace')
        else:
            print('RUN',label,flush=True)
            execute(common+extra+['--output-dir',str(out/label)],out/(label+'.log'))
    if not args.resume_owned_output or not (out/'frozen_reference.json').is_file():
        print('RUN frozen 97dc436 A reference',flush=True)
        execute([sys.executable,__file__,'--output',str(out),'--reference-worker'],out/'reference.log')
    read=lambda path:json.loads((out/path).read_text())
    paired=read('paired/local_training.json');quiet=read('quiet/local_training.json');old=read('frozen_reference.json')
    for left,right in zip(paired,quiet):
        for key in ('final_model_sha256','final_rng_sha256','local_proto_packet_sha256','all_batch_indices_sha256',
            'train_batches','processed_train_samples','ce_loss','regularization_loss','raw_mse','total_loss'):
            assert left[key]==right[key],(key,left['client_id'],left['branch'])
    for expected in old:
        actual=next(r for r in paired if r['client_id']==expected['client_id'] and r['branch']=='A')
        for key in ('final_model_sha256','final_rng_sha256','local_proto_packet_sha256'):assert actual[key]==expected[key],key
        for key in ('ce_loss','regularization_loss','raw_mse','total_loss','processed_train_samples','train_batches'):
            assert actual[key]==expected['training'][key],key
    assert all(r['batch_order_equal'] and r['training_RNG_equal'] and r['SGD_steps_per_branch']==5 for r in read('paired/client_pairs.json'))
    assert len(read('paired/first_step_gradients.json'))==4 and len(read('paired/first_batch_CE_signal.json'))==4
    updates=read('paired/fixed_slot_updates.json');assert len(updates)==12
    assert {r['stage'] for r in updates}=={'step_1','epoch_1','epoch_5'}
    assert len(read('paired/endpoints.json'))==24
    import torch
    from utils.multi_group_svd import load_checkpoint
    from utils.fc1_tail_diagnostic import evaluate
    from common.client_data import read_image_client
    from types import SimpleNamespace
    torch.set_num_threads(2)
    source_config=read('paired/manifest.json')['source']['source_config']
    data_args=SimpleNamespace(**source_config);data_args.data_root=str(out/'data');data_args.data_subdir='fixture'
    for path in (out/'paired/models').glob('*.pt'):
        model,payload=load_checkpoint(path)
        row=next(r for r in paired if r['client_id']==payload['client_id'] and r['branch']==payload['offline_diagnostic']['branch'])
        from utils.head_snapshot import tensor_fingerprint
        assert tensor_fingerprint(model.state_dict())==row['final_model_sha256']
        assert torch.isfinite(model(torch.zeros(1,3,32,32))).all()
        metrics,_=evaluate(model,read_image_client(data_args,payload['client_id'],False),16,'cpu')
        endpoint=next(r for r in read('paired/endpoints.json') if r['client_id']==payload['client_id'] and
            r['branch']==payload['offline_diagnostic']['branch'] and r['stage']=='epoch_5' and r['split']=='test')
        for key in ('correct','samples','accuracy','ce'):assert metrics[key]==endpoint[key],key
    assert len(list((out/'paired/models').glob('*.pt')))==4
    # Type/round/data/source mismatch must fail before training, not silently fallback.
    execute(common+['--source-round','101','--output-dir',str(out/'bad_round_output')],out/'bad_round.log',False)
    execute([sys.executable,str(ROOT/'system/fc1_tail_diagnostic.py'),'--server-file',str(out/'missing/Server_model_0.pt'),
        '--device','cpu','--output-dir',str(out/'bad_missing_output')],out/'bad_missing.log',False)
    bad=out/'bad_packet';bad.mkdir(exist_ok=True)
    if not (bad/'Server_model_0.pt').exists():
        try:os.link(server,bad/'Server_model_0.pt')
        except OSError:shutil.copy2(server,bad/'Server_model_0.pt')
    shutil.copy2(server.parent/'manifest.json',bad/'manifest.json')
    packet=bytearray((server.parent/'Global_class_sample_latest_generated.packet').read_bytes());packet[-1]^=1
    (bad/'Global_class_sample_latest_generated.packet').write_bytes(packet)
    altered=[str(bad/'Server_model_0.pt') if a==str(server) else a for a in common]
    execute(altered+['--output-dir',str(out/'bad_packet_output')],out/'bad_packet.log',False)
    wrong=out/'wrong_data'
    if not wrong.exists():shutil.copytree(out/'data',wrong)
    file=wrong/'Cifar100/fixture/train/0.npz';v=np.load(file,allow_pickle=True)['data'].tolist();v['x'][0,0,0,0]+=1
    np.savez(file,data=v)
    altered=[str(wrong) if a==str(out/'data') else a for a in common]
    execute(altered+['--output-dir',str(out/'bad_data_output')],out/'bad_data.log',False)
    # The CLI default saves only tables; no final/transient model survives in quiet.
    assert not list((out/'quiet').rglob('*.pt')) and not list((out/'paired').glob('transient_training_*'))
    result=dict(status='PASS_CPU',synthetic_only=True,source_actual_rounds=2,offline_round_label=3,
        source_N=20,diagnosed_clients=[0,2],rank_rates=[.90,.35],actual_fc1_ranks=[720,280],minimum_rank=120,
        local_epochs=5,paired_SGD_steps_per_client_branch=5,
        frozen_97dc436_A_parameters_losses_prototype_packets_RNG_exact=True,
        diagnostics_on_off_parameters_losses_prototype_packets_batches_RNG_exact=True,
        AB_batches_RNG_initial_predictions_equal=True,finite_metrics=True,models_reloaded=4,
        invalid_round_packet_missing_input_and_data_rejected=True,
        default_cohorts_four_point35_and_four_point90_verified=True,
        reloaded_endpoint_correct_samples_accuracy_CE_exact=True,
        production_default_model_export=False,real_accuracy_or_hypothesis_verified=False)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':main()
