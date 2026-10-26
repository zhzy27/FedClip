"""Short synthetic real-architecture CLI acceptance, not an accuracy experiment."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from common.experiment import sha256,write_json,environment
from utils.multi_group_svd import model_capacity,load_checkpoint
from utils.offline_distillation import observe,MODES
from flcore.trainmodel.models import Hyper_CNN_512
from compare_offline_distillation import compare_runs


def verify(args):
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);records=[]
    # Owned, isolated temporary fixtures only. Original datasets/checkpoints
    # are never opened or modified by this synthetic verifier.
    with tempfile.TemporaryDirectory(prefix='synthetic_',dir=output) as temp:
        root=Path(temp);dataset=root/'data/Cifar100/pat_20';gen=np.random.default_rng(31)
        for split,count in [('train',7),('test',5)]:
            (dataset/split).mkdir(parents=True)
            np.savez(dataset/split/'2.npz',data=dict(x=gen.normal(size=(count,3,32,32)).astype('float32'),y=np.arange(count)%100))
        (dataset/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat')))
        paths=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            full=Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=16,ratio_LR=1.)
            for k in range(3):
                model=Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=16,ratio_LR=.9)
                path=root/f'Client_2_model_{k}.pt';paths.append(path)
                torch.save(dict(model_class='flcore.trainmodel.models.Hyper_CNN_512',
                    model_family='Decom_CNN-5-512',model_kwargs=dict(in_features=3,num_classes=100,n_kernels=16,ratio_LR=.9,input_size=32),
                    state_dict=model.state_dict(),state_schema={n:list(p.shape) for n,p in model.state_dict().items()},
                    capacity=model_capacity(model,dict(full.named_parameters())),role='ordinary_post_local_low_rank',
                    prediction_checkpoint=True,client_id=2,model_id=k,assigned_model_id=1,completed_train_count=201,
                    loop_round=200,class_order=list(range(100))),path)
            del full,model
        original_hashes=[sha256(p) for p in paths]
        for name,coefficient in [('positive_kd',1),('zero_kd',0)]:
            destination=output/name
            command=[sys.executable,'-u',str(Path(__file__).with_name('offline_distillation.py')),
                '--checkpoints',*map(str,paths),'--client-id','2','--assigned-model-id','1','--completed-round','201',
                '--data-root',str(root/'data'),'--mode','all','--epochs',str(args.epochs),'--batch-size','4',
                '--test-batch-size','4','--teacher-batch-size','4','--lr','.005','--f-lambda','.001',
                '--grad-clip','10','--temperature','2','--kd-lambda',str(coefficient),'--seed','7',
                '--device',args.device,'--device-id',str(args.device_id),'--threads','2',
                '--verify-cache','--output-dir',str(destination)]
            print(f'Running synthetic {name}: 3 modes x {args.epochs} epochs, original width16/rank.9 CNN',flush=True)
            with (output/f'{name}_cli.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            summaries=[json.loads((destination/m/'summary.json').read_text()) for m in MODES]
            if len({s['batch_order_sha256'] for s in summaries})!=1: raise AssertionError('Batch mismatch')
            if len({s['student_initial_state_sha256'] for s in summaries})!=1: raise AssertionError('Epoch0 mismatch')
            if len({s['final_training_rng_sha256'] for s in summaries})!=1: raise AssertionError('Training RNG mismatch')
            states=[]
            test_raw=np.load(dataset/'test/2.npz',allow_pickle=True)['data'].tolist()
            from common.data_io import decode_image_data,TENSOR_LIST
            test_data=decode_image_data(test_raw,list_mode=TENSOR_LIST)
            for mode,summary in zip(MODES,summaries):
                device=f'cuda:{args.device_id}' if args.device=='cuda' else 'cpu'
                model,payload=load_checkpoint(destination/mode/'Student_A.pt',device=device)
                states.append(copy.deepcopy(model.state_dict()))
                endpoint=observe(model,test_data,argparse.Namespace(device=device,test_batch_size=4))
                expected=json.loads((destination/mode/'test_predictions.json').read_text())[-1]['predictions']
                if endpoint['predictions'].tolist()!=expected: raise AssertionError('Reload prediction mismatch')
                if payload['model_id']!=1 or model.ratio_LR!=.9: raise AssertionError('Assignment/capacity mismatch')
                if summary['actual_sample_visits']!=7*args.epochs or summary['actual_optimizer_steps']!=2*args.epochs:
                    raise AssertionError('Incorrect training budget')
            errors=[]
            for other in states[1:]:
                errors.append(max(float((p-other[n]).abs().max()) for n,p in states[0].items()))
            if coefficient==0 and errors!=[0.,0.]: raise AssertionError('Zero KD did not exactly reduce to CE+F')
            if [sha256(p) for p in paths]!=original_hashes: raise AssertionError('Input checkpoints changed')
            compare_runs([destination],output/f'{name}_comparison')
            records.append(dict(case=name,coefficient=coefficient,epochs=args.epochs,client_id=2,assigned_model_id=1,
                architecture='actual Hyper_CNN_512 width16, 100 classes, rank_rate .9',
                student_parameter_count=sum(p.numel() for p in model.parameters()),
                equal_batch_orders=True,equal_training_rng=True,equal_epoch0_states=True,teachers_unchanged=True,
                input_files_unchanged=True,reload_predictions_exact=True,
                maximum_state_difference_vs_continue=errors,
                cache_verification=json.loads((destination/'teacher_cache.json').read_text())['direct_verification']))
            if not args.keep_models:
                # Single, explicitly owned files only; retain all readable
                # metrics/provenance and free bulky synthetic model/cache data.
                for mode in MODES: (destination/mode/'Student_A.pt').unlink()
                for cache in (destination/'cache').glob('teacher_logits_*.pt'): cache.unlink()
    actual=json.loads((output/'positive_kd/manifest.json').read_text())['environment']
    report=dict(status='passed',kind='synthetic_only_no_accuracy_conclusion',cases=records,environment=actual,verifier_environment=environment(),
        synthetic_binary_artifacts_retained=args.keep_models)
    write_json(output/'acceptance.json',report);print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--device-id',type=int,default=0)
    parser.add_argument('--keep-models',action='store_true')
    args=parser.parse_args()
    if args.epochs<1: parser.error('epochs must be positive')
    verify(args)
