"""Independent continue-CE / mean-probability KD / fixed KL+DP experiment."""
import argparse
import copy
from contextlib import redirect_stdout, redirect_stderr
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from common.experiment import rng_state, write_json, environment, code_version
from utils.cnn512_control import seed_control
from utils.offline_distillation_io import Bundle, load_sources, read_data, digest_json
from utils.offline_distillation import (MODES,preserved_rng,freeze_teachers,teacher_cache,selection_report,
    teacher_identity,assert_frozen,observe,train_mode,write_csv,sync)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--bundle',help='Original zip or extracted directory; reads exact models/Client_ID_model_K.pt members')
    src.add_argument('--checkpoints',nargs=3,help='Three explicit original prediction checkpoints, in model-ids order')
    p.add_argument('--model-ids',type=int,nargs=3,default=[0,1,2])
    p.add_argument('--client-id',type=int,default=0)
    p.add_argument('--assigned-model-id',type=int)
    p.add_argument('--completed-round',type=int)
    p.add_argument('--group-file');p.add_argument('--class-order',help='Optional JSON list; otherwise original integer class-index order')
    p.add_argument('--data-root',default='../dataset');p.add_argument('--dataset',default='Cifar100');p.add_argument('--data-subdir',default='pat_20')
    p.add_argument('--train-file');p.add_argument('--test-file');p.add_argument('--data-config');p.add_argument('--data-manifest')
    p.add_argument('--test-from-bundle',action='store_true',help='Explicitly use bundled original test shard')
    p.add_argument('--native-test-only',action='store_true',help='No training/data synthesis: evaluate frozen originals, permits missing train shard')
    p.add_argument('--mode',choices=(*MODES,'all'),default='all')
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--test-batch-size',type=int,default=16);p.add_argument('--teacher-batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=.005);p.add_argument('--momentum',type=float,default=0.);p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--f-regular',type=int,choices=[0,1],default=1);p.add_argument('--f-lambda',type=float,default=.001)
    p.add_argument('--grad-clip',type=float,default=10.,help='0 explicitly disables norm clipping')
    p.add_argument('--temperature',type=float,default=2.);p.add_argument('--kd-lambda',type=float,default=1.)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--device-id',type=int,default=0);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--deterministic',type=int,choices=[0,1],default=1)
    p.add_argument('--cache-dir',help='Verified immutable teacher logits; default OUTPUT/cache')
    p.add_argument('--verify-cache',action='store_true',help='Compare entire cache with a fresh protected teacher pass (atol=1e-6,rtol=1e-5)')
    p.add_argument('--output-dir',required=True,help='New independent output directory; existing output is refused')
    return p


def validate(args):
    for name in ('epochs','batch_size','test_batch_size','teacher_batch_size','threads'):
        if getattr(args,name)<1: raise ValueError(f'{name} must be a positive integer')
    for name in ('lr','temperature'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0: raise ValueError(f'{name} must be finite and positive')
    for name in ('f_lambda','grad_clip','kd_lambda','weight_decay','momentum'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<0: raise ValueError(f'{name} must be finite and nonnegative')
    if args.momentum>=1: raise ValueError('SGD momentum must be less than 1')
    if not math.isfinite(args.kd_lambda*args.temperature**2): raise ValueError('lambda_KD*T^2 must be finite')
    if not 0<=args.seed<2**32 or args.client_id<0 or args.device_id<0: raise ValueError('Invalid seed/client/device ID')
    if args.completed_round is not None and args.completed_round<1: raise ValueError('completed-round must be positive')
    if args.test_from_bundle and (not args.bundle or args.test_file):
        raise ValueError('test-from-bundle requires bundle and cannot also specify test-file')
    if not args.data_subdir or Path(args.dataset).name!=args.dataset or Path(args.data_subdir).is_absolute() or '..' in Path(args.data_subdir).parts:
        raise ValueError('Use explicit dataset and relative data-subdir; no path discovery')
    output=Path(args.output_dir).resolve()
    dataset=(Path(args.data_root)/args.dataset/args.data_subdir).resolve()
    if output==dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError('Output must be separate from original dataset')
    if args.cache_dir:
        cache=Path(args.cache_dir).resolve()
        if cache==dataset or dataset in cache.parents or cache in dataset.parents:
            raise ValueError('Cache must be separate from original dataset')
    for path in ([args.bundle] if args.bundle else args.checkpoints):
        source=Path(path).resolve()
        if not source.exists(): raise FileNotFoundError(source)
        if source==output or output in source.parents or source.is_dir() and source in output.parents:
            raise ValueError('Output must be separate from input checkpoints/bundle')
        if args.cache_dir and (cache==source or source.is_dir() and source in cache.parents):
            raise ValueError('Cache must be separate from input checkpoints/bundle')
    if args.device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; select --device cpu')
    return output


def _run(args):
    output=Path(args.output_dir).resolve()
    if not args.cache_dir: args.cache_dir=str(output/'cache')
    torch.set_num_threads(args.threads)
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
    if args.device=='cuda': torch.cuda.set_device(args.device_id);args.device=f'cuda:{args.device_id}'
    bundle=Bundle(args.bundle);started=time.perf_counter()
    try:
        with preserved_rng():
            read_started=time.perf_counter()
            entries,source=load_sources(args,bundle)
            data,data_record=read_data(args,bundle,source,train_required=not args.native_test_only)
            input_loading_and_audit_seconds=time.perf_counter()-read_started
            teachers=[e['model'] for e in entries];freeze_teachers(teachers,args.device)
            frozen=teacher_identity(teachers)
            manifest=dict(schema=1,role='offline_distillation_not_federated_training',effective=vars(args),source=source,data=data_record,
                environment=environment(),code=code_version(Path(__file__).resolve().parents[1]),
                selection_rule='fixed requested final epoch, no test-based selection or validation split',
                class_order_limitation=source['class_order_verification'],
                input_loading_and_audit_seconds=input_loading_and_audit_seconds,
                inference_model_count=1,loss_rule='CE + original weighted F + lambda_KD*T^2*KL(q||p_student_T)',
                no_SVD_or_recovery=True,no_federated_feedback=True)
            write_json(output/'manifest.json',manifest)
            if args.native_test_only:
                rows=[]
                for i,m in enumerate(teachers):
                    r=observe(m,data['test'],args);r.pop('predictions')
                    rows.append(dict(role=source['role_order'][i],model_id=source['model_id_order'][i],**r))
                assert_frozen(teachers,frozen);write_json(output/'native_test.json',rows);write_csv(output/'native_test.csv',rows)
                manifest['wall_seconds']=time.perf_counter()-started;write_json(output/'manifest.json',manifest)
                print(json.dumps(rows,indent=2));return dict(native=rows,manifest=manifest)
            cache=teacher_cache(teachers,data['train'],source,data_record['files']['train'],args)
            assert_frozen(teachers,frozen)
            choices=selection_report(cache,source)
            manifest.update(teacher_cache=cache['report'],teacher_selection=choices)
            write_json(output/'teacher_selection_summary.json',choices);write_json(output/'teacher_cache.json',cache['report'])
            choice_rows=[]
            for i,(selected,p) in enumerate(zip(cache['selected'].tolist(),cache['true_probabilities'].tolist())):
                choice_rows.append(dict(sample_id=i,label=int(data['train'][i][1]),selected_role=source['role_order'][selected],
                    selected_model_id=source['model_id_order'][selected],A0_true_probability=p[0],B0_true_probability=p[1],C0_true_probability=p[2],
                    max_external_minus_A0=max(p[1:])-p[0]))
            write_csv(output/'teacher_selection.csv',choice_rows)
            baseline_started=time.perf_counter()
            baseline={split:observe(teachers[0],data[split],args) for split in ('train','test')}
            sync(args.device);manifest['original_A0_baseline_evaluation_seconds']=time.perf_counter()-baseline_started
            write_json(output/'baseline.json',{key:{k:v for k,v in value.items() if k!='predictions'} for key,value in baseline.items()})
            # Reset once after ALL protected preparation, then restore this same state for every independent mode.
            seed_control(args);initial_rng=rng_state();modes=MODES if args.mode=='all' else (args.mode,)
            results={};comparison=[]
            for mode in modes:
                result=train_mode(mode,teachers,data,cache,args,initial_rng);results[mode]=result
                if result['initial_state_sha256']!=source['checkpoints'][0]['state_sha256']:
                    raise RuntimeError('Student epoch0 differs from original A0')
                row0=result['rows'][0]
                for split in ('train','test'):
                    if (row0[f'{split}_correct'],row0[f'{split}_samples'])!=(baseline[split]['correct'],baseline[split]['samples']):
                        raise RuntimeError('Epoch0 accuracy failed to reproduce original A0')
                folder=output/mode;folder.mkdir()
                write_json(folder/'metrics.json',result['rows']);write_csv(folder/'metrics.csv',result['rows'])
                write_json(folder/'batch_order.json',result['batches']);write_json(folder/'test_predictions.json',result['test_predictions'])
                final=result['rows'][-1]
                summary=dict(mode=mode,fixed_final_epoch=args.epochs,selection_rule='fixed_final_epoch_not_best',
                    final_metrics=final,student_initial_state_sha256=result['initial_state_sha256'],
                    batch_order_sha256=result['batch_order_sha256'],final_training_rng_sha256=result['final_rng_sha256'],
                    initial_training_rng_sha256=result['initial_training_rng_sha256'],
                    teachers_unchanged=True,total_train_seconds=sum(r['train_seconds'] for r in result['rows']),
                    total_endpoint_evaluation_seconds=sum(r['evaluation_seconds'] for r in result['rows']),
                    actual_sample_visits=sum(r['sample_visits'] for r in result['rows']),actual_optimizer_steps=sum(r['optimizer_steps'] for r in result['rows']))
                with preserved_rng():
                    saved=copy.deepcopy(result['student']).cpu()
                    torch.save(dict(model_class=entries[0]['payload']['model_class'],model_kwargs=entries[0]['payload']['model_kwargs'],
                        model_family=entries[0]['payload']['model_family'],state_dict=saved.state_dict(),
                        state_schema={n:list(p.shape) for n,p in saved.state_dict().items()},
                        capacity=entries[0]['payload']['capacity'],role='offline_distilled_student',
                        client_id=args.client_id,model_id=source['assigned_model_id'],source_completed_round=source['completed_round'],
                        offline_mode=mode,offline_epochs=args.epochs,source=source,configuration=vars(args)),folder/'Student_A.pt')
                summary['student_file_bytes']=(folder/'Student_A.pt').stat().st_size
                write_json(folder/'summary.json',summary);write_json(folder/'manifest.json',dict(manifest,training=summary))
                comparison.append(dict(mode=mode,epoch=args.epochs,test_accuracy_percent=100*final['test_accuracy'],
                    delta_from_A0_pp=final['test_accuracy_change_pp'],test_ce=final['test_ce'],train_accuracy_percent=100*final['train_accuracy'],
                    train_ce=final['train_ce'],original_wrong_now_correct=final['original_wrong_now_correct'],
                    original_correct_now_wrong=final['original_correct_now_wrong']))
                # Do not keep trained students/optimizer tensors across modes.
                del result['student'],result['optimizer']
            if len({r['batch_order_sha256'] for r in results.values()})!=1: raise RuntimeError('Modes consumed different batch orders')
            assert_frozen(teachers,frozen)
            write_json(output/'comparison.json',comparison);write_csv(output/'comparison.csv',comparison)
            lines=['|mode|fixed epoch|test accuracy %|delta pp|test CE|wrong→right|right→wrong|',
                   '|---|---:|---:|---:|---:|---:|---:|']
            for r in comparison:
                lines.append(f"|{r['mode']}|{r['epoch']}|{r['test_accuracy_percent']:.6f}|{r['delta_from_A0_pp']:.6f}|{r['test_ce']:.8f}|{r['original_wrong_now_correct']}|{r['original_correct_now_wrong']}|")
            (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
            manifest.update(wall_seconds=time.perf_counter()-started,teacher_cache=cache['report'],teacher_selection=choices,
                output_storage_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                all_modes_batch_order_sha256=next(iter(results.values()))['batch_order_sha256'])
            write_json(output/'manifest.json',manifest)
            return dict(comparison=comparison,manifest=manifest,results=results)
    finally:
        bundle.close()


class _Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value);stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def run(args):
    output=validate(args);output.mkdir(parents=True,exist_ok=False)
    with (output/'training.log').open('w',encoding='utf-8') as log:
        with redirect_stdout(_Tee(sys.stdout,log)),redirect_stderr(_Tee(sys.stderr,log)):
            print(f'Independent offline experiment: {output}')
            try: return _run(args)
            except Exception as error:
                write_json(output/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
                print(f'Offline experiment FAILED: {error}')
                raise


def main():
    args=parser().parse_args()
    # A standalone process only. Federated main.py and its defaults are untouched.
    result=run(args)
    print(f'Offline outputs: {Path(args.output_dir).resolve()}')


if __name__=='__main__': main()
