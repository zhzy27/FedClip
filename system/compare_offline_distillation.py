"""Combine independent fixed-final-epoch runs; never select by test accuracy."""
import argparse
import csv
import json
from pathlib import Path


MODES=('continue_ce','kd_mean','kl_dp')
CONFIG_KEYS=('epochs','batch_size','test_batch_size','teacher_batch_size','lr','momentum','weight_decay',
             'f_regular','f_lambda','grad_clip','temperature','kd_lambda','seed','deterministic')


def signature(manifest,summary):
    source=manifest['source'];data=manifest['data']
    return dict(configuration={k:manifest['effective'][k] for k in CONFIG_KEYS},
        source={k:source[k] for k in ('client_id','assigned_model_id','completed_round','model_id_order','class_order_sha256')},
        checkpoints=[r['checkpoint_sha256'] for r in source['checkpoints']],
        data={split:{k:r[k] for k in ('sha256','decoded_x_sha256','decoded_y_sha256','count')}
              for split,r in data['files'].items()},data_config=data['config'],
        initial_state=summary['student_initial_state_sha256'],batches=summary['batch_order_sha256'],
        initial_training_rng=summary['initial_training_rng_sha256'],
        training_rng=summary['final_training_rng_sha256'],
        torch_version=manifest['environment']['packages']['torch'],
        backend={k:manifest['environment'][k] for k in
                 ('deterministic','cudnn_deterministic','cudnn_benchmark','matmul_tf32','cudnn_tf32')})


def compare_runs(runs,output):
    collected={};expected=None
    for root in map(Path,runs):
        folders=[root] if (root/'summary.json').is_file() else [root/m for m in MODES if (root/m/'summary.json').is_file()]
        if not folders: raise ValueError(f'No completed offline mode under {root}')
        for folder in folders:
            summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            mode=summary['mode']
            if mode not in MODES or mode in collected: raise ValueError(f'Unknown or duplicate mode: {mode}')
            if summary['selection_rule']!='fixed_final_epoch_not_best': raise ValueError('Only fixed-final-epoch results can be compared')
            current=signature(manifest,summary)
            if expected is not None and current!=expected:
                raise ValueError('Runs differ in input, training configuration, batch order, RNG, or backend; refusing comparison')
            expected=current;final=summary['final_metrics']
            if final['epoch']!=summary['fixed_final_epoch']: raise ValueError('Incomplete fixed-budget run')
            collected[mode]=dict(mode=mode,epoch=final['epoch'],test_accuracy_percent=100*final['test_accuracy'],
                delta_from_A0_pp=final['test_accuracy_change_pp'],test_ce=final['test_ce'],
                train_accuracy_percent=100*final['train_accuracy'],train_ce=final['train_ce'],
                original_wrong_now_correct=final['original_wrong_now_correct'],original_correct_now_wrong=final['original_correct_now_wrong'],
                result_directory=str(folder.resolve()))
    if set(collected)!=set(MODES): raise ValueError('Need all three distinct modes for comparison')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[collected[m] for m in MODES]
    (output/'comparison.json').write_text(json.dumps(dict(selection_rule='fixed_requested_final_epoch',
        verified_common_signature=expected,results=rows),indent=2,allow_nan=False),encoding='utf-8')
    with (output/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['|mode|fixed epoch|test accuracy %|delta from A0 pp|test CE|wrong→right|right→wrong|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['mode']}|{r['epoch']}|{r['test_accuracy_percent']:.6f}|{r['delta_from_A0_pp']:.6f}|{r['test_ce']:.8f}|{r['original_wrong_now_correct']}|{r['original_correct_now_wrong']}|")
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',help='Run roots or individual mode result directories')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();compare_runs(args.runs,args.output_dir)
    print(f'Fixed-budget comparison: {Path(args.output_dir).resolve()}')
