# C0 two-stage training with server feedback

Three independent `FedTargetProj` modes use the same new loop:

| Mode | First-stage aggregation H | Stage2 / next server |
| --- | --- | --- |
| `feedback_avg` | All clients' ordinary post-local full-W, original sample-count Avg | C0 downloads H, trains local5, server parameters become recovered C0 |
| `feedback_avg_self` | C0 .125, helpers .875 allocated by their relative train counts; weighted-parameter Avg | Same |
| `feedback_projection_self` | Identical .125/.875 count weights; original individual-helper full-model global-delta Projection | Same |

The .125 constant is internal and logged; it is not controlled by
`projection_self_weight` or APA arguments. The two self modes call exactly the
same sample-mass function. No softmax, weight learning, guidance, Prox or meta
holdout is involved. Helpers with positive counts participate regardless of
cosine sign. Existing modes and defaults retain their old code paths.

## Round flow and state

For every loop round 0..200 (`round=loop_round+1`):

1. All 20 clients download the one shared recovered server S, using their original
   capacity and existing SVD/U/V/head and local-buffer download rules.
2. All 20 run ordinary local5, with fresh SGD lr .005, batch16, CE + .001 Frobenius
   regularization and clip norm10. Capture immutable first-stage recovered uploads
   and C0's first-stage low-rank checkpoint, plus independent copied training stats.
3. Construct intermediate H from **this round's** S and first-stage uploads, using
   normal upload order. Avg uses the original weighted-parameter arithmetic;
   Projection reuses the original per-helper coefficients, before the weighted sum.
4. Save H as the shared server, then **only C0** performs a real normal SVD download.
   Observe this actual downloaded C0 checkpoint before training. C0's buffers stay
   local, carrying its stage1 state through this download.
5. C0 runs another ordinary local5 with another fresh optimizer. Observe its
   stage2 checkpoint. These real download/train operations retain all RNG changes.
6. Recover C0 full-W, check the identical named-parameter set including classifier,
   and overwrite **all server parameters** with it. Never mix H or old uploads back
   in. Server buffers remain the original shared server buffers, independent of C0's.
   Next round's 20 clients download this feedback server.

C0 therefore trains 10 epochs per round; each helper trains 5. There is no prewarm
epoch. `client.train_time_cost['num_rounds']` counts training calls (C0 twice per
federated round). Download/evaluation/restore diagnostics do not mutate ordinary
uploads or advance the normal random flow; actual stage2 optimization is not rolled
back as a temporary guidance branch. Stale first-stage caches are rejected using
the explicit source `loop_round`, client ID and stage.

The on-disk first-stage cache uses two bounded slots:
`<checkpoint_run>/feedback_stage1/slot_0|slot_1/`. Each holds all 20 full-W uploads,
the C0 stage1 low-rank model, metadata/stats, starting S, intermediate H and feedback
server. Stage2 never overwrites these ordinary uploads or metadata. A slot is
replaced only when a later round of the same parity captures new stage1 uploads;
this avoids retaining 201 copies of all 20 full models. All rounds' diagnostic and
training statistics remain in CSV/JSON/H5. The first-stage cache does not retain
every historical raw upload indefinitely.

## Evaluation and files

The primary metric is `target_stage2_post_local_acc`. Every round also records
`target_stage1_post_local_acc` and `target_intermediate_download_acc`.
`target_stage2_finetune_gain` subtracts the two endpoints from the same **actual**
H download and ensuing stage2 train. Each series reports final, earliest best
round, last10/last20 and their counts. Machine-readable accuracy fields use the
same [0,1] fractions as existing metrics; console summaries display percent and
finetune gain in percentage points.

Logs contain actual target/helper masses and each client's original/aggregation
weights, raw cosine/negative-dot conflict, actual projection coefficient and
removed norm. `conflict` remains the raw negative-helper-dot diagnostic;
`projection_applied` separately indicates execution. Avg modes have zero actual
coefficients/removal despite any raw conflicts. The two change norms are distinct:
`target_stage1_global_delta_norm = ||W0_stage1 - S||` and
`target_stage2_vs_intermediate_full_delta_norm = ||C - H||`, including SVD truncation
in the recovered parameter space. Every training row identifies stage and client,
with CE, unweighted F penalty, weighted F loss, batches, sample visits, optimizer
time and call wall time. Timings additionally cover cache recovery/writes,
aggregation (including cache reads and intermediate writes), actual C0 stage2
download/SVD, recovered feedback/writes and the total round.

The formal array is in `system/target_feedback_commands.sh`. Source it from
`system/` to define `COMMANDS`; this file starts no processes. The three GPU IDs are
replaceable examples. All commands specify complete CIFAR100/pat_20 data, original
`Decom_CNN-5-512` capacities, seed0, lr .005, batch16, per-stage local5, F .001,
full participation, `-gr 200 -eg 1`, and zero Prox. No extra weight flag is needed.

Independent log roots:

* `system/training_logs/feedback_avg_sample_seed0_gr200/`
* `system/training_logs/feedback_avg_self_self0p125_seed0_gr200/`
* `system/training_logs/feedback_projection_self_self0p125_seed0_gr200/`

Their roots hold `train.log` and `status.json`; `run_<id>/` holds
`feedback_metrics.csv/json`, `feedback_clients.csv`, `feedback_training.csv`,
the experiment JSON and mirrored H5. A sibling log-only ZIP excludes model tensors.
Checkpoint, original H5 and final-model roots are separate and include mode/mass.
H5 has a distinct `target_feedback` group, all three accuracy series, summaries,
per-client weights/projection diagnostics and stage-labelled training arrays.
Final `Server_model.pt` is the feedback server; C0's exported local model is
stage2 and helpers' local models are stage1. The manifest explicitly records this
scope, buffer convention and the 10/5 epoch budget.

Launcher preview (no training):

```bash
python run_target_proj.py --modes feedback_avg feedback_avg_self feedback_projection_self --rounds 200 --device-ids 0 1 2 --dry-run
```

## Verification and interpretation

```bash
python -m unittest discover -s tests -p test_target_feedback.py
python -m unittest discover -s tests -p test_target_feedback_runtime.py
```

Math tests require identical self-mode weights, exact legacy Avg/Projection
arithmetic, analytic weighted head-inclusive sums, agreement without conflicts,
different results with conflicts, finite zero-reference handling, invalid/stale
inputs and correct earliest-best/tail summaries. Runtime smoke exercises all
three modes with actual heterogeneous `Hyper_CNN_512` low-rank parameterization
(.9/.37/.35/.25/.15), 20 clients, 100-class head, normal5+5 and two rounds. It uses
synthetic 32x32 data and reduced convolution width (`n_kernels=4`), and directly
compares actual downloads to normal SVD of H, each next server to previous C,
immutable first uploads/stats, real RNG advancement versus protected observation,
client/server buffers and complete CSV/JSON/H5/final exports.

`feedback_avg_self` vs `feedback_avg` tests first-stage target mass;
`feedback_projection_self` vs `feedback_avg_self` tests Projection within this
same feedback loop. Comparison against old modes must state that C0's budget
increases from 5 to 10 epochs per round. These three tests alone cannot attribute
all improvement to feedback itself; a later equal-budget, no-feedback control
would be necessary. Synthetic smoke supplies no accuracy-improvement evidence.
No full experiment, push, dataset modification or approximate Projection is part
of implementation verification.

Local acceptance (2026-10-14): the three-mode two-round CNN smoke passed on CPU
in 196.6 seconds, using PyTorch 2.0.1+cu117 / torchvision 0.15.2+cu117 (`htfllib`).
This time includes training, extra comparison checks, evaluation and file exports,
and is not a real-data/GPU throughput claim. The 206 existing numerical/runtime
unit checks for target/Projection variants, sample/softmax modes, groups, layer
mask/budget, APA/DWA, Prox, launchers, checkpoint storage and log handling also passed.
With 11 feedback math/options/summary tests and the three-mode smoke test, 218
tests passed in total. Both main CLI choices and the 201-round launcher preview
were checked without starting formal training.
