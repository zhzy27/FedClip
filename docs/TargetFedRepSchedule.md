# FedRep-style alternating local training ablation

This is a training-schedule ablation in the existing heterogeneous low-rank
FedTargetProj framework, not a complete reproduction of the FedRep paper.
The default `--target_local_schedule joint` keeps the existing training path.

The only supported alternating configuration is:

```text
FedTargetProj / avg / target_head_mode=local / target_local_schedule=fedrep
fedrep_head_epochs=5 / -ls 5 (BACKBONE epochs) / Prox=0
```

The first experiment fixes original CIFAR100/pat_20, 20 clients, 100 output
classes, C0, seed0, original Decom_CNN-5-512 capacities, full participation,
SGD lr .005, batch16, momentum0, weight_decay0, F coefficient .001, every-round
evaluation and no LR decay. It uses `-gr 30`, hence 31 inclusive rounds. All
clients use full original local data; nothing is regenerated or split.

Projection, group, feedback, DWA, meta, old `head_diag_collect`, nonzero Prox,
resume/pretraining, truncated epochs, early stopping and alternate initial
budgets are rejected. Main, launcher and server validate the same policy.
The old head-diagnostic collector and offline entry also reject a non-joint
schedule, preventing accidental pairing with old joint snapshots.

## One round

1. Each client normally downloads the shared backbone, preserving its previous
   local fc3. Before round zero only, all fc3 parameters use the same original
   public server initialization, exactly as in local-head joint training.
2. Freeze every backbone parameter and buffer, set the backbone to eval, and
   train only `fc3.weight/bias` for five complete epochs with CE. Clip the norm
   of these two parameters' gradients to10.
3. Continue with the SAME low-rank model and loader, freeze fc3 and train all
   other originally trainable parameters for five complete epochs with the
   original CE + .001 * Frobenius objective. Clip only the active backbone
   gradients to10. CE gradients pass through the fixed classifier into features.
4. Restore original requires_grad flags and individual module modes, then save
   one final client checkpoint. Restore those flags/modes on exceptions too;
   unsuccessful training never overwrites the downloaded checkpoint.
5. Recover/upload only backbone parameters and aggregate by the original
   sample-count weighted Avg. Server head remains an unused placeholder.

There is no download, SVD, aggregation or head reset between the two stages,
and no final extra head refit. `fc1/fc2`, including low-rank U/V, remain backbone.
Classifier membership uses the existing alias-aware fc3 parameter identities.
Head-stage `model.eval(); model.fc3.train()` avoids toggling the ReLU shared by
base/head. In the backbone stage, `model.train(); model.fc3.eval()` retains the
normal backbone behavior; frozen fc3 does not detach features or logits.

**Budget: head_epochs=5, backbone_epochs=5, total_data_epochs=10.** Each parameter
group gets five epochs of updates, while data undergoes ten training passes.
This is not a five-total-epoch local run. C0 and all 19 helpers have the same
training budget. Endpoint inference reads are logged separately from training.

## C0 evaluation and stage logs

C0 gets read-only observations at:

* `downloaded_before_head`: after the one real download, before head updates.
* `after_head`: after head5, before backbone updates.
* `after_backbone_primary`: after both stages; this is the primary endpoint.

Observers evaluate a private model copy, use CE-only train/test endpoint losses,
and restore Python/NumPy/Torch CPU/CUDA RNG. They write no intermediate checkpoint
and do not modify real parameters, buffers, modes or the next stage's batches.
Training-set endpoint metrics use full data in fixed order. C0 test correct/sample
counts after both stages are checked against ordinary saved-checkpoint accuracy.

Existing `target_proj_metrics.csv/json` continue to summarize only final C0
post-local accuracy: final, earliest best round, last10, last20. They also contain
`target_downloaded_before_head_acc`, `target_after_head_acc` and
`target_after_backbone_acc`, with the last equal to `target_post_local_acc`.
Aggregate-download accuracy stays a separate metric, using aggregate backbone
plus the same C0 local head, without training or server feedback.

Additional files in the independent log tree:

| File | Rows / fields |
| --- | --- |
| `fedrep_training.csv` | Two rows per client/round: `stage`, `train_epochs`, `train_batches`, `train_sample_visits`, `train_mean_ce`, `train_mean_weighted_frobenius`, `train_seconds`, optimizer/clip scope |
| `fedrep_clients.csv` | One row per client/round: schedule, 5/5/10 budget, per-stage counts/means/times, total optimizer time, endpoint observation time and total local call time |
| `fedrep_target_stages.csv` | Three C0 rows per round: endpoint CE, train/test accuracy, correct/sample counts and observation time |
| `fedrep_training.json` | Full phase/client/endpoint trajectories, budget policy, timing/loss semantics and final summary |
| Result H5 | `target_projection/fedrep_training` has round x20 training arrays and round x3 C0 endpoint arrays |

`train_mean_*` is a sample-visit weighted mean DURING optimization. `endpoint_*`
is separate evaluation AFTER a stage, CE-only. Head-stage weighted F is0.
Stage train times exclude endpoint inference; `total_local_call_seconds` includes
setup, observations and final checkpoint I/O. The compatibility normal-stat
`local_epochs` is10 for this schedule; CLI `args.local_epochs` remains5, explicitly
the backbone epoch count. No output should label this as five total data epochs.

Final export keeps `Client_0_model.pt` as the post-head5/post-backbone5 primary
model, `C0_aggregated_backbone_local_head.pt` as the separate aggregation-evaluation
model, and `Server_backbone_only.pt` as a non-inferable backbone dictionary.
The manifest states schedule/budget/model roles and final-round client statistics.

## Launch and controls

From `system/`, source `fedrep_local_commands.sh` for the full one-entry
`declare -a COMMANDS` array. Sourcing only defines it; no job is launched.
GPU0 is replaceable. Names/checkpoint/H5/final/log roots contain
`avg_local_head_fedrep_head5_backbone5`. Logs are in:

```text
system/training_logs/avg_local_head_fedrep_head5_backbone5_seed0_gr30/
```

Its run subdirectory contains CSV/JSON/H5, with captured train.log at the process
root and a sibling log-only ZIP. No checkpoints are placed in that log bundle.

Launcher preview without training:

```bash
python run_target_proj.py --modes avg --rounds 30 --target_head_mode local --target_local_schedule fedrep --fedrep_head_epochs 5 --device-id 0 --dry-run
```

Reuse the user's `10-15-03.zip` controls (Avg shared joint5 and Avg local joint5).
Do not automatically rerun them. The new comparison is Avg local head5->backbone5
with ten training data passes, not joint5, and is not evidence of a complete
FedRep reproduction.

## Verification

```bash
python -m unittest discover -s tests -p test_fedrep_schedule.py
python -m unittest discover -s tests -p test_fedrep_runtime.py
```

Tests cover phase-only optimizer/clipping membership; immutable head-stage
backbone parameters and BatchNorm buffers; fixed backbone-stage head; nonzero
CE and F gradients reaching backbone; originally frozen parameters and mixed
module-mode restoration; restoration on either phase failing; and exact final
parameters/RNG with observations enabled versus omitted. CLI/diagnostic guards
and main/launcher forwarding are exercised without starting real training.

The two-round runtime check uses 20 heterogeneous Hyper_CNN_512 clients, reduced
convolution width4 but full fc1/fc2 dimensions and a 100-class fc3, synthetic
32x32 inputs and all five epochs in BOTH stages. It compares default joint
uploads/server/RNG with pre-change `7e58fda`, verifies head persistence, exact
sample Avg, no in-training download/recovery/SVD, equal per-client budgets,
CSV/JSON/H5 and final models. Synthetic checks cannot establish CIFAR accuracy gains.

Local acceptance completed: 152 tests passed (144 existing regressions, seven
new policy/gradient/state/CLI/command tests and one full CNN runtime test). The
runtime check took100.0 s on CPU for three instrumented two-round runs (pre-change
joint, explicit joint and FedRep-style), including full5 epochs per active stage.
This is not real-data throughput. Joint uploads, server parameters and RNG
matched exactly; head-only/backbone-only changes and 5/5/10 logs/exports passed.

Real CIFAR100/pat_20 shards were not present locally, so the formal31-round run
was not started and no accuracy improvement is claimed. No push was performed.
