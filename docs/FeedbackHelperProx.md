# Helper Prox on the corrected feedback_avg flow

This ablation adds only `--target_prox_mu` to `feedback_avg`, with
`--target_prox_scope helpers`. The flow stays S -> 19 helper local5 uploads ->
helper sample-count Avg H -> C0 local5 -> recovered C0 feedback server.
C0 never receives Prox; every client downloads and trains once per round.
Server anchor retention, Projection, softmax, guidance and data holdout remain absent.

Both validators enforce finite nonnegative mu and allow only this new combination.
Feedback `scope=all` is rejected, including mu0. Positive Prox on `feedback_avg_self`
is rejected. `feedback_projection_self` remains deferred pending a reference definition.
The existing `projection_softmax` Prox policy, including self .125 and its helpers/all
scope choices, is unchanged. Legacy `-mu` is never consulted.

## Objective and actual download anchor

The existing client's `_train_with_prox` is used without a new loss implementation:
CE + lambda*Frobenius + mu/2*sum(trainable low-rank parameter displacement squared).
`make_prox_anchor` clones detached parameters after the actual SVD download and
before the first optimizer step. It includes U/V and head, excludes buffers/frozen
parameters, stays fixed for five epochs and is rebuilt next round. SGD lr .005,
regularization .001 and total-gradient clip norm10 remain unchanged.

`observe_prox_local_train` now handles `feedback_avg` diagnostics for helpers and C0,
including mu0. **It calls train once internally; the outer runtime does not call it
again.** Its before/after CPU reads use the actual saved low-rank checkpoint, with
no extra decomposition or full-W distance substituted. Python/NumPy/Torch CPU/CUDA
states are protected for observations; actual SGD/download/feedback changes remain.
Mu0 and C0 still take the exact ordinary client training path, allocating no training
Prox anchor and adding no Prox term to the original loss/gradient.

## Logs and output compatibility

The primary metric remains C0 ordinary post-H local5 accuracy, with final,
earliest best round, last10/last20, pre-train H-download accuracy and same-download
gain. No evaluation loss definition is changed to include Prox.

`feedback_training.csv` and JSON history store per client/stage:

* `target_prox_mu`, `target_prox_scope`, `effective_prox_mu`, `prox_enabled`;
* CE, weighted Frobenius loss, raw F penalty and `proximal_loss`;
* actual-start low-rank `low_rank_start_distance` and `low_rank_start_squared_distance`;
* `end_proximal_loss`, trainable parameter count, epochs, batches, sample visits and times.

`proximal_loss` is the sample-visit weighted **mean during training** and already
contains mu. `end_proximal_loss = effective_mu*end_squared_distance/2` is a separate
endpoint value; it must not be confused with the mean or scaled by mu again.
C0's effective mu/mean/end Prox are 0 while its parameter displacement may be nonzero.
These low-rank-coordinate distances are distinct from recovered full-W helper/C0
delta norms. H5 stores `(round,20)` training diagnostic arrays plus configuration
and enabled client IDs. Final manifest retains configuration and final-round
per-client training diagnostics; the complete history stays in the log tree.
Positive-mu experiment JSON uses a shorter `feedback_avg_run_<id>.json` leaf to
avoid Windows path limits; mode/scope/mu remain in its directory and full experiment
name remains in its configuration. Existing mu0 filenames are unchanged.

Names/checkpoint/log/H5/final roots for positive mu contain mode, scope and mu.
Mu0 uses the existing `feedback_avg_helpers_only_local5` naming, including when mu0
is explicitly passed, so the existing control commands and output layout remain
compatible. Zero diagnostics add columns but preserve the numerical trajectory.

## Server commands

From `system/`, source `feedback_helper_prox_commands.sh` to define the complete
three-entry `declare -a COMMANDS` array; it starts no processes. GPU0/1/2 are examples.
The array uses mu .001/.01/.1, helpers scope, existing CIFAR100/pat_20 and original
Decom_CNN-5-512 capacities, seed0,20 clients/100 classes, lr .005,batch16, local5,
F .001, `-gr 200 -eg 1` (201 rounds). No anchor-retention flag is passed.
Reuse the completed/running same-budget mu0 `feedback_avg` control; the existing
`target_feedback_commands.sh` still provides that control if it is needed.

Independent log folders:

* `system/training_logs/feedback_avg_helpers_only_local5_prox_helpers_mu0p001_seed0_gr200/`
* `system/training_logs/feedback_avg_helpers_only_local5_prox_helpers_mu0p01_seed0_gr200/`
* `system/training_logs/feedback_avg_helpers_only_local5_prox_helpers_mu0p1_seed0_gr200/`

Each contains `train.log`, `status.json`, and `run_<id>/` for metrics/clients/training
CSV, JSON and mirrored H5; a sibling ZIP bundles logs without model tensors.

Launcher preview (no training):

```bash
python run_target_proj.py --modes feedback_avg --rounds 200 --target_prox_mu .001 --target_prox_scope helpers --device-id 0 --dry-run
python run_target_proj.py --modes feedback_avg --rounds 200 --target_prox_mu .01 --target_prox_scope helpers --device-id 1 --dry-run
python run_target_proj.py --modes feedback_avg --rounds 200 --target_prox_mu .1 --target_prox_scope helpers --device-id 2 --dry-run
```

## Checks and interpretation

```bash
python -m unittest discover -s tests -p test_feedback_helper_prox.py
python -m unittest discover -s tests -p test_feedback_helper_prox_runtime.py
```

Math/CLI tests exercise both validators, independent parameter naming, narrow
scope, U/V/head/frozen exclusion, start-zero and mu*(theta-anchor) gradients,
command forwarding and the actual main parser without starting its training loop.
The CNN smoke compares mu0 with the full original `0af37b3` feedback runtime at
every helper upload, intermediate H, C0 recovered upload, feedback server and RNG
state, at two rounds. Deliberate random draws inside protected observations test
isolation. Positive mu .1 exercises 19 helper anchors, verifies actual-download
parameters/fixed anchors/rebuilding, C0 exclusion, five-epoch budget, next-round
feedback, frozen parameters/buffers and CSV/JSON/H5/manifest diagnostics.
It uses actual heterogeneous Hyper_CNN_512 structure with reduced convolution
width4,100-class head and synthetic32x32 data; it is not real CIFAR accuracy evidence.

Local verification completed: 227 tests passed, including the new CLI/math tests,
the two-round heterogeneous CNN smoke and existing Prox, feedback, Projection,
softmax, group, DWA, APA, layer-mask and storage/logging regressions. The new CNN
smoke ran on CPU and took 215.7 seconds including three instrumented runs
(original reference, zero mu and positive mu); this is not a server throughput
measurement. All zero-mu parameter and RNG comparisons were exact. Positive mu
produced nonzero helper Prox statistics in CSV/JSON/H5 while C0's effective mu,
mean Prox and endpoint Prox stayed zero. No real CIFAR accuracy claim is made.

Smaller parameter drift alone does not establish useful transfer. Evaluate final,
late-round means and best ordinary C0 accuracy against the same mu0 control.
No push, full 201-round training, dataset regeneration or split is part of local verification.
