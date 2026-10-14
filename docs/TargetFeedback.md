# Helper-first feedback with one C0 local5 per round

This specification supersedes the earlier local5+5 C0 implementation. Only two
feedback modes are runnable. All existing non-feedback modes keep their old paths.

| Mode | Intermediate H | C0 / next server |
| --- | --- | --- |
| `feedback_avg` | Training-count normalized Avg of 19 helpers only | Download H, train local5 once, recover parameters as next server |
| `feedback_avg_self` | .125 * immutable starting S + .875 * Avg(19 helpers) | Same |

`feedback_projection_self` is **deferred pending a Projection reference definition**.
Main, launcher (including dry-run), server initialization and aggregation entry all
reject it explicitly before training. There is no current-round pre-aggregation C0
update, so no previous-round reference, post-aggregation reference, target gradient
or extra reference-generating training is silently substituted. The formal command
array contains no feedback Projection job. Existing `projection`,
`projection_softmax` and other old Projection variants remain unchanged.

## One round: helpers -> H -> C0 -> next shared server

For every loop round 0..200 (`round=loop_round+1`):

1. Freeze the recovered starting server **S_t** before any download, decomposition
   or training. Round0 uses the original common initialized server, with no C0
   prewarm. The starting snapshot contains independent parameter clones and a
   separate model/buffer copy, and its source round is checked.
2. Download S_t **only to C1-C19**, using original capacities, normal SVD factors
   and local-buffer rules. Each helper runs ordinary local5 once with fresh SGD,
   lr .005, batch16, CE + .001 Frobenius and clip norm10. C0 is neither downloaded
   nor trained during this stage; its old checkpoint is never used as an upload.
3. Receive **exactly 19 helper IDs** and cache recovered full-W named parameters,
   including head, without buffers. Check ID completeness, sample counts and source
   round. Calculate q_i = n_i / sum(helper train counts), independent of C0 data size.
4. Form H using the appropriate helper Avg and optional immutable S_t retention.
   Normal upload order is retained; no softmax or Projection is calculated. Save H
   as the one shared server without changing server buffers.
5. C0 performs its **only actual download** this round from H using its original
   low-rank capacity. Every parameter is replaced by the normal decomposed H
   parameter; C0's own buffers remain local under the existing download convention.
   Observe the actual downloaded checkpoint before training.
6. C0 trains ordinary local5 **once**, using a fresh optimizer and the same loss,
   regularization and gradient clipping. Observe this checkpoint after training.
   Real helper/C0 downloads and training retain their RNG and model changes.
7. Recover C0 full-W and overwrite all server named parameters with it. Server
   buffers remain its original shared buffers; no C0 buffers, H, S or helper models
   are mixed back into the parameters. Next round's helpers download this C0 feedback.

Each of all 20 clients trains5 and downloads once per round. Prox defaults to0;
`feedback_avg` now optionally supports helper-only Prox (see `FeedbackHelperProx.md`).
`feedback_avg_self` remains without Prox, and feedback scope=all is rejected.
There is no DWA guidance, meta split, learned weight, warmup, extra epoch or dataset regeneration.
Extra snapshot/recovery/evaluation observations preserve RNG; they never undo the
actual training or actual server feedback update.

## Server-anchor semantics

`--feedback_anchor_weight` belongs only to `feedback_avg_self`, defaults .125 and
must be finite in [0,1]. Formal experiments explicitly use .125, with no scan.
`feedback_avg` always has anchor0 and rejects an explicit anchor argument. Other
modes reject this parameter as well. Do not pass `projection_self_weight`: that
old flag describes a current-round C0 upload in the old synchronous modes.

Here alpha retains **this round's starting server S_t**, which came from previous
round's C0 training (initialized S_0 in the first round). It is not this round's C0
upload mass. `H = alpha*S_t + (1-alpha)*helper_Avg`, equivalently
`H-S_t = (1-alpha)*(helper_Avg-S_t)`. Final feedback replaces server parameters
completely with C0's post-H training output, irrespective of alpha.

Pure helper Avg uses weighted-parameter arithmetic, including classifier, in
normal upload order. Explicit alpha0 takes exactly the pure Avg code path. Zero
helper deltas return S_t exactly; alpha1 also returns S_t while all helpers still
train/upload. Helper sample weights and actual aggregation weights are logged
separately; zero actual helper mass has effective-helper count0 and min/max weight0.

## Metrics, snapshots and output

The nonexistent first-stage C0 metrics are removed (new JSON schema2). Every round
records `target_intermediate_download_acc`, primary `target_post_local_acc` and
`target_finetune_gain = post_local - intermediate_download`, from the same actual
download and five-epoch training. Summary `target_local` reports final, earliest
best/round, last10/last20 and counts; `intermediate_download` retains its own series.
Machine-readable fields use the existing [0,1] fractions; console uses percent and
gain in percentage points.

`server_anchor_weight` records 0/.125, not a C0 upload weight. The client-weight
table contains19 helper rows; training contains20 stage-labelled rows (19
`helper_local`, then C0 `target_post_aggregation`). Logs include each client's CE,
unweighted F penalty, weighted F loss, epochs, batches, sample visits and training
time. Round timings include helper downloads/training/capture, aggregation with
reads/writes, C0 actual download/SVD, C0 training, recovery/feedback and total time.
Update norms distinguish helper full-W changes from S_t, H-S_t, and C0's recovered
post-local change relative to H, including low-rank download truncation effects.
No nonexistent C0 projection direction/cosine/conflict is invented.

Two bounded cache slots at `<checkpoint_run>/feedback_helper_uploads/slot_0|slot_1/`
hold the frozen S,19 immutable helper uploads and stats, H and feedback server.
There is no `Client_0_full.pt` or first-stage C0 low-rank upload. C0 training never
overwrites helper uploads. A later round of the same parity replaces that slot;
all historical statistics remain in CSV/JSON/H5. Round/client checks reject stale
or mislabeled cached uploads and explicitly reject a request for helper upload0.

The complete two-command `declare -a COMMANDS` array is in
`system/target_feedback_commands.sh`. From `system/`, sourcing it only defines the
array and starts no processes. GPU0/1 are replaceable examples. Both use existing
CIFAR100/pat_20, original Decom_CNN-5-512 capacities,20 clients,100 classes,seed0,
full participation (helpers first, C0 last), lr .005,batch16,local5,F .001,
`-gr 200 -eg 1` (201 rounds) and zero Prox.

Corrected-flow log roots, distinct from previous local10 experiments:

* `system/training_logs/feedback_avg_helpers_only_local5_seed0_gr200/`
* `system/training_logs/feedback_avg_self_anchor0p125_local5_seed0_gr200/`

Roots hold `train.log`/`status.json`; `run_<id>/` contains `feedback_metrics.csv/json`,
`feedback_clients.csv`, `feedback_training.csv`, experiment JSON and mirrored H5.
The sibling ZIP bundles logs only. Checkpoint, original H5 and final-model roots
are separately tagged by mode, anchor and `local5`. The H5 `target_feedback` group
contains the two accuracy series, summaries,19 helper weights and20 training rows.
Final `Server_model.pt` is recovered C0 feedback with original server buffers;
the manifest states the corrected five-epoch budget and checkpoint roles.

Launcher preview (no training):

```bash
python run_target_proj.py --modes feedback_avg feedback_avg_self --rounds 200 --device-ids 0 1 --dry-run
python run_target_proj.py --modes feedback_avg_self --feedback_anchor_weight .125 --rounds 200 --device-id 1 --dry-run
```

## Checks

```bash
python -m unittest discover -s tests -p test_target_feedback.py
python -m unittest discover -s tests -p test_target_feedback_runtime.py
```

Math tests cover19-helper sample normalization, alpha0 bitwise Avg equivalence,
head-inclusive anchor formula, exact zero-update behavior, finite boundaries,
read-only inputs/RNG, invalid uploads, deferred Projection, summary timing and CLI.
Runtime smoke uses actual heterogeneous `Hyper_CNN_512` low-rank parameterization
(.9/.37/.35/.25/.15),20 clients,100-class head and two rounds of normal local5 on
synthetic32x32 data with reduced convolution width (`n_kernels=4`). It checks
exactly19 uploads, forbids reading C0 during the helper stage, verifies C0 training
once and actual H download, compares next-helper downloads with previous C feedback,
checks frozen anchors, immutable helper caches, buffers/RNG, CSV/JSON/H5 and final
model export. This is a correctness smoke, not real-data accuracy evidence.

Compare the two corrected modes to isolate starting-server retention in the same
helper-first feedback workflow. Projection awaits a separate reference definition.
Do not compare or combine the superseded local10 feedback results as if they used
this five-epoch protocol. No long experiment, push or dataset change is performed.

Local acceptance (2026-10-14): 12 corrected-flow math/options tests, the two-mode
CNN smoke test and 206 existing target/Projection/group/layer/APA/DWA/Prox/launcher/
storage/log regressions passed (219 tests total). Smoke ran on CPU with PyTorch
2.0.1+cu117 / torchvision 0.15.2+cu117 and took133.3 seconds including both modes,
comparison instrumentation and file exports; this is not a real-data/GPU speed
measurement. Main's deferred-Projection failure and the two-mode 201-round launch
preview were also checked without running formal training.
