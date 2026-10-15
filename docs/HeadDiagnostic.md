# Local-head accuracy-drop diagnostic

This diagnostic separates classifier adaptation from representation quality using
ordinary C0 post-local snapshots. It does not change aggregation, Frobenius
regularization, normal SGD, client order or any federated update. It introduces
no FedRep, feedback training, extra online C0 epochs or validation holdout.

## Run the Avg pair first

From `system/`, source `head_diagnostic_commands.sh` for a two-entry
`declare -a COMMANDS` array: sample-count Avg with shared head, then Avg with local
fc3. Sourcing defines commands only; it launches no process. Both have
`-gr 30` (31 inclusive rounds), the original full CIFAR100/pat_20 dataset,
20 clients/100 classes, original heterogeneous Decom_CNN-5-512 capacities, C0,
seed0, full participation, ordinary local5, lr .005, batch16, F coefficient .001,
Prox0 and every-round evaluation. LR decay remains off. GPUs0/1 are examples.
Names, logs, checkpoints, H5 and final-model roots are independent.

After both complete, run:

```bash
python head_diagnostic.py --snapshots head_diag_snapshots/avg_seed0_gr30 --dataset-root ../dataset --output-dir head_diag_results/avg_seed0_gr30 --device cuda --device-id 0 --shuffle-seed 1729
```

Inspect `head_diag_results/avg_seed0_gr30/comparison.md` before deciding whether
to run the second batch. `head_diagnostic_projection_commands.sh` contains the
separate two-entry shared/local ProjectionSoftmax array, both with C0 mass .125,
helper mass .875 and the original temperature .2. Its offline command is:

```bash
python head_diagnostic.py --snapshots head_diag_snapshots/projection_seed0_gr30 --dataset-root ../dataset --output-dir head_diag_results/projection_seed0_gr30 --device cuda --device-id 0 --shuffle-seed 1729
```

Each output directory must be new. Use a new name when repeating a diagnostic;
original final/best/last10/last20 metrics are never overwritten. No automatic
second-batch launch, test-based epoch choice or diagnostic-model feedback exists.
If multiple collection runs already exist under a root, pass the exact two
matched run folders to `--snapshots` instead of that entire root; duplicate
mode/head/round entries are deliberately rejected.

## Optional independent collection

* `--head_diag_collect`: off by default; only saves snapshots during training.
* `--head_diag_rounds 10,20,30`: completed-round IDs, `loop_round+1`.
* `--head_diag_dir ROOT`: separate from logs and model checkpoints; defaults to
  `head_diag_snapshots` when collection is enabled.

Only synchronous Avg and self-.125 ProjectionSoftmax with shared/local heads are
supported in this diagnostic. Nonzero Prox, meta protocols, pretraining/resume,
early stopping, epoch truncation or LR decay are rejected. The disabled switch
does not impose new restrictions on existing experiments or create any artifacts.

Snapshots are written to
`ROOT/<mode>_<shared-or-local>_seed0/<run_id>/R10.pt` (and R20/R30), with `index.json`.
R10 is collected after all ordinary training in loop_round9, immediately after
C0 post-local evaluation and before receive_ids/aggregation. The last R30 is
loop_round29; `-gr 30` still performs the additional loop_round30/R31 normally.
No snapshot is mislabeled as the next download or aggregate.

Each snapshot contains C0's original low-rank model, current fc3 and buffers,
native correct/test-sample counts and accuracy, completed/loop round, mode,
head mode, training seed, C0 capacity, model structure and parameter specification,
full training arguments/stats, backend settings and library versions. It also
contains an exact tensor copy and hash of the run's existing initial public
server fc3. No extra model is constructed for this initialization.

Original C0 train/test `.npz` paths, sizes, modification times, raw-file SHA256,
sample counts and processed-tensor fingerprints are recorded. Offline processing
requires matching raw and processed data; changing data or labels is an error,
not a reason to re-split or regenerate anything. Relocated identical data is
supported via `--dataset-root`.

Collection loads a separate checkpoint object, writes only to its independent
snapshot tree and preserves Python/NumPy/Torch CPU/CUDA RNG streams. It neither
recovers the model nor changes the ordinary upload, buffers, optimizer or order.
The initial public classifier is read from the already-created server. The
`index.json` public-head hash should agree across all four configurations.

## Fixed offline protocol

For every snapshot, the tool independently evaluates:

1. `native`, epoch0: original model, no training. Correct/sample counts must
   exactly reproduce the stored post-local accuracy; failure stops diagnosis.
2. `fresh_head`, epoch0: original low-rank backbone and the saved public fc3.
3. `fresh_head`, epoch1: the same model after one head-only epoch.
4. `fresh_head`, epoch5: after the fixed total of five head-only epochs.

Only fc3 weight/bias, identified by object identity despite the `head.1` alias,
are optimized. SGD uses lr .005, batch16, momentum0, weight_decay0, CE only and
gradient norm cap10 over these two parameters. There is no F/Prox/head penalty.
The backbone stays in eval and frozen; only `fc3.train()` is set, avoiding a
mode change to the ReLU shared by base/head. Backbone parameters and buffers are
checked bitwise after each epoch. No SVD, module recovery or truncation occurs.

Every snapshot starts from an independent copy and the same saved public head,
never a preceding result. A private CPU `torch.Generator` seeded1729 creates
the five permutations. The exact indices and their hash are exported; global
training RNG streams do not select these batches. Paired shared/local comparisons
check initialization and batch-order equality. Native inference and head training
use the original forward, including the ReLU immediately before fc3. No feature
cache is used.

## Metrics, norms and comparison

`head_diagnostic.csv/json` store all individual observations, including source
mode/head/round/seed/capacity, variant/head epoch, original and diagnostic accuracy,
train CE/accuracy, test CE/accuracy, counts, fc3 weight/bias norms, reconstructed
full `fc2` weight Frobenius norm and mean L2 norm of the actual fc3 input. The
feature norm uses all original C0 training samples in fixed original order and
records the count; a pre-hook observes the true post-ReLU classifier input.
The fc2 matrix is measured as `weight_u @ weight_v`, without assigning new
parameters/modules. Factor norms are not used to infer backbone scale.

Accuracy values are fractions. `head_comparison.csv` and `comparison.md` use
percentage points: shared minus local gaps for native and fresh1/fresh5, signed
gap reductions and absolute gap reductions. Positive native gap means local is
lower. A positive absolute reduction means the two accuracies moved closer even
if their ordering switched. Comparisons require matching mode/round/seed/data/
capacity; duplicate configurations or mismatched public heads/batches are errors.

Substantial shrinkage supports head adaptation/training as a factor. A persistent
gap supports representation quality as another possible factor; partial shrinkage
is consistent with both. Five fixed epochs do not establish optimal classifier
convergence, and none of these observations alone identifies F regularization as
the cause. Diagnostic accuracies are never substituted for federated results.

## Validation

```bash
python -m unittest discover -s tests -p test_head_diagnostic.py
python -m unittest discover -s tests -p test_head_diagnostic_runtime.py
```

The runtime smoke prioritizes Avg shared/local on temporary synthetic `.npz`
shards, a real heterogeneous low-rank Hyper_CNN_512 (convolution width4, full
fc1/fc2 dimensions, 100-class head), two rounds and complete five-epoch local
training. It compares pre-change `22ea7cf`, collection off and collection on;
checks all upload/server parameter hashes, upload order and RNG states; verifies
pre-aggregation low-rank snapshot timing, native equality, public initialization,
identical shuffle orders, independent retraining, frozen backbone/buffers and
input-file immutability. The offline CLI and CSV/JSON/Markdown exports are tested.
This local synthetic smoke is not evidence about real CIFAR accuracy drops.

Local acceptance: 146 tests passed (136 existing regressions, 8 diagnostic
policy/CLI/command checks, one four-configuration public-initialization check
and one complete Avg collection/offline runtime check). The Avg runtime check
completed in 216.1 s on CPU, including six instrumented two-round runs
(pre-change/off/on for shared and local) and offline/CLI checks. This is not
formal-run throughput. The four initial public-head hashes were identical.

Real `dataset/Cifar100/pat_20/train/0.npz` was absent locally. Real-data 31-round
collection and the corresponding accuracy-gap diagnosis have therefore not been
run. The supplied first-batch commands perform those steps on the server using
existing data. No real-data conclusion about classifier versus backbone cause
is claimed, and no second-batch training was automatically started.
