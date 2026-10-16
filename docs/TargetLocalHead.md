# Local classifier ablation in synchronous FedTargetProj

Optional Avg/local `--target_local_schedule fedrep` performs head5 then backbone5
(ten total training data epochs), with joint as the unchanged default. See
[TargetFedRepSchedule.md](TargetFedRepSchedule.md) for its restricted scope and logs.

For the separate short-run snapshot plus offline frozen-backbone head diagnostic,
see [HeadDiagnostic.md](HeadDiagnostic.md). It preserves this training protocol
and does not add online head retraining or a meta holdout.

`--target_head_mode shared` is the default and retains the original full-model
download, aggregation, evaluation and export paths. `local` supports only these
two synchronous configurations with Decom_CNN-5-512, a 100-class output and C0:

* `avg --target_head_mode local`: sample-count weighted backbone Avg over all clients.
* `projection_softmax --projection_self_weight .125 --target_head_mode local`:
  C0 backbone mass .125, helper mass .875, raw backbone global-delta cosine
  softmax at the existing temperature .2 and the existing full-vector Projection.

No group, feedback, DWA, meta holdout/snapshot protocol, nonzero Prox, pretrained
initialization or resume is supported by this first local-head implementation.
Invalid combinations are rejected in main, the launcher and server construction.
Shared mode does not acquire these restrictions. Training epochs, loss,
Frobenius coefficient, SGD, clipping and dataset files are unchanged.

## Parameter and download semantics

Only `fc3.weight` (100 x 512) and `fc3.bias` (100) are local. `fc1` and `fc2`
remain in the backbone. The scope is resolved by parameter identity: `head.1`
aliases `fc3`, while normal deduplicated `named_parameters()` lists `fc3.*`.
An unexpected alias, shape, missing bias or frozen classifier is an error.
Startup prints the canonical excluded names, aliases and shared parameter names.
`local_head_policy.json` records the complete scope and counts.

Before round zero, every client receives one exact copy of the same original
public server classifier that the shared baseline would download. No extra
classifier is randomly constructed. A checkpoint marker guards this one-time
initialization. Thereafter each normal SVD download copies only backbone
parameters and retains that client's previous post-local classifier and buffers.
The classifier is still trainable and receives ordinary CE gradients.

All clients synchronously train once for five epochs, including C0. There is no
post-aggregation C0 training or server feedback. Recovered uploads passed to the
aggregation kernels contain backbone parameters only. Global and C0 parameter
maps use the same scope, so deltas, reference norms, dot products, cosines,
softmax scores, Projection coefficients and the final sum all exclude fc3.
The original aggregation kernels and their arithmetic are reused without changes.
The internal server fc3 stays at its initialization as an unused placeholder.

## Accuracy, diagnostics and model artifacts

The primary metric `target_post_local_acc` uses C0's ordinary five-epoch
checkpoint, with its trained local classifier, before aggregation. Summary fields
include final, earliest best round, last10 and last20 means. `target_client_test_acc`
uses the current aggregate backbone decomposed at C0 capacity plus C0's same-round
post-local classifier. This observation composes independent loaded models in
memory, does not write any checkpoint, and restores Python/NumPy/Torch CPU/CUDA
RNG streams. Server placeholder fc3 parameters are never used for this evaluation.

CSV/JSON/H5 retain accuracy and raw cosine/conflict/weight/Projection diagnostics,
with explicit `target_head_mode=local` and backbone scope. H5 also stores actual
per-client weights and coefficients. Avg's existing hypothetical Projection
summary is labelled as such; its actual projection-enabled flag, coefficients
and removed components are zero. Weight diagnostics are recalculated normally
from each run's own uploads, never replayed from another experiment.

Final export deliberately differs from the full-model server export:

* `Server_backbone_only.pt`: recovered backbone parameter dictionary, no classifier,
  explicitly marked as not directly inferable. Do not load it as a full predictor.
* `Client_i_model.pt`: ordinary final post-local model with client i's own classifier.
  `Client_0_model.pt` corresponds to the primary endpoint.
* `C0_aggregated_backbone_local_head.pt`: final aggregate downloaded at C0 capacity
  plus final C0 local fc3, corresponding to the aggregation evaluation endpoint.
* `manifest.json`: parameter scopes, model roles, configuration and accuracy summary.

The temporary internal `Server_model.pt` still contains its unused placeholder
fc3 for the existing SVD machinery. It is explicitly described as backbone-only
in metadata and is not copied into the final export under a full-model name.
Server buffers are not aggregated; download keeps local buffers, as before.

## Commands and logs

From `system/`, source `target_local_head_commands.sh` for the full two-entry
`declare -a COMMANDS` array. It only defines the array and starts no training.
Both configurations use original CIFAR100/pat_20, 20 clients, 100 classes,
heterogeneous Decom_CNN-5-512, seed0, full participation, lr .005, batch16,
local5, F coefficient .001 and `-gr 100 -eg 1`: 101 ordinary rounds.
Prox is zero by default. GPU0/1 are replaceable examples.

Independent log-only folders are:

* `system/training_logs/avg_local_head_seed0_gr100/`
* `system/training_logs/projection_softmax_self0p125_local_head_seed0_gr100/`

Each contains captured `train.log`, `status.json`, and per-run CSV/JSON/H5.
The sibling ZIP bundles logs without model tensors. Checkpoint, H5 and final-model
roots also contain `local_head`; launcher and main enforce this for custom roots.

Launcher previews without training:

```bash
python run_target_proj.py --modes avg --rounds 100 --target_head_mode local --device-id 0 --dry-run
python run_target_proj.py --modes projection_softmax --rounds 100 --projection_self_weight .125 --target_head_mode local --device-id 1 --dry-run
```

## Local verification

```bash
python -m unittest discover -s tests -p test_target_local_head.py
python -m unittest discover -s tests -p test_target_local_head_runtime.py
```

Unit checks cover aliases, scope rejection, main/launcher forwarding, independent
names, head-only perturbation invariance, exact sample Avg, analytical conflicting
Projection and zero backbone reference. The runtime check uses two real
heterogeneous Hyper_CNN_512 rounds with synthetic 32x32 data, convolution width4,
the full fc1/fc2 dimensions and a 100-class fc3; every client trains all five epochs.
It checks first-head equality, local classifier training/persistence, normal SVD
downloads, buffer/checkpoint/RNG isolation, weights and CSV/JSON/H5/model exports.
An explicit shared run is compared bitwise with the pre-change `715c80b` runtime.
Synthetic smoke is correctness evidence, not a claim of CIFAR accuracy improvement.

Completed local validation: 8 new unit/CLI/command checks, 128 existing numerical,
lifecycle, Prox, feedback, group, storage and logging regressions, and the CNN
runtime test all passed (137 tests total). The runtime test completed in 157.2 s
on CPU, including four instrumented runs (pre-change shared, explicit shared and
the two local-head configurations); this is not formal training throughput.
Shared uploads, server parameters and RNG states matched exactly. Final exports
contained all 20 ordinary client models, the backbone-only server artifact and
the separate C0 aggregate evaluation model. No dataset was regenerated or split,
and no formal 101-round experiment was started.
