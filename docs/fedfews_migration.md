# FedFewS standard migration and acceptance

本阶段已完成独立复现入口及原代码对照验收工具；CPU不等样本量对照精确一致，
CUDA短程对照在显式容差内通过。真实α=.5分片仍需在服务器完成短程验收，
再由用户启动正式实验。未加入低秩FedFew，未启动长实验。

The independent entries are `system/run_fedfews.py` and
`system/verify_fedfews.py`. They never import FedTargetProj. See
[`PROVENANCE.md`](../system/fedfews_compat/PROVENANCE.md) for upstream commit,
license and every adaptation. Existing plotting changes are excluded.

## What is preserved

The standard 3-server / K-local-model lifecycle uses the original full CNN,
parameter names and construction order. All K servers are deep copies of the
same model. Each client retains its original K SGD optimizers across downloads.
It trains models in order 0,1,2, each for one epoch. Downloads and aggregation
copy parameters only. This CNN has no buffers.

Aggregation loss is the last training epoch's batch CE sum weighted by batch
size and divided by client train_samples. Endpoint train CE is separately
computed in eval mode to choose the deployed post-local model. Source comments
calling the former “eval loss” are misleading and intentionally not implemented
as a change to its actual arithmetic. Selection never uses test accuracy.

Sample fractions multiply the losses before the source SciPy logsumexp weight
calculation. Actual increments are alpha_i * w_ik * (W_ik - S_k_old), accumulated
in the original Python-random receive order. No per-server mass normalization
or additional sample weighting occurs. Uploaded IDs map losses to the matching
clients even when receive order differs from ID order.

The source evaluation order is preserved: train all selected clients, endpoint
selection for all clients, test all clients, train_metrics for all clients, then
receive and aggregate. Both train and test loaders shuffle, use global Torch
RNG, drop_last=False, workers=0. Source evaluation advances RNG, as before.

There is no SVD, low-rank capacity allocation, F penalty, clipping, Prox,
Projection or private head. K=3 / local_epochs=1 means three data traversals per
client per round.

## Data and configuration

`--data-root` points to the original **dataset root**, containing
`Cifar100/noniid_dir_20_a0p5/config.json`, `train/0.pkl` or `0.npz`, etc.
Nothing is generated, rewritten, normalized or repartitioned. On Linux, case
matters: original YAML spells `cifar100`; the supplied commands explicitly use
`Cifar100` to match the requested existing files and historical log. Change that
one dataset string if the actual directory is lowercase, on both acceptance and
training commands. No implicit fallback chooses another dataset.

`.pkl` wins if both formats exist. The audit records the selected file's SHA256,
whether a stale npz was shadowed, decoded float32 x / int64 y content hashes in
sample order, per-client counts and label histograms, and original config bytes
and SHA256. `--expected-alpha .5 --expected-partition dir` validates the actual
config values; names alone are never evidence. Missing/wrong config or shards
stop training. Optional pat10 uses ten clients; inspect recorded label counts
to verify the intended ten classes/client before running it.

Base YAML plus algorithm YAML is merged with CLI overrides. Every source parser
default is retained and exported, including unrelated unused options. Of note:

* Test batch is **256** for `Cifar100/noniid_dir_20_a0p5` and pat10. The source's
  exact-name check would give 1024 only for the unsuffixed `Cifar100` string.
* Source `momentum=.1` is an unused CLI default. Effective standard SGD is
  momentum=0, weight_decay=0; formal commands explicitly record `-mo 0` too.
* `-nth 1` does not call `torch.set_num_threads(1)` in this source. It means
  sequential client execution. Actual intra/inter-op threads and backend flags
  are recorded. Controlled acceptance uses two intra-op threads on both sides.
* Source bool-typed flags such as `-ld`/`-ab` should be left absent for False;
  passing the string `False` to source `type=bool` would enable them. Rep/cache
  flags use upstream str2bool and may explicitly receive False/True.

The migrated entry restricts this first-stage standard protocol, rejects
unsupported modes, and uses `--output-dir` instead of conflicting legacy model/
result-directory overrides. Existing output directories must be empty.

## Commands (definitions do not launch experiments)

All complete command arrays are in
[`system/fedfews_commands.sh`](../system/fedfews_commands.sh). Run from the
FedClip root. First obtain an independent reference on the server, without
overwriting a working FedFew checkout:

```bash
git clone https://github.com/pgg3/FedFew.git /data/zh/AllRepo/FedFew_reference
git -C /data/zh/AllRepo/FedFew_reference checkout 1b29dec70b66d89b956d90e0da024897f698897a
export FEW_SOURCE_ROOT=/data/zh/AllRepo/FedFew_reference
export FEW_DATA_ROOT=/data/zh/AllRepo/FedFew/PFLlib/dataset
export FEW_OUTPUT_ROOT=/data/zh/AllRepo/FedClip/fedfews_runs
export FEW_PYTHON=python  # use the existing intended environment, do not install/change it
source system/fedfews_commands.sh
```

Review the array contents, then explicitly choose a step:

```bash
eval "${PREVIEW_COMMANDS[0]}"               # configuration + read-only data audit, no train
eval "${SYNTHETIC_ACCEPTANCE_COMMANDS[0]}"  # original vs migrated, 20 unequal small clients
eval "${REAL_ACCEPTANCE_COMMANDS[0]}"       # existing alpha=.5 shards, 3 actual rounds
# Only after examining PASS and the real data manifest, manually start:
eval "${COMMANDS[0]}"                       # original RNG protocol, 2001 training rounds
# Optional later:
eval "${PAT10_COMMANDS[0]}"
```

For strict CPU real-data acceptance, replace `--device cuda --device-id 0` with
`--device cpu` and use `--atol 0 --rtol 0`. GPU examples explicitly allow
atol=1e-6, rtol=1e-5 for parameters/CE/weights. IDs, sampler indices, selections,
counts and RNG hashes must still match exactly. These are same-environment
parity tolerances, not accuracy tolerances. A failed comparison exits nonzero
and writes FAIL; never use historical accuracy as a substitute for acceptance.

`--rng-mode original` only invokes the source's main Torch seed=0, leaves
Python/NumPy externally determined, and retains torch.manual_seed(0) inside
every client constructor. `--rng-mode controlled --control-seed 0` additionally
fixes Python/NumPy before the same source resets. This deliberate controlled
protocol can also be used for a separate long run, with a different output dir;
it must not be described as exact replay of old logs. Initial full RNG states
and final states are saved, and each observed event carries state hashes.

## Outputs

For `--output-dir fedfews_runs/dir05_original`:

```text
manifest.json                 effective args, source/file hashes, environment, exact command
data_manifest.json            config, actual alpha, selected file/decoded hashes, label counts
data_config.json              exact original config.json bytes
rng_before_initialization.pt  complete initial states
logs/train.log                UTF-8 text log, separate from model files
run_0/manifest.json           actual resolved model/metrics paths + summary
run_0/rng_after_initialization.pt full states, including CUDA after model initialization
run_0/metrics/rounds.jsonl     every aggregation, upload/train order, losses/alpha/w/mass/distances
run_0/metrics/events.jsonl     RNG hashes + actual batch-order hashes for each model/epoch
run_0/metrics/clients.csv      per-client aggregation and deployment data
run_0/metrics/evaluations.csv  overall sample-weighted post-local results at evaluation points
run_0/metrics/evaluations.json detailed per-client selections/correct counts/test sizes
run_0/metrics/summary.json     final/best(earliest)/last10/last20 EVALUATIONS, percent
run_0/metrics/Cifar100/.../FedFewS_ours_0.h5
run_0/models/FedFewS_final.pt  actual K servers + K post-local models/client + optimizers/RNG
```

`--trace-batches` additionally saves the full actual sample-index lists per
loader iteration; default saves a hash, count and model/epoch/phase attribution
to limit log volume. Acceptance always saves full batches. No extra loader pass
is introduced. The source's H5 fraction units are preserved; new CSV/JSON
accuracies use percent and explicitly distinguish loop_round from completed
train count. At eval_gap=10, last10/last20 mean evaluation points, not rounds.
Final servers are after the last aggregation; primary final accuracy is from
the selected last **post-local** client models before that aggregation.

Acceptance output contains `acceptance_report.json`, source and migrated logs,
the two complete outputs and initial/final RNG files. `reference_states/`
contains per-event reference tensors for error location and may be large. It is
an acceptance artifact, not necessary for formal training. Do not confuse its
I/O-inclusive timing with algorithm-only runtime.

## Local result and remaining validation

See [`fedfews_acceptance_20261016.md`](fedfews_acceptance_20261016.md). Synthetic
actual-CNN CPU parity passed with unequal counts, including 20 clients and all
required stages. The normal entry exported metrics/H5/models. No local real
alpha=.5 shards are available, so real-data parity and historical performance
are **not yet accepted**. Server real-data acceptance is the next step, followed
by a user-started long run. Historical 41.30/42.07/41.24/41.36 percentages are
context only and are not a correctness oracle.
