# FedFewS migration: local acceptance report, 2026-10-16

Status: **synthetic-data behavioral acceptance passed; real alpha=.5 data
acceptance remains pending on the server**. No formal 2001-round experiment was
started and no accuracy improvement is claimed.

## Source and environment

The user confirmed that no local FedFew checkout existed and authorized GitHub
as the reference. A fresh clone at `D:\All_Git_Repo\FedFew_reference` was clean
on `main`, commit `1b29dec70b66d89b956d90e0da024897f698897a`. The migrated vendor
classes are isolated from FedClip, with the adaptations documented in
[`PROVENANCE.md`](../system/fedfews_compat/PROVENANCE.md). AST comparison after
removing import declarations found identical bodies in the two client files,
two server files and reader. This supplements, rather than replaces, dynamic
source-vs-migration acceptance.

Existing environment used, without installation or modification:

| Component | Version |
|---|---|
| Python | 3.11.14, `D:\miniconda\envs\htfllib\python.exe` |
| PyTorch / CUDA build | 2.0.1+cu117 / 11.7 |
| cuDNN | 8500 |
| NumPy | 1.26.4 |
| SciPy | 1.16.3 |
| torchvision | 0.15.2+cu117 |
| scikit-learn | 1.7.2 |
| h5py / PyYAML | 3.15.1 / 6.0.3 |
| Acceptance threads | Torch intra-op 2, inter-op 8; source client num_threads=1 |

No deterministic or TF32 settings were silently changed: deterministic=False,
cudnn.deterministic=False, cudnn.benchmark=False, matmul TF32=False,
cuDNN TF32=True. The default Python 3.13.9 environment lacks SciPy,
scikit-learn and h5py. The entry reports those missing dependencies and exits;
it does not install them or change environments.

## Controlled external source comparisons

Both endpoints executed the original standard FedFewS bodies, but the reference
loaded the files from the independent Git checkout, not the migrated modules.
The reference also executed the original runner's base/algorithm merge and CLI
construction; effective defaults and merged values matched the migration.

Synthetic images have the real 3x32x32 input and the original 100-class CNN,
K=3, SGD .005, one epoch/model. Client train counts are `7+3*i`, test counts
`3+3*i`, deliberately unequal. Some clients have both a valid pkl and a stale
npz; both readers selected the pkl. Synthetic config says alpha=null and is
explicitly marked synthetic; no real Dirichlet partition is implied.

| Check | 20-client CPU | 3-client CUDA |
|---|---:|---:|
| Actual training/aggregation count | 4 (loop 0..3) | 3 (loop 0..2) |
| Evaluation loop indices | 0, 2, 3 | 0, 2 |
| Non-evaluation loop indices | 1 | 1 |
| Observed events | 705 | 91 |
| Compared parameter elements | 235,800,540 | 36,063,612 |
| Maximum parameter absolute error | **0** | **7.450580596923828e-9** |
| Maximum scalar absolute error | **0** | **6.151888720662413e-11** |
| Parameter/scalar atol / rtol | 0 / 0 | 1e-6 / 1e-5 |
| Result | PASS | PASS |

For CPU zero error, the reported first maximum location is event 0,
`initialization/model_0.conv1.0.weight`, flat index 0 (all locations are exact).
CUDA's maximum is event 7, `client_train/model_0.conv1.0.weight`, flat index 368.
GPU tolerances account for same-environment floating-point execution; this
report does not establish a particular CUDA-kernel cause. Discrete choices,
sample indices/counts and all recorded RNG hashes still matched exactly.

Compared stages include initial state_dict names/shapes/dtypes and parameter
count (924,708), fixed-input forward, each client's K trained models,
last-epoch training losses, endpoint deployment CE and argmin, test correct/
sample counts, endpoint train_metrics, selected-client/receive order, sample
shares, alpha/w and all aggregated servers. Every observed event compares
Python/NumPy/Torch CPU/all initialized CUDA RNG state hashes. The CPU run also
covered the forced final evaluation at an index not divisible by eval_gap.

CPU wall times with trace I/O were 34.42s (reference) and 47.51s (migration).
CUDA wall times were 11.27s and 5.22s. These include per-event reference tensor
writing versus loading/comparison, logging and initialization; **they are not
algorithm throughput comparisons**.

## Additional tests and smoke

The final FedFewS test suite has **9 passing tests**. It checks the independent
unbalanced-sample analytic aggregation, old-server subtraction, ID mapping,
absence of per-model renormalization, last-epoch loss versus endpoint loss,
data hashes/priority, source defaults and protocol rejection, original RNG
mode, model schema/copies, earliest-best/tail summaries, failure detection,
optimizer persistence, and CSV/JSON/H5/actual-K-model exports.

A separate two-round uninstrumented-vs-observed test found all final local and
server parameters and RNG states **exactly identical**. Thus common observation
hooks on both acceptance endpoints are not the only evidence of noninterference.
H5 test-correct counts divided by test sizes agree with the overall rs_test_acc.

The normal CLI completed three actual rounds on synthetic shards with separate
logs, metrics and final models. A Windows GBK/emoji output failure found in its
first invocation was fixed using UTF-8 stream handling, then rerun successfully.
The final CLI also passed a two-round original-RNG smoke under
`fedfews_runs/local_entry_20261016`, including resolved manifests, initial/final
RNG, all extended H5 fields and actual K-model export.

The selected old-mode mixed regression suite passed **143 tests** (including
the then-current eight FedFewS tests). It covered Projection, softmax, layer
mask/budget/weighting, group, feedback, helper Prox, local head and FedRep,
including the two-round heterogeneous-CNN FedRep runtime. The later ninth
FedFewS test passed in the final dedicated suite. Existing unrelated optional
Transformers/Swin warnings did not prevent those tests from passing.

The independent direct-exp analytic check allows 2e-14 absolute rounding
between exp and logsumexp paths; the observed difference was about 5.6e-15.
This is separate from CPU migration parity, which requires exact equality.

## Reproduce and inspect

From the FedClip repository root, in PowerShell:

```powershell
& D:\miniconda\envs\htfllib\python.exe -m unittest discover -s tests -p test_fedfews_migration.py -v
& D:\miniconda\envs\htfllib\python.exe system/verify_fedfews.py --source-root D:/All_Git_Repo/FedFew_reference --output-dir D:/All_Git_Repo/FedClip/fedfews_runs/recheck_cpu --num-clients 20 --rounds 4 --eval-gap 2 --device cpu --torch-threads 2 --atol 0 --rtol 0
```

Original full local artifacts are in:

* `D:\temp\fedfews_acceptance_20261016_final` (20-client CPU).
* `D:\temp\fedfews_acceptance_20261016_cuda` (3-client CUDA).
* `D:\temp\fedfews_entry_smoke_20261016_b` (normal entry smoke).

Small machine-readable reports are also copied into the local workspace under
`fedfews_runs/local_validation_20261016/`; bulky reference tensors remain in the
acceptance directories. These generated files are not committed.

Server preview, actual-alpha checks, real-shard short acceptance, formal full
configuration and deferred pat10 commands are all in
[`system/fedfews_commands.sh`](../system/fedfews_commands.sh).
The server should first return PASS on the actual existing shards and intended
software stack. This local report does not reproduce the historical CIFAR-100
curve, certify unrecorded server edits, or substitute synthetic parity for
real-data acceptance.

The user's staged plot remains unchanged (SHA256
`A066FD21F6DF5DADC2DF1BBAF31C247D7C8326348C36F793B3B1FB7C0A175E8A`) and
is excluded from the migration commit.
