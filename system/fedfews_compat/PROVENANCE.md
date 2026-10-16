# FedFewS source and compatibility adaptations

Upstream: [pgg3/FedFew](https://github.com/pgg3/FedFew), commit
`1b29dec70b66d89b956d90e0da024897f698897a`, MIT, copyright Ping Guo.
The full license is retained in `vendor/LICENSE` and `../common/LICENSE.fedfew`.

At the initial migration the user reported no local FedFew working copy and
authorized GitHub as the reference. A fresh clean clone was obtained at
`D:\All_Git_Repo\FedFew_reference`. The user subsequently supplied their server
checkout's diff; see the local-profile correction below. No claim is made about
other unrecorded changes on that server. `source_manifest.py` retains original raw file
hashes, while each run records actual reference status/hashes and current
migration/common-module hashes. CRLF/LF checkout conversion can change raw
hashes; behavioral acceptance, not a normalized-hash substitution, establishes
the tested agreement.

## User-reported server profile correction (2026-10-16)

The supplied diff changes only these known settings:

* `PFLlib/dataset/utils/dataset_utils.py`: generator alpha .1 -> .5.
* Both CIFAR-100 dir05/pat10 base YAMLs: dataset `cifar100` -> `Cifar100`, GPU 0 -> 2.

The migrated base YAMLs now use Cifar100/GPU 2. The command arrays default to
the user's actual edited `FedFew` checkout as the independent reference, rather
than assuming a pristine clone represents that checkout. GPU remains explicitly
overridable. Training/algorithm settings, readers and original RNG behavior do
not change. The dataset-generation module is NOT migrated, imported or executed;
its source hash and literal alpha are read only for provenance. Actual existing
data still must pass config.json alpha/partition and shard-content checks. In
particular, changing generator alpha cannot relabel old alpha=.1 shards as .5.

Original upstream hashes are retained as upstream evidence, not replaced with
hashes of the reported modifications. `user_reported_profile` is explicitly
separate from the actual source checkout's Git state/file hashes. The server's
two untracked files shown in the prompt have not been inspected.

Acceptance reads the selected partition's external YAML using its own runner
and parser. `configuration_alignment.json` records source/migration YAML leaves,
effective defaults and explicit short-run overrides. Dataset-name CASE and GPU
ID differences from a pristine reference are allowed only when the exact same
runtime dataset/device reach both endpoints. All other YAML/default differences
remain fatal, even when short-run epoch overrides might otherwise mask them.
The independent external reader/CNN/client/server implementation is unchanged.

The local validation fixture copies recorded source files from pristine commit
1b29dec and applies the three reported edits without modifying that checkout.
It is a simulation of the supplied diff, not inspection of the real server.
See `../../docs/fedfews_local_profile_20261016.md` for results and commands.

## Common primitives after FedClip 189bacd

The initial migration in `189bacd` primarily changed imports and whitespace.
**The current migration no longer claims that all vendor bodies are identical
to upstream.** The changes below are intentional compatibility adapters; their
behavior is checked against the independent external source.

| Shared implementation | FedClip adapter | Migrated FedFew adapter |
|---|---|---|
| `common.data_io.read_shard` | original dataset/partition/alpha/cpc root, explicit npz only | independent data root/full suffix, pkl before npz |
| `common.data_io.decode_image_data` | original Tensor(list) conversion | original stack(from_numpy) for lists, Tensor(array) otherwise |
| `common.data_loading.make_data_loader` | train shuffle=True, test=False, original batch | train/deployment/train_metrics/test shuffle=True, original train/test batches |
| `common.models.FedAvgCNN` | old name re-exported from `flcore.trainmodel.models` | direct common import, with `vendor.models` compatibility export |
| `common.metrics` | existing algorithm-specific summaries unchanged | export-only accuracy and final/earliest-best/tail reduction |

The common file reader only reads an explicitly selected file. Format policies
are explicit ordered suffixes; it never searches another partition. Each
adapter retains text/Shakespeare routing, few-shot behavior, root rules and
function signatures. Caches remain entirely caller-owned. The audit calls the
same selected-file decoder used by migrated training. For reference acceptance
it instead explicitly calls the external reader.

The factory adds no Generator, seed reset or sampler traversal. It receives the
call site's module-local DataLoader class, so Recorder's injected class observes
the actual batches on the migrated side. The external reference still invokes
its own DataLoader constructor, with the same observation wrapper; it never
uses the common factory.

The single ordinary CNN class has unchanged executable statements, constructor
order/defaults, registered parameter order, initialization and forward. It
imports only Torch. The legacy `flcore.trainmodel.models.FedAvgCNN` and
`fedfews_compat.vendor.models.FedAvgCNN` attributes remain resolvable aliases,
including unpickling old complete-model checkpoints. New complete-model saves
can resolve `common.models.FedAvgCNN`. No SVD or capacity distribution is added.

Metrics consume already-collected counts/points, never models or loaders.
Sample accuracy is a fraction. Summary inputs declare fraction/percent units;
FedFew's legacy percent export reductions stay in percent to preserve rounding.
Empty/zero/nonfinite invalid public inputs raise errors; the legacy FedFew
summary adapter retains its existing explicit empty-history `{}` result.
The algorithm's own evaluation is unchanged, and collected-count export
accuracy is checked against it. Loop index, completed train count and evaluation
object are explicit; tails count evaluation points.

## Algorithm-specific code deliberately retained

`vendor/clientfedfews.py` and `serverfedfews.py` keep the original K-model/K-SGD
lifecycle, training order, last-epoch training losses, independent endpoint
deployment selection, SciPy logsumexp weights and old-server-relative incremental
aggregation. Server/client upload state, standard/controlled RNG protocols,
internal Torch resets and source evaluation's RNG consumption are unchanged.
The bases are not merged with FedTargetProj. Source parser defaults and supported
YAML mapping are retained; the external runner's mapping is independently checked.

The existing output adaptation remains: source inherited save_global_model saves
an unused single template; the migration saves the actual K servers, all K local
post-local models, indices, optimizers/schedulers and RNG. Original H5 fields
remain, with additional diagnostics. UTF-8 console handling supports Windows.

## Independent reference and limits

`reference.py` still executes the external checkout's own reader, CNN, base
loaders and algorithm bodies, resolving only original package imports into
isolated namespaces and stubbing disabled DLG. It does not import common
reader/model/loader implementations. Original evaluation/non-evaluation branches
execute. Same initial Python/NumPy/Torch CPU/CUDA states and observation hooks
are supplied; the original internal reseeding remains in place.

Read the latest common-refactor acceptance report in
`../../docs/fedfews_shared_acceptance_20261016.md`. The earlier migration report
describes `189bacd`, not a blanket vendor-body claim for this refactor.
Synthetic parity does not establish real-data parity, historical curve replay,
or fair performance comparison between ordinary and heterogeneous low-rank CNNs.
