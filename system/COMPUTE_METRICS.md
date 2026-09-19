# Round cost measurement

`run_compare_compute.py` enables `--measure_round_costs 1` automatically.
Existing comparison commands remain valid. Direct `main.py` runs can opt in
with the same flag; its default is 0.

## New CSV columns

| Column | Meaning |
| --- | --- |
| `local_train_sum_seconds` | Sum of participating clients' local training wall times |
| `local_train_max_seconds` | Maximum local training wall time among participating clients |
| `server_processing_seconds` | Complete server round processing, including download preparation/SVD |
| `upload_payload_bytes` | Sum of actual uploaded tensor/array bytes |
| `upload_payload_mb` | Upload bytes divided by 1,000,000 |
| `local_train_client_seconds_json` | Per-client local training durations |
| `server_processing_events_json` | Exclusive server processing durations by operation |
| `upload_client_details_json` | Per-client, per-upload-object byte counts |
| `round_cost_schema_version` | Version of these measurement definitions (currently 2) |

The original `server_total_seconds` / `server_events_json` columns are retained
for compatibility. They use the OLD whitelist timing and include file I/O;
do not use them as the new server processing metric. Appending to an old CSV
extends its header atomically and leaves new fields blank for historical rows.

All new records, including client details, are also stored in each experiment's
JSON under `round_cost_records`. The console prints one `[RoundCost]` summary
when a round closes (at the next client selection, or at the final save).

## Timing boundaries

- CUDA work is synchronized at accounting boundaries. Nested operations count
  only once. The scope is the sequential simulated training loop, not a network
  deployment. `local_train_max_seconds` is not a measured distributed latency.
- Local time covers `client.train()`: data loading, optimizer construction,
  forward/backward, all algorithm losses, steps, and upload preparation there.
- Server time covers the rest of the active round: downlink preparation,
  full-W recovery, aggregation, generator/head/prototype training, parameter
  preparation, and other server bookkeeping. In particular FedCLIP's existing
  `send_parameters -> client.set_parameters -> decom_larger_model` path is
  counted as server processing, even though the SVD method is in a client class.
- Model/object `save_item` and `load_item` operations are excluded, including
  serialization, deserialization and any device restoration performed inside
  `torch.load`. Explicit device copies outside those helpers count in their
  owning processing phase. Data batch I/O is part of local execution.
- Evaluation, client selection, FLOPs estimation, final exports, JSON writes,
  console writes and FedCLIP mechanism diagnostics are excluded. Diagnostics
  still execute and can affect cache/memory pressure; disable optional diagnostic
  flags for benchmark runs when possible.
- Round numbering is the existing zero-based loop index. `--rounds 1` executes
  indices 0 and 1. Download/SVD time is assigned to the loop in which it runs:
  round 0 includes initial decomposition, later rounds include decomposition of
  the preceding aggregate. No extra final download/SVD is invented for timing.

## Upload payload by method

| Method | Counted object |
| --- | --- |
| FedCLIP | Uploaded low-rank model parameters, including head; before full-W recovery |
| FML / FedMRL / PFedAFM | Uploaded shared `global_model` parameters, not the private model |
| LG-FedAvg / FedGen | Classifier/head parameters only |
| FD | Class logits dictionary |
| FedProto / FedGH / FedTGP | Class prototype dictionary |
| FedKD | Compressed parameter arrays, including U, singular values and V |
| FedSPU | Actual extracted subnetwork arrays returned by `get_updated_parameters` |

Tensor payload is `numel * element_size`, NumPy payload is `nbytes`. No `.pt`
file sizes, pickle metadata, class identifiers, sample-count scalars, transport
headers, private optimizer states or local-only auxiliary models are included.
Repeated server reads of the same client upload in a round count only once;
downloading or evaluating a client model does not count as uploading it.
Ordinary module uploads count parameters, matching these servers' aggregation
loops; FedSPU counts its actual returned arrays, including buffers if present.

Only the 12 methods in `run_compare_compute.py` are audited. Enabling the new
measurement for another algorithm raises an error rather than silently guessing
its upload protocol. No new packages are required beyond the training runtime.

For stable comparisons use multiple rounds/repeats, a consistent device, and
avoid concurrent competing experiments. Report steady-state statistics separately
from round 0. Excluding I/O does not eliminate GPU contention or warm-up effects.

## FedCLIP fine-grained server timing (version 2)

Enabled automatically with `--measure_round_costs 1`, for CNN and low-rank
ResNet. No change to model math, random initialization, aggregation or LR.
The following CSV totals partition `server_processing_seconds`:

- `server_send_prepare_seconds`: complete downlink preparation, NOT network time.
- `server_aggregation_seconds`: complete full-W recovery and weighted aggregation.
- `server_other_seconds`: remaining round bookkeeping, e.g. receiving IDs/weights.

`server_svd_seconds` measures ONLY `torch.linalg.svd(..., full_matrices=False)`;
`server_svd_calls` counts actual calls. It is a subset of send preparation, not
an additional cost to add to the totals. Truncation and construction of balanced
U/V factors are reported separately. A large send preparation time alone is
not evidence that SVD takes all that time.

Every operation below is an exclusive entry in `server_processing_events_json`
and has its own CSV column `server_<phase>_<operation>_seconds` (SVD uses the
shorter `server_svd_seconds` above):

| Phase | Operation | Timed work |
| --- | --- | --- |
| send | device_transfer | Explicit model/layer `.to(device)` calls |
| send | layer_init | Allocation and initialization of low-rank layers |
| send | matrix_prepare | Conv permutation/reshape into a 2D SVD input |
| send | svd | Thin SVD itself, before capacity truncation |
| send | factor_build | Rank truncation, square roots, diagonal products forming U/V |
| send | factor_copy | Copy U/V and bias into newly allocated low-rank layers |
| send | parameter_copy | Clone decomposed parameters into the client's model |
| send | decomposition_other | Remaining adaptation traversal and base rebuild |
| aggregate | deepcopy | Clone uploaded models before recovery |
| aggregate | device_transfer | Explicit device moves before/after recovery |
| aggregate | reconstruct_matmul | U @ V and full-weight reshape |
| aggregate | layer_init | Allocation/initialization of recovered full-rank layers |
| aggregate | weight_copy | Write recovered full weights and bias into layers |
| aggregate | recovery_other | Remaining recovery traversal and base rebuild |
| aggregate | parameter_index | Build named-parameter dictionaries |
| aggregate | validation | Validate full-model parameter names |
| aggregate | zero_init | Zero the aggregation destination |
| aggregate | weighted_add | Sample-weighted parameter multiply/add, including shape checks |

In the exclusive event map, `send_parameters` and `aggregate_parameters_avg`
now mean their *remaining unclassified overhead*, not their inclusive totals.
Use the new group total columns for comparisons. All exclusive entries still
sum to `server_processing_seconds`; never add group totals and their children.

`server_processing_details_json` contains event/client_id/layer/seconds/calls.
Client IDs distinguish capacity-specific SVD costs; layer names distinguish
conv2/fc1/fc2 (CNN) or residual block convolutions (ResNet). A null client ID
denotes shared server work. Empty layer names denote model-level work.
Parent residual timing entries can have calls=0 if no explicit child scope was
entered for that particular layer. These are not extra SVD calls.

The console prints `[FedCLIPServerBreakdown]` and two compact detail lines each
round; full details remain in the experiment JSON and comparison CSV. Historical
rows and other methods have blank fine-grained columns, not invented zero costs.

File save/load and console writes remain excluded. Device copies *inside*
serialization remain excluded too. CNN layer constructors currently allocate
on CPU: cross-device copies can therefore occur in factor_copy/weight_copy,
not just device_transfer. Instrumentation deliberately preserves these operations.
More CUDA synchronization boundaries add profiling overhead and reduce possible
overlap; these are synchronized operation wall times, not a CUDA kernel-only
profile or true network latency. Rerun to obtain the breakdown: the old 4.249s
cannot be retrospectively split into SVD/copy/initialization costs.
