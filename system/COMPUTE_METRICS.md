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
| `round_cost_schema_version` | Version of these measurement definitions (currently 1) |

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
