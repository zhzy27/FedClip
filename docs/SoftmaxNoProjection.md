# Softmax controls without Projection

`softmax_only --projection_self_weight 0.125` uses C0 mass .125 and helper mass .875.
`group_softmax_only --target_group_ids 0,1,2,3,4,6,7 --target_group_weight 0.50`
uses group mass .50 and outside mass .50. All 20 clients still download the same
server, train five epochs and upload recovered full-W named parameters including
the classifier. No buffers aggregate. Prox is disabled.

On the **same server/upload snapshot**, both paired modes use the identical raw
global-delta reference, pre-projection cosine, temperature .2, stable softmax and
group/self mass. Only the projected control removes opposing parallel components.
The new group mode shares the reference and scoring path with the projected group
mode; its accumulation branch sums raw deltas and never calculates a correction.
The single-target mode calls the same similarity-weight function as its projected
control and never calls the Projection kernel. Dynamic helper weights are computed
afresh each round; no historical CSV is replayed.

Omitting self mass leaves old `softmax_only` unchanged. Omitting group mass uses
the original group's training-sample fraction. Finite [0,1] mass, normalized group
IDs, zero reference, zero mass and all-members/no-outside checks remain enforced.
Nonzero `target_prox_mu` remains restricted to the previous projected experiment.

## Commands (server)

From `system/`, the complete two-command `declare -a COMMANDS` array is in
`softmax_no_projection_commands.sh`. Sourcing it only defines the array; it starts
no processes. GPU 0/1 are examples. Both commands use seed0, CIFAR100/pat_20,
20 clients, normal 5 epochs, batch16, SGD .005, Frobenius .001, `-gr 200 -eg 1`
(201 training/aggregation rounds), complete training data and zero Prox.

Launcher alternatives, also starting no training when `--dry-run` is present:

```bash
python run_target_proj.py --modes softmax_only --projection_self_weight .125 --rounds 200 --device-id 0 --dry-run
python run_target_proj.py --modes group_softmax_only --target_group_ids 0,1,2,3,4,6,7 --target_group_weight .50 --rounds 200 --device-id 1 --dry-run
```

## Metrics and files

The single-target primary metric is ordinary C0 `target_post_local_acc`. It reports
final, earliest best round, last10 and last20. Aggregate accuracy remains
`target_client_test_acc`, a download observation before the next ordinary train.

For **the new group mode only**, the primary is
`group_post_local_sample_weighted_acc = sum(member_post_local_correct)/sum(member_test_samples)`.
The summary selects the best *round* using that weighted mean, with earliest ties;
it never averages the members' historical bests. It reports final/best/last10/last20
for the group and each member. Macro accuracy remains `group_post_local_acc`,
with a separate `group_post_local_macro` summary. Correct counts, test counts,
member accuracies, worst member, dispersion and downloaded aggregate accuracies
are retained. Older group modes retain their macro-primary convention and their
numerical output. For the completed projected group control, compare the existing
`group_post_local_sample_weighted_acc` time series rather than its macro summary;
recompute final/best/last10/last20 from that same-round weighted series if needed.

Follow `local(t) -> agg(t) -> local(t+1)`: each row's aggregation weights affect the
next initialization. Independent training trajectories can therefore produce
different dynamic weights after the first aggregation.

The formal commands store downloadable **logs only** in:

* `system/training_logs/softmax_only_self0p125_seed0_gr200/`
* `system/training_logs/group_softmax_only_beta0p5_seed0_gr200/`

`train.log` and `status.json` are at the process root; CSV/JSON and mirrored H5
are below `run_<id>/`. The process creates a sibling `.zip` bundle, excluding
checkpoint/final model tensors. Single-target files include `softmax_only_metrics.csv`,
`softmax_only_clients.csv`, `softmax_only_matrices.json` and `target_proj_metrics.*`.
Group files are `target_group_metrics.csv/json`, `target_group_members.csv`,
`target_group_clients.csv`, `target_group_matrices.json` plus H5 `target_group/`.
Checkpoints, H5 originals and final models use independent `temp/`, `h5_results/`,
`final_models/` roots containing the mode and mass, as shown in the commands.

`projection_enabled=0`, all actual projection coefficients and removed norms are
zero. Raw conflicts are retained: single-target `conflict` is the original negative
helper dot; group `raw_conflict_with_reference` and `raw_outside_conflict_count`
record the negative raw dots. Existing group `conflict`/`outside_conflict_count`
mean projection actually performed and thus remain zero, as does
`projected_client_count`. Raw softmax scores, weights and effective counts stay visible.

## Local verification

Use a Python environment with the repo's PyTorch/torchvision/h5py dependencies:

```bash
python -m unittest discover -s tests -p test_softmax_no_projection.py
python -m unittest discover -s tests -p test_softmax_no_projection_runtime.py
```

The math suite checks strict paired weights at .125/.50, analytic raw sums with
conflicts, zero/near-zero references and role mass, validation, finite effective
counts, weighted summaries and launcher commands. It compares old output tensors
and diagnostic records exactly with commit `6449fc5`, in float32 and float64.
The runtime smoke uses the real `Hyper_CNN_512` low-rank architecture, five capacity
ratios .9/.37/.35/.25/.15, 20 clients, full normal5 epochs and two rounds, with a
reduced convolution width (`n_kernels=4`) and synthetic 32x32 images/100 classes.
It verifies the classifier, downloads, saved uploads/buffers/RNG protection,
unequal test counts, CSV/JSON/H5/manifest/final export. It is a correctness smoke,
not evidence of CIFAR accuracy improvement or a full-width performance benchmark.

No formal 201-round training, push or weight replay is part of implementation verification.

Local acceptance (2026-10-13): 209 tests passed across the new math/runtime suites
and existing target Projection, softmax weighting, three group controls, layer
mask/budget, APA, DWA, Prox validation, launchers, storage and training-log suites.
Checks ran in `htfllib` (PyTorch 2.0.1+cu117 / torchvision 0.15.2+cu117), with the
CNN smoke explicitly on CPU. The two-new-mode smoke took 122.1 seconds locally;
this elapsed time includes both two-round trainings, evaluations and exports,
and is not a real-data/GPU throughput claim. Original group-control smoke also
passed. Main CLI help recognizes both no-Projection options.
