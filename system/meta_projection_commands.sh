# Run from system/. Source this file to define commands; no command runs automatically.
# Invoke the functions in the listed order. META_GPU_ID/META_DEVICE can select the GPU.
META_WORK_DIR="${META_WORK_DIR:-meta_work}"
META_GPU_ID="${META_GPU_ID:-0}"
META_DEVICE="${META_DEVICE:-cuda:0}"

prepare_meta_split() {
    python meta_projection.py split --dataset-root ../dataset --fraction 0.2 --split-seed 1729 --output "$META_WORK_DIR/c0_split.json"
}
collect_meta_baseline() {
    python run_target_proj.py --modes projection --rounds 100 --device-id "$META_GPU_ID" --seeds 0 --meta_c0_split "$META_WORK_DIR/c0_split.json" --meta_collect_snapshots --meta_snapshot_rounds 20,50,80,100 --meta_snapshot_dir "$META_WORK_DIR/projection_seed0_snapshots"
}
time_meta_real_data() {
    python meta_projection.py benchmark --snapshots "$META_WORK_DIR/projection_seed0_snapshots" --dataset-root ../dataset --output "$META_WORK_DIR/real_data_benchmark.json" --device "$META_DEVICE" --checkpoint-steps 8
}
fit_meta_fixed() {
    python meta_projection.py optimize --snapshots "$META_WORK_DIR/projection_seed0_snapshots" --dataset-root ../dataset --output "$META_WORK_DIR/fixed" --strategy fixed --outer-lr 0.05 --updates 50 --initializations uniform biased --bias-client 0 --bias-logit 0.25 --virtual-seed 777 --dtype float32 --device "$META_DEVICE" --checkpoint-steps 8
}
fit_meta_per_snapshot() {
    python meta_projection.py optimize --snapshots "$META_WORK_DIR/projection_seed0_snapshots" --dataset-root ../dataset --output "$META_WORK_DIR/per_snapshot" --strategy per_snapshot --outer-lr 0.05 --updates 50 --initializations uniform biased --bias-client 0 --bias-logit 0.25 --virtual-seed 777 --dtype float32 --device "$META_DEVICE" --checkpoint-steps 8
}
run_meta_fixed_seed0() {
    python run_target_proj.py --modes meta_projection_fixed --rounds 100 --device-id "$META_GPU_ID" --seeds 0 --meta_c0_split "$META_WORK_DIR/c0_split.json" --meta_weight_file "$META_WORK_DIR/fixed/fixed_weights.json"
}
run_meta_projection_softmax_control() {
    python run_target_proj.py --modes projection_softmax --rounds 100 --device-id "$META_GPU_ID" --seeds 0 --meta_c0_split "$META_WORK_DIR/c0_split.json"
}
run_meta_fixed_multiple_seeds() {
    python run_target_proj.py --modes meta_projection_fixed --rounds 100 --device-id "$META_GPU_ID" --seeds 0 1 2 --meta_c0_split "$META_WORK_DIR/c0_split.json" --meta_weight_file "$META_WORK_DIR/fixed/fixed_weights.json"
}
