# Model-capacity code check

This check was generated from the `main` branch at commit
`08d73838b8f4637427876ede9ee2fd59245e0ed7`. The machine-readable audit is in
`paper/model_capacity_audit/`, and the reproducing script is
`system/audit_model_capacity.py`.

## Confirmed from the current code

- FedLAP CNN rank ratios in `main.py` are `0.90, 0.37, 0.35, 0.25, 0.15` in
  largest-to-smallest model-list order.
- FedLAP ResNet-18 rank ratios are `0.50, 0.40, 0.29, 0.20, 0.12` in the same
  order.
- `Model_Distribe` assigns models with `cid % len(args.models)`. For 20 clients,
  every capacity is therefore assigned to four fixed client IDs.
- CNN factorizes Conv2, FC1, and FC2. Conv1 and the classifier remain full.
- ResNet-18 factorizes both main-branch 3x3 convolutions in all eight residual
  blocks, including Stage 1. The stem, shortcut convolutions, normalization
  layers, and classifier remain full.
- The ResNet-18 semantic aligners pool the four stage outputs and apply four
  client-local linear maps from dimensions `64, 128, 256, 512` to 512. Their
  combined parameter count is 493,568. They are not part of the saved model,
  server upload, or aggregation, so the capacity table reports model-only
  counts and states this exclusion explicitly.
- Every parameter in each instantiated model currently has
  `requires_grad=True`; total and trainable model counts are identical.

## Inconsistencies and points requiring care

1. The corresponding full-rank comparison families are only approximately
   parameter matched. They are not dense versions of exactly the same FedLAP
   model. The CNN comparison family changes convolutional or hidden widths
   across levels, while FedLAP keeps one macro-architecture and changes ranks.
   The ResNet comparison family changes its base width.
2. FedLAP ResNet-18 uses `LayerNorm2d`, whereas `resnet18_family.py` uses
   GroupNorm. A claim that the low-rank and corresponding full-rank ResNet
   families differ only in rank would therefore be inconsistent with the
   current code.
3. Any older description stating that ResNet Stage 1 is not factorized is no
   longer correct: the current constructor creates `FactorizedConv` for all 16
   main-branch convolutions.
4. CIFAR-10 and CIFAR-100 do not have identical parameter counts because their
   classifier output dimensions are 10 and 100. The appendix therefore reports
   them as paired `C10/C100` values rather than one merged number.
5. The CNN `args.global_model` expression uses rank ratio 0.15, while the
   ResNet expression uses 1.0. In `FedCLIP.__init__`, the selected CNN server
   model is immediately recovered to the full effective model before it is
   saved. This implementation detail should not be confused with a sixth
   client capacity.
6. Capacity labels in the new table are ordered `L1` (smallest) through `L5`
   (largest). The earlier appendix draft numbered the main-list order from the
   largest model downward, so references to a bare "level 1" in older text
   need to be checked before merging.
7. The exact realized rank is rounded separately for every unfolded layer.
   Consequently, the configured rank ratio is not always numerically identical
   to `actual_rank / maximum_unfolded_rank`; the audit CSV contains the exact
   integer ranks.
