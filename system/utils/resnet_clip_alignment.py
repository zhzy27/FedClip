"""Configuration helpers for ResNet CLIP-alignment ablations."""

from dataclasses import dataclass


STAGE_NAMES = ("S1", "S2", "S3", "S4")
DEEP_STAGE_WEIGHTS = (1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class ResNetClipAlignmentStrategy:
    legacy: bool
    levels: int
    selected_stage_indices: tuple
    anchor_mode: str
    weighting: str
    final_projector: bool

    @property
    def selected_stage_names(self):
        return tuple(STAGE_NAMES[index] for index in self.selected_stage_indices)

    @property
    def aligner_stage_indices(self):
        return tuple(
            index
            for index in self.selected_stage_indices
            if index < 3 or self.final_projector
        )

    @property
    def stage_weights(self):
        if not self.selected_stage_indices:
            return ()
        if self.weighting == "equal":
            return tuple(1.0 for _ in self.selected_stage_indices)
        return tuple(DEEP_STAGE_WEIGHTS[index] for index in self.selected_stage_indices)


def resolve_resnet_clip_alignment(args):
    legacy = bool(getattr(args, "resnet_clip_legacy", 0))
    if legacy:
        return ResNetClipAlignmentStrategy(
            legacy=True,
            levels=4,
            selected_stage_indices=(0, 1, 2, 3),
            anchor_mode="depth",
            weighting="equal",
            final_projector=True,
        )

    levels = int(getattr(args, "resnet_clip_levels", 1))
    if levels < 0 or levels > 4:
        raise ValueError(f"resnet_clip_levels must be in [0, 4], got {levels}.")
    anchor_mode = str(getattr(args, "resnet_clip_anchor_mode", "final"))
    if anchor_mode not in {"depth", "final"}:
        raise ValueError(f"Unknown ResNet CLIP anchor mode: {anchor_mode}.")
    weighting = str(getattr(args, "resnet_clip_weighting", "equal"))
    if weighting not in {"equal", "deep"}:
        raise ValueError(f"Unknown ResNet CLIP weighting: {weighting}.")

    return ResNetClipAlignmentStrategy(
        legacy=False,
        levels=levels,
        selected_stage_indices=tuple(range(4 - levels, 4)),
        anchor_mode=anchor_mode,
        weighting=weighting,
        final_projector=bool(getattr(args, "resnet_clip_final_projector", 0)),
    )


def print_resnet_clip_alignment_summary(args):
    if getattr(args, "algorithm", None) != "FedCLIP":
        return
    if "resnet" not in getattr(args, "model_family", "").lower():
        return

    strategy = resolve_resnet_clip_alignment(args)
    selected = ",".join(strategy.selected_stage_names) or "None"
    print("===== ResNet CLIP Alignment =====")
    print(f"Legacy: {'ON' if strategy.legacy else 'OFF'}")
    print(f"Levels: {strategy.levels}")
    print(f"Selected stages: {selected}")
    print(f"Anchor mode: {strategy.anchor_mode}")
    print(f"Weighting: {strategy.weighting}")
    print(f"Final projector: {'ON' if strategy.final_projector else 'OFF'}")
    print("Loss: raw MSE")
    print("================================")
