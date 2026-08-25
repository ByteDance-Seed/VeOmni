# Import side effect: registers the dit_offline / dit_online /
# minimax_h3_online data transforms used by the offline / online training
# flows (DiTTrainer looks "dit_offline" up at transform-build time).
from veomni.data.multimodal.dit import data_transform  # noqa: E402,F401

from . import minimax_h3_condition, minimax_h3_transformer
from .minimax_h3_core.offline_loader import (
    build_minimax_h3_offline_dataset,
    build_minimax_h3_online_dataset,
)


__all__ = [
    "minimax_h3_condition",
    "minimax_h3_transformer",
    "build_minimax_h3_offline_dataset",
    "build_minimax_h3_online_dataset",
]
