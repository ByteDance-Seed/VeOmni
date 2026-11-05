"""Decision model, data, objective, and training configuration."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from .arguments_types import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments


@dataclass
class DecisionModelArguments(ModelArguments):
    freeze_modules: list[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.accelerator.ulysses_size != 1 or self.accelerator.cp_size != 1:
            raise ValueError("Decision models do not support SP/CP yet; use ulysses_size=cp_size=1")
        if self.lora_config:
            raise ValueError("Decision models do not support LoRA training/export yet")
        if self.ep_sharded_stream_load:
            raise ValueError("Decision head weight sources do not support ep_sharded_stream_load yet")
        if self.accelerator.torch_compile.enable:
            raise ValueError("Decision preprocessing does not support torch_compile yet")


@dataclass
class DecisionDataArguments(DataArguments):
    supports_torch_compile = False
    train_path: str = ""
    dataset: str = "LocalLLaMA/typed-decisions"
    dataset_version: str = "all"
    dataset_revision: str = "1415d375cb86b418c53abec12795adf5b194e7fc"
    cache_dir: Optional[str] = None
    validation_fraction: float = 0.1
    limit_train_cases: Optional[int] = None
    limit_eval_cases: Optional[int] = None
    max_request_tokens: int = 16384
    shuffle_choices: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.validation_fraction < 1 or self.max_request_tokens < 1:
            raise ValueError("Decision validation_fraction must be in [0, 1) and max_request_tokens positive")
        if self.enable_multisource:
            raise ValueError("Decision data does not support multisource metrics yet")
        if self.dataloader.type != "native" or self.dataloader.persistent_workers:
            raise ValueError("Decision data requires native, non-persistent workers for epoch augmentation")


@dataclass
class DecisionTaskArguments:
    decision_model: str = "clef"
    decision_model_config: dict = field(default_factory=dict)
    mode: Literal["train", "evaluate"] = "train"
    eval_split: Literal["validation", "test"] = "validation"
    loss_type: Optional[str] = None
    label_smoothing: float = 0.1
    brier_weight: float = 1.0
    rps_weight: float = 0.5

    def __post_init__(self):
        if self.mode not in ("train", "evaluate") or self.eval_split not in ("validation", "test"):
            raise ValueError("Invalid decision mode or evaluation split")
        if self.mode == "train" and self.eval_split == "test":
            raise ValueError("Official test is allowed only in explicit evaluation mode")


@dataclass
class DecisionTrainingArguments(TrainingArguments):
    dyn_bsz: bool = False


@dataclass
class VeOmniDecisionArguments(VeOmniArguments):
    model: DecisionModelArguments = field(default_factory=DecisionModelArguments)
    data: DecisionDataArguments = field(default_factory=DecisionDataArguments)
    train: DecisionTrainingArguments = field(default_factory=DecisionTrainingArguments)
    decision: DecisionTaskArguments = field(default_factory=DecisionTaskArguments)

    def __post_init__(self):
        super().__post_init__()
        from veomni.models.decision import get_decision_model

        get_decision_model(self.decision.decision_model)
