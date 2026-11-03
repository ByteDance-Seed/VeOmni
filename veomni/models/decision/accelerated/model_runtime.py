"""Model lifecycle for registered decision architectures."""

import math

from veomni.checkpoint.dcp_checkpointer import ModelState
from veomni.models.model_runtime import VeOmniModelRuntime

from ..base import get_decision_model


class DecisionModelRuntime(VeOmniModelRuntime):
    def __init__(self, args, model_name="base", *, train, decision_model, decision_model_config):
        self.decision_model = decision_model
        self.decision_model_config = decision_model_config
        super().__init__(args, model_name, train=train)

    def _freeze_model_module(self):
        for name in self.args.freeze_modules:
            self.model.get_submodule(name).requires_grad_(False)
        super()._freeze_model_module()

    def _build_model(self):
        args = self.args
        self.spec = get_decision_model(self.decision_model)
        self.model = self.spec.build_model(
            args.model_path,
            config_path=args.config_path,
            decision_model_config=self.decision_model_config,
            init_device=args.accelerator.init_device,
            torch_dtype="float32" if args.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
            ops_implementation=args.ops_implementation,
            model_config=args.model_config,
            load_pretrained_weights=not self.skip_hf_weight_load,
        )
        self.model_config = self.model.config

    def _additional_weight_sources(self):
        get_sources = getattr(self.model, "get_additional_weight_sources", None)
        return get_sources() if get_sources is not None else ()

    def _build_model_assets(self):
        super()._build_model_assets()
        if self.tokenizer is None:
            raise ValueError("Decision preprocessing requires a checkpoint tokenizer")
        module = self.unwrapped_module
        position_hook = getattr(module, "get_position_id_func", lambda: None)()
        self.model_assets.extend(getattr(module, "get_assets", lambda: [])())
        self.preprocessor = self.spec.build_preprocessor(
            self.tokenizer, self.model_config, position_hook=position_hook
        )

    def clip_grad_norm(self):
        norm = super().clip_grad_norm()
        if not math.isfinite(norm):
            raise FloatingPointError("Nonfinite decision gradient norm")
        return norm

    def save_hf_or_lora(self, state, stage="step_end"):
        self.checkpoint.wait_for_pending_save()
        self.checkpoint.prepare_export(state, stage)
        module = self.unwrapped_module
        model_state = ModelState(self.model, parallel_state=self.parallel_state).state_dict()
        self.spec.export(
            self.checkpoint.hf_export_dir(state),
            model_state,
            model=module,
            model_assets=self.model_assets,
            source_dir=self.args.model_path,
            global_rank=self.parallel_state.global_rank,
        )
