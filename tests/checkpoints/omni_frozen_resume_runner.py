"""Two-rank runner for frozen SeedOmni module resume weight loading."""

from pathlib import Path
from unittest.mock import patch

import torch

from veomni.arguments import OmniArguments, build_module_runtime_args, parse_omni_args
from veomni.distributed import torch_parallelize
from veomni.models.seed_omni.accelerator.module_runtime import ModuleRuntime
from veomni.trainer.omni import OmniTrainer


MODULE_NAME = "frozen_llm"
WEIGHT_SENTINEL = 0.125


def _assert_all_parameters_equal_sentinel(model: torch.nn.Module) -> None:
    parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter_count += 1
        local_parameter = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        expected = torch.full_like(local_parameter, WEIGHT_SENTINEL)
        torch.testing.assert_close(
            local_parameter.detach(),
            expected,
            rtol=0,
            atol=0,
            msg=lambda message, parameter_name=name: (
                f"{parameter_name} was not restored from the frozen module's HF snapshot: {message}"
            ),
        )
    assert parameter_count > 0


def main() -> None:
    args = parse_omni_args(OmniArguments)
    fsdp_config = args.model.accelerator.fsdp_config
    assert fsdp_config.fsdp_mode == "fsdp2"
    assert fsdp_config.forward_prefetch is False

    # ModuleRuntime assumes the orchestrator has already initialized distributed
    # state. Only that setup half is needed for this focused runner — the graphs,
    # data pipeline and train loop are irrelevant to weight loading.
    OmniTrainer.setup_distributed(args)

    # Bypass `resolve_omni_model` (which would demand training/generation graph
    # YAML) and build just this one module's args off the split-checkpoint root.
    module_args = build_module_runtime_args(
        args._to_module_global_args(),
        args.model.model_path,
        {MODULE_NAME: {"model_path": MODULE_NAME}},
    )[MODULE_NAME]

    module_dcp_path = Path(args.train.checkpoint.load_path) / MODULE_NAME
    assert not module_dcp_path.exists(), "The frozen module must not have a DCP payload in this regression scenario."

    original_load_model_weights = torch_parallelize.load_model_weights
    with patch.object(
        torch_parallelize,
        "load_model_weights",
        wraps=original_load_model_weights,
    ) as load_model_weights:
        module_runtime = ModuleRuntime(module_args, module_name=MODULE_NAME, train=args.train)

    assert module_runtime.has_trainable_parameters is False
    assert load_model_weights.call_count == 1, "A frozen module with persistent state must load its HF snapshot."
    _assert_all_parameters_equal_sentinel(module_runtime.model)

    # Frozen modules get no checkpoint manager, so a resume must neither read the
    # missing module subdirectory nor alter the HF-loaded weights.
    module_runtime.load()
    _assert_all_parameters_equal_sentinel(module_runtime.model)


if __name__ == "__main__":
    main()
