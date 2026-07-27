"""Two-rank runner for frozen SeedOmni module resume weight loading."""

from pathlib import Path
from unittest.mock import patch

import torch

from veomni.arguments import VeOmniArguments, parse_args
from veomni.distributed import torch_parallelize
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.omni.omni_module_trainer import OmniModuleTrainer


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
    args = parse_args(VeOmniArguments)
    assert args.train.accelerator.fsdp_config.fsdp_mode == "fsdp2"
    assert args.train.accelerator.fsdp_config.forward_prefetch is False

    # OmniModuleTrainer assumes the outer orchestrator has already initialized
    # distributed state. Only that setup half is needed for this focused runner.
    outer = BaseTrainer.__new__(BaseTrainer)
    outer.args = args
    outer._setup()

    module_dcp_path = Path(args.train.checkpoint.load_path) / MODULE_NAME
    assert not module_dcp_path.exists(), "The frozen module must not have a DCP payload in this regression scenario."

    original_load_model_weights = torch_parallelize.load_model_weights
    with patch.object(
        torch_parallelize,
        "load_model_weights",
        wraps=original_load_model_weights,
    ) as load_model_weights:
        module_trainer = OmniModuleTrainer(args, module_name=MODULE_NAME)

    assert module_trainer.has_trainable_parameters is False
    assert load_model_weights.call_count == 1, "A frozen module with persistent state must load its HF snapshot."
    _assert_all_parameters_equal_sentinel(module_trainer.base.model)

    # Frozen modules own no DCP callback. A resume lifecycle event must therefore
    # neither read the missing module subdirectory nor alter the HF-loaded weights.
    module_trainer.on_train_begin(TrainerState())
    _assert_all_parameters_equal_sentinel(module_trainer.base.model)


if __name__ == "__main__":
    main()
