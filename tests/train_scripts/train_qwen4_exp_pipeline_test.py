import json
import math
import os
from pathlib import Path
from typing import Dict

import torch

from veomni.arguments import parse_args
from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMTrainer


os.environ["NCCL_DEBUG"] = "OFF"


def process_dummy_example(example: dict, **kwargs):
    """Restore tensors serialized through the dummy parquet fixture."""
    return [{key: torch.as_tensor(value) for key, value in example.items()}]


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value


def _optimizer_state_entries(optimizer) -> int:
    optimizers = getattr(optimizer, "optimizers_dict", None)
    if optimizers is not None:
        return sum(len(sub_optimizer.state) for sub_optimizer in optimizers.values())
    return len(optimizer.state)


class Qwen4ExpPipelineTrainer(VLMTrainer):
    """Trainer-level smoke assertions for PLE+EP gradients and DCP resume."""

    def __init__(self, args: VeOmniVLMArguments):
        super().__init__(args)
        if self.base.model_config.model_type != "qwen4_exp":
            raise ValueError("Qwen4ExpPipelineTrainer requires a qwen4_exp model.")

        self.losses = []
        self.ple_grad_seen = False
        self.ple_grad_finite = True
        self.expert_grad_seen = False
        self.expert_grad_finite = True
        self.ple_params: Dict[str, torch.nn.Parameter] = {
            name: param
            for name, param in self.base.model.named_parameters()
            if ".ple.ple_embedding.ngram_embedding." in name
        }
        if not self.ple_params:
            raise AssertionError("The Qwen4-Exp smoke test found no PLE parameters.")
        self.expert_params: Dict[str, torch.nn.Parameter] = {
            name: param for name, param in self.base.model.named_parameters() if ".mlp.experts." in name
        }
        if not self.expert_params:
            raise AssertionError("The Qwen4-Exp smoke test found no expert parameters.")

        optimizer_groups = set(getattr(self.base.optimizer, "optimizers_dict", {}))
        expected_optimizer_groups = {"ple", "ep", "non_extra_parallel"}
        if not expected_optimizer_groups.issubset(optimizer_groups):
            raise AssertionError(
                f"Expected optimizer groups {sorted(expected_optimizer_groups)}, got {sorted(optimizer_groups)}."
            )
        self.optimizer_groups = sorted(optimizer_groups)

        def capture_parallel_grads(optimizer, args, kwargs):
            for params, seen_attr, finite_attr in (
                (self.ple_params, "ple_grad_seen", "ple_grad_finite"),
                (self.expert_params, "expert_grad_seen", "expert_grad_finite"),
            ):
                grad_seen = getattr(self, seen_attr)
                grad_finite = getattr(self, finite_attr)
                for param in params.values():
                    if param.grad is None:
                        continue
                    local_grad = _local_tensor(param.grad)
                    grad_finite &= bool(torch.isfinite(local_grad).all().item())
                    grad_seen |= bool(torch.count_nonzero(local_grad).item())
                setattr(self, seen_attr, grad_seen)
                setattr(self, finite_attr, grad_finite)

        # FSDP2 DTensor parameters do not reliably invoke Tensor hooks. Inspect
        # the reduced gradients immediately before the optimizer clears them.
        self.parallel_grad_hook_handle = self.base.optimizer.register_step_pre_hook(capture_parallel_grads)
        self.initial_ple = {}
        self.initial_expert = {}
        self.optimizer_state_restored = False

    def _build_model_assets(self):
        self.base.model_assets = []

    def _build_data_transform(self):
        self.base.data_transform = process_dummy_example

    @staticmethod
    def _parameter_state(params: Dict[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
        return {name: _local_tensor(param).cpu().clone() for name, param in params.items()}

    def _ple_state(self) -> Dict[str, torch.Tensor]:
        return self._parameter_state(self.ple_params)

    def _expert_state(self) -> Dict[str, torch.Tensor]:
        return self._parameter_state(self.expert_params)

    @staticmethod
    def _assert_restored(reference_path: Path, current: Dict[str, torch.Tensor], kind: str):
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        if reference.keys() != current.keys():
            raise AssertionError(f"{kind} parameter names changed across DCP resume.")
        for name in reference:
            torch.testing.assert_close(current[name], reference[name], rtol=0, atol=0)

    def on_train_begin(self):
        super().on_train_begin()
        expected_start_step = int(os.getenv("QWEN4_EXP_EXPECT_START_STEP", "0"))
        if self.base.state.global_step != expected_start_step:
            raise AssertionError(
                f"Expected Qwen4-Exp smoke to start at step {expected_start_step}, got {self.base.state.global_step}."
            )

        if self.base.args.train.checkpoint.load_path is not None:
            if _optimizer_state_entries(self.base.optimizer) == 0:
                raise AssertionError("Qwen4-Exp DCP resume did not restore optimizer state before training.")
            self.optimizer_state_restored = True
            checkpoint_path = Path(self.base.args.train.checkpoint.load_path)
            rank = self.base.args.train.global_rank
            self._assert_restored(
                checkpoint_path / f"qwen4_exp_ple_rank_{rank}.pt",
                self._ple_state(),
                "PLE",
            )
            self._assert_restored(
                checkpoint_path / f"qwen4_exp_expert_rank_{rank}.pt",
                self._expert_state(),
                "Expert",
            )

        self.initial_ple = self._ple_state()
        self.initial_expert = self._expert_state()

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None, aux_metrics=None):
        if loss is None or not math.isfinite(float(loss)):
            raise AssertionError(f"Qwen4-Exp smoke produced a non-finite loss: {loss}")
        self.losses.append(float(loss))
        super().on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm, aux_metrics=aux_metrics)

        checkpoint_path = (
            Path(self.base.args.train.checkpoint.save_path) / f"global_step_{self.base.state.global_step}"
        )
        if checkpoint_path.is_dir():
            rank = self.base.args.train.global_rank
            torch.save(self._ple_state(), checkpoint_path / f"qwen4_exp_ple_rank_{rank}.pt")
            torch.save(self._expert_state(), checkpoint_path / f"qwen4_exp_expert_rank_{rank}.pt")

    def on_train_end(self):
        super().on_train_end()
        final_ple = self._ple_state()
        final_expert = self._expert_state()
        ple_parameter_changed = any(
            not torch.equal(self.initial_ple[name], final_ple[name]) for name in self.initial_ple
        )
        expert_parameter_changed = any(
            not torch.equal(self.initial_expert[name], final_expert[name]) for name in self.initial_expert
        )
        if not self.losses:
            raise AssertionError("Qwen4-Exp smoke did not execute a training step.")
        if not self.ple_grad_seen or not self.ple_grad_finite:
            raise AssertionError("Qwen4-Exp smoke did not observe a finite non-zero PLE gradient.")
        if not self.expert_grad_seen or not self.expert_grad_finite:
            raise AssertionError("Qwen4-Exp smoke did not observe a finite non-zero expert gradient.")
        if not ple_parameter_changed:
            raise AssertionError("Qwen4-Exp smoke did not update any PLE parameter.")
        if not expert_parameter_changed:
            raise AssertionError("Qwen4-Exp smoke did not update any expert parameter.")

        optimizer_state_entries = _optimizer_state_entries(self.base.optimizer)
        if optimizer_state_entries == 0:
            raise AssertionError("Qwen4-Exp smoke optimizer has no state after stepping.")

        if self.base.args.train.global_rank == 0:
            output_dir = Path(self.base.args.train.checkpoint.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "qwen4_exp_pipeline_result.json").write_text(
                json.dumps(
                    {
                        "start_global_step": int(os.getenv("QWEN4_EXP_EXPECT_START_STEP", "0")),
                        "end_global_step": self.base.state.global_step,
                        "losses": self.losses,
                        "ple_grad_seen": self.ple_grad_seen,
                        "ple_grad_finite": self.ple_grad_finite,
                        "ple_parameter_changed": ple_parameter_changed,
                        "expert_grad_seen": self.expert_grad_seen,
                        "expert_grad_finite": self.expert_grad_finite,
                        "expert_parameter_changed": expert_parameter_changed,
                        "optimizer_state_restored": self.optimizer_state_restored,
                        "optimizer_state_entries": optimizer_state_entries,
                        "optimizer_groups": self.optimizer_groups,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    parsed_args = parse_args(VeOmniVLMArguments)
    Qwen4ExpPipelineTrainer(parsed_args).train()
