"""FSDP2 integration test: validates VeOmni FSDP2 against single-GPU HuggingFace baseline.

Usage:
    pytest test_fsdp2_equivalence.py::test_generate_baseline_via_subprocess  # Generate baseline
    pytest test_fsdp2_equivalence.py::test_qwen3_fsdp2_via_subprocess        # Run FSDP2 test
"""

import contextlib
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

import veomni.distributed.parallel_state as ps_module
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.models import build_foundation_model, build_tokenizer
from veomni.optim import build_optimizer
from veomni.utils.device import empty_cache, get_device_type, get_dist_comm_backend, get_torch_device, set_device
from veomni.utils.helper import enable_full_determinism
from veomni.utils.loss_utils import count_loss_token, mean_global_loss

from .dataset import build_test_dataloader
from .results_io import (
    load_training_results,
    save_training_results,
    verify_config_compatibility,
)
from .test_parameters import TestArguments, parse_test_arguments
from .utils import (
    build_fsdp_model_optim,
    mean_global_loss_baseline,
    unapply_all_veomni_patches,
    verify_model_backend,
)


# =============================================================================
# Constants
# =============================================================================
NPROC_PER_NODE = 8  # Fixed for 8-GPU test environments
QWEN3_MODEL_PATH = "/mnt/hdfs/models/Qwen3-0.6B"


def global_grad_norm_pre_clip(model, is_distributed: bool) -> float:
    """Calculate global gradient L2 norm before clipping."""
    local_sq = torch.zeros((), device=get_device_type())
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        with contextlib.suppress(AttributeError):
            g = g.to_local()
        local_sq += (g.float() ** 2).sum()

    if is_distributed and dist.is_initialized():
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)

    return local_sq.sqrt().item()


def assert_labels_shift_correctness(batch: dict, sp_enabled: bool, context: str = "") -> None:
    """Assert that labels are shifted correctly based on SP setting.

    When SP is disabled:
        - labels[i] should equal input_ids[i] (no shift applied by collator)
        - The model itself handles the causal LM shift internally

    When SP is enabled:
        - labels[i] should equal input_ids[i+1] (shifted by SequenceParallelCollator)
        - The last position of labels is IGNORE_INDEX (padding after shift)
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    IGNORE_INDEX = -100

    if sp_enabled:
        shifted_input_ids = input_ids[..., 1:]
        labels_to_check = labels[..., :-1]

        valid_mask = labels_to_check != IGNORE_INDEX
        if not valid_mask.any():
            print(f"[Assert] {context}SP enabled: no valid labels to check (all IGNORE_INDEX)")
            return

        matches = (shifted_input_ids == labels_to_check) | (labels_to_check == IGNORE_INDEX)
        all_match = matches.all().item()

        assert all_match, (
            f"{context}SP enabled but labels not properly shifted. "
            f"Expected labels[:-1] == input_ids[1:] for all non-IGNORE_INDEX positions. "
            f"Mismatched positions: {(~matches).sum().item()}"
        )
        print(f"[Assert] {context}SP enabled: labels correctly shifted (exact match verified)")
    else:
        valid_mask = labels != IGNORE_INDEX
        if not valid_mask.any():
            print(f"[Assert] {context}SP disabled: no valid labels to check (all IGNORE_INDEX)")
            return

        matches = (input_ids == labels) | (labels == IGNORE_INDEX)
        all_match = matches.all().item()

        assert all_match, (
            f"{context}SP disabled but labels do not match input_ids. "
            f"Expected labels == input_ids for all non-IGNORE_INDEX positions. "
            f"Mismatched positions: {(~matches).sum().item()}"
        )
        print(f"[Assert] {context}SP disabled: labels correctly unshifted (exact match verified)")


def assert_hf_baseline_uses_original_implementations(model, attn_implementation: str) -> None:
    """Assert that model uses HuggingFace original loss and attention implementations."""
    from transformers.integrations.flash_attention import flash_attention_forward as hf_flash_attention_forward
    from transformers.loss.loss_utils import ForCausalLMLoss as HFForCausalLMLoss
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model_loss_fn = model.loss_function
    assert model_loss_fn is HFForCausalLMLoss, (
        f"model.loss_function is not HuggingFace's ForCausalLMLoss. "
        f"Got: {model_loss_fn.__module__}.{model_loss_fn.__name__}. "
        f"Expected: {HFForCausalLMLoss.__module__}.{HFForCausalLMLoss.__name__}. "
        "VeOmni patches may not have been properly unpatched."
    )
    print("[HF Baseline] model.loss_function uses HuggingFace original ForCausalLMLoss")

    if attn_implementation in ALL_ATTENTION_FUNCTIONS:
        actual_attn_fn = ALL_ATTENTION_FUNCTIONS[attn_implementation]
        assert actual_attn_fn is hf_flash_attention_forward, (
            f"ALL_ATTENTION_FUNCTIONS['{attn_implementation}'] is not HuggingFace's flash_attention_forward. "
            f"Got: {actual_attn_fn.__module__}.{actual_attn_fn.__name__}. "
            f"Expected: {hf_flash_attention_forward.__module__}.{hf_flash_attention_forward.__name__}. "
            "VeOmni patches may not have been properly unpatched."
        )
        print(f"[HF Baseline] ALL_ATTENTION_FUNCTIONS['{attn_implementation}'] uses HuggingFace original")


def run_training_single_gpu_baseline(
    config_path: str,
    data_path: str,
    tokenizer,
    global_batch_size: int,
    max_seq_len: int,
    num_train_steps: int,
    weights_path: str = None,
    attn_implementation: str = "flash_attention_2",
) -> dict:
    """Run single-GPU HuggingFace baseline training."""
    enable_full_determinism(42)
    print(f"\n[HF Baseline] Starting (padded_bsh, {attn_implementation})")
    if weights_path:
        print(f"[HF Baseline] Weights: {weights_path}")

    os.environ["MODELING_BACKEND"] = "hf"

    # Restore HuggingFace original implementations that were patched during `import veomni`
    unapply_all_veomni_patches()

    model = build_foundation_model(
        config_path=config_path,
        weights_path=weights_path,
        torch_dtype="bfloat16",
        attn_implementation=attn_implementation,
        init_device=get_device_type(),
    )

    # Verify that the model is from HuggingFace, not VeOmni.
    verify_model_backend(model, expected_backend="hf")

    # Double-check that loss and attention functions are HuggingFace originals
    assert_hf_baseline_uses_original_implementations(model, attn_implementation)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.train()

    optimizer = build_optimizer(
        model,
        lr=1e-4,
        weight_decay=0,
        fused=True,
        optimizer_type="adamw",
        no_decay_modules=[],
        no_decay_params=[],
    )

    dataloader = build_test_dataloader(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=global_batch_size,
        max_seq_len=max_seq_len,
        data_format="padded_bsh",
        is_distributed=False,
        num_train_steps=num_train_steps,
    )

    print(f"[HF Baseline] samples={len(dataloader.dataset)}, batch_size={global_batch_size}, steps={num_train_steps}")

    # Verify labels are NOT shifted for baseline (SP disabled)
    first_batch = next(iter(dataloader))
    assert_labels_shift_correctness(first_batch, sp_enabled=False, context="[HF Baseline] ")

    per_step_losses = []
    per_step_grad_norms = []
    dataloader_iter = iter(dataloader)

    for step in range(num_train_steps):
        optimizer.zero_grad()

        try:
            batch = next(dataloader_iter)
        except StopIteration:
            raise RuntimeError(f"Ran out of data at step {step}") from None

        micro_batch_token_len = count_loss_token(batch)
        micro_batches_token_len = micro_batch_token_len.copy()

        batch = {
            k: (v.to(device=get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        if micro_batch_token_len["foundation_tokens"].item() == 0:
            raise RuntimeError(f"No valid tokens in batch at step {step}")

        outputs = model(**batch, use_cache=False)

        loss_bwd, loss_dict = mean_global_loss_baseline(outputs.loss, micro_batch_token_len, micro_batches_token_len)

        loss_bwd.backward()

        pre_clip_gn = global_grad_norm_pre_clip(model, is_distributed=False)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_loss = loss_dict.get("foundation_loss", loss_bwd.item())
        per_step_losses.append(avg_loss)
        per_step_grad_norms.append(pre_clip_gn)

        print(f"[HF Baseline] Step {step + 1}/{num_train_steps}: loss={avg_loss:.6f}, grad_norm={pre_clip_gn:.6f}")
        empty_cache()

    results = {
        "per_step_losses": per_step_losses,
        "per_step_grad_norms": per_step_grad_norms,
        "num_steps": num_train_steps,
    }
    return results


@dataclass
class FSDP2TrainingConfig:
    """Configuration for FSDP2 distributed training."""

    config_path: str
    data_path: str
    tokenizer: any
    rank: int
    world_size: int
    data_format: str
    global_batch_size: int
    micro_batch_size: int
    max_seq_len: int
    num_train_steps: int
    attn_implementation: str = "flash_attention_2"
    weights_path: Optional[str] = None
    ulysses_sp_size: int = 1

    @property
    def dp_size(self) -> int:
        return self.world_size // self.ulysses_sp_size

    @property
    def mode_label(self) -> str:
        if self.ulysses_sp_size > 1:
            return f"FSDP2+Ulysses + {self.data_format}"
        return f"VeOmni FSDP2 + {self.data_format}"


def run_training_fsdp2_distributed(config: FSDP2TrainingConfig) -> Optional[dict]:
    """Run VeOmni FSDP2 training with optional Ulysses SP. Returns results on rank 0 only."""
    enable_full_determinism(42)

    rank = config.rank
    world_size = config.world_size
    ulysses_sp_size = config.ulysses_sp_size
    dp_size = config.dp_size
    label = config.mode_label

    if ulysses_sp_size > 1 and world_size != dp_size * ulysses_sp_size:
        raise ValueError(
            f"Invalid configuration: world_size ({world_size}) != "
            f"dp_size ({dp_size}) * ulysses_sp_size ({ulysses_sp_size})"
        )

    if rank == 0:
        print(f"\n[{label}] Starting (attn={config.attn_implementation}, world={world_size}, dp={dp_size})")
        if ulysses_sp_size > 1:
            print(f"[{label}] Ulysses SP size: {ulysses_sp_size}")
        if config.weights_path:
            print(f"[{label}] Weights: {config.weights_path}")

    model, optimizer = build_fsdp_model_optim(
        config_path=config.config_path,
        weights_path=config.weights_path,
        attn_implementation=config.attn_implementation,
        dp_size=dp_size,
        ulysses_sp_size=ulysses_sp_size,
        torch_dtype="float32",
        enable_mixed_precision=True,
    )

    dataloader = build_test_dataloader(
        data_path=config.data_path,
        tokenizer=config.tokenizer,
        batch_size=config.micro_batch_size,
        max_seq_len=config.max_seq_len,
        data_format=config.data_format,
        is_distributed=True,
        num_train_steps=config.num_train_steps,
        global_batch_size=config.global_batch_size,
    )

    if rank == 0:
        print(
            f"[{label}] samples={len(dataloader.dataset)}, global_bs={config.global_batch_size}, "
            f"micro_bs={config.micro_batch_size}, steps={config.num_train_steps}"
        )

    # Verify labels shift correctness based on SP setting
    sp_enabled = ulysses_sp_size > 1
    first_batch = next(iter(dataloader))
    if rank == 0:
        assert_labels_shift_correctness(first_batch, sp_enabled=sp_enabled, context=f"[{label}] ")

    model.train()
    per_step_losses = []
    per_step_grad_norms = []
    grad_acc_steps = config.global_batch_size // (config.micro_batch_size * config.dp_size)

    if rank == 0:
        print(f"[{label}] grad_acc_steps={grad_acc_steps}")

    dataloader_iter = iter(dataloader)

    for step in range(config.num_train_steps):
        optimizer.zero_grad()

        all_batches = []
        for acc_step in range(grad_acc_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                raise RuntimeError(f"Ran out of data at step {step}, acc_step {acc_step}") from None

            batch = {
                k: (v.to(device=get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            all_batches.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()})

        micro_batches_token_len = count_loss_token(all_batches)

        if micro_batches_token_len["foundation_tokens"].item() == 0:
            raise RuntimeError(f"No valid tokens were processed at step {step}")

        total_loss = 0.0
        for acc_step, batch in enumerate(all_batches):
            micro_batch_token_len = count_loss_token(batch)
            batch = {
                k: (v.to(device=get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            if micro_batch_token_len["foundation_tokens"].item() == 0:
                continue

            sync_now = acc_step == len(all_batches) - 1
            ctx = model.no_sync() if hasattr(model, "no_sync") and not sync_now else torch.enable_grad()

            with ctx:
                outputs = model(**batch, use_cache=False)
                loss_dict = mean_global_loss(outputs.loss, micro_batch_token_len, micro_batches_token_len)
                loss_bwd = sum(loss_dict.values())
                total_loss += loss_dict.get("foundation_loss", loss_dict.get("loss", 0.0))
                loss_bwd.backward()

        pre_clip_gn = global_grad_norm_pre_clip(model, is_distributed=True)
        veomni_clip_grad_norm(model, max_norm=1.0)
        optimizer.step()

        avg_loss = torch.tensor(total_loss, device=get_device_type(), dtype=torch.float32)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = (avg_loss / ps_module.get_parallel_state().fsdp_size).item()

        per_step_losses.append(avg_loss)
        per_step_grad_norms.append(pre_clip_gn)

        if rank == 0:
            print(
                f"[{label}] Step {step + 1}/{config.num_train_steps}: loss={avg_loss:.6f}, grad_norm={pre_clip_gn:.6f}"
            )

    results = {
        "per_step_losses": per_step_losses,
        "per_step_grad_norms": per_step_grad_norms,
        "num_steps": config.num_train_steps,
    }
    return results if rank == 0 else None


def initialize_distributed_environment():
    """Initialize distributed environment. Returns (local_rank, rank, world_size)."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("ERROR: Must run with torchrun in distributed mode")
        sys.exit(1)

    if not get_torch_device().is_available():
        print("ERROR: Accelerator device not available")
        sys.exit(1)

    gpu_count = get_torch_device().device_count()
    if gpu_count < 2:
        print(f"ERROR: Requires at least 2 GPUs, found {gpu_count}")
        sys.exit(1)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=get_dist_comm_backend())

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return local_rank, rank, world_size


def cleanup_distributed_environment():
    """Clean up distributed environment."""
    dist.barrier()
    ps_module._PARALLEL_STATE = None
    if dist.is_initialized():
        dist.destroy_process_group()


def validate_test_arguments(args: TestArguments, mode: str) -> list[str]:
    """Validate test arguments and return list of error messages."""
    errors = []

    if args.data_path is None:
        errors.append("--data_path is required")
    if args.global_batch_size is None:
        errors.append("--global_batch_size is required")
    if args.max_seq_len is None:
        errors.append("--max_seq_len is required")
    if args.num_train_steps is None:
        errors.append("--num_train_steps is required")

    if mode == "generate_baseline":
        if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
            errors.append("Baseline generation must run without torchrun")
    elif mode == "test":
        if args.rtol_loss is None:
            errors.append("--rtol_loss is required for test mode")
        if args.rtol_grad_norm is None:
            errors.append("--rtol_grad_norm is required for test mode")
        if args.micro_batch_size is None:
            errors.append("--micro_batch_size is required for test mode")

    return errors


def check_and_exit_on_errors(errors: list[str], rank: int = 0):
    """Print errors and exit if there are any validation errors."""
    if errors:
        if rank == 0:
            for error in errors:
                print(f"ERROR: {error}")
        sys.exit(1)


def run_baseline_generation(args: TestArguments):
    """Generate baseline results using single-GPU HuggingFace training."""
    errors = validate_test_arguments(args, mode="generate_baseline")
    check_and_exit_on_errors(errors)

    config_path = QWEN3_MODEL_PATH
    weights_path = QWEN3_MODEL_PATH

    print(f"Model: {args.model_name}, Config: {config_path}, Weights: {weights_path}")
    print(f"Data: {args.data_path}, Output: {args.baseline_dir}")

    try:
        tokenizer = build_tokenizer(QWEN3_MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Tokenizer not available - {e}")
        sys.exit(1)

    attn_implementation = "flash_attention_2"

    baseline_results = run_training_single_gpu_baseline(
        config_path=config_path,
        data_path=args.data_path,
        tokenizer=tokenizer,
        weights_path=weights_path,
        global_batch_size=args.global_batch_size,
        max_seq_len=args.max_seq_len,
        num_train_steps=args.num_train_steps,
        attn_implementation=attn_implementation,
    )

    baseline_config = {
        "model_name": args.model_name,
        "implementation": "HuggingFace",
        "data_format": "padded_bsh",
        "attn_implementation": attn_implementation,
        "num_gpus": 1,
        "global_batch_size": args.global_batch_size,
        "max_seq_len": args.max_seq_len,
        "num_train_steps": args.num_train_steps,
        "data_path": args.data_path,
    }

    save_training_results(
        results=baseline_results,
        output_dir=args.baseline_dir,
        run_name="baseline",
        config=baseline_config,
    )

    print("BASELINE GENERATION COMPLETE")
    for step in range(args.num_train_steps):
        loss = baseline_results["per_step_losses"][step]
        grad_norm = baseline_results["per_step_grad_norms"][step]
        print(f"Step {step + 1}: loss={loss:.6f}, grad_norm={grad_norm:.6f}")
    print(f"Saved to: {args.baseline_dir}")


def run_fsdp2_test(args: TestArguments):
    """Run FSDP2 test against pre-generated baseline."""
    _, rank, world_size = initialize_distributed_environment()

    try:
        run_fsdp2_test_impl(args, rank, world_size)
    finally:
        cleanup_distributed_environment()


def run_fsdp2_test_impl(args: TestArguments, rank: int, world_size: int):
    """FSDP2 test implementation supporting FSDP2 and FSDP2+Ulysses."""
    errors = validate_test_arguments(args, mode="test")
    check_and_exit_on_errors(errors, rank=rank)

    ulysses_sp_size = args.ulysses_sp_size
    dp_size = world_size // ulysses_sp_size

    if ulysses_sp_size > 1 and world_size != dp_size * ulysses_sp_size:
        if rank == 0:
            print(f"ERROR: world_size ({world_size}) != dp_size ({dp_size}) * ulysses_sp_size ({ulysses_sp_size})")
        sys.exit(1)

    config_path = QWEN3_MODEL_PATH
    weights_path = QWEN3_MODEL_PATH

    if ulysses_sp_size > 1:
        mode_label = "FSDP2+Ulysses"
    else:
        mode_label = "VeOmni FSDP2"

    if rank == 0:
        print(f"TEST: {mode_label} ({args.data_format}, {args.attn_implementation}) vs HF Baseline")
        print(f"Model: {args.model_name}, Config: {config_path}, Data: {args.data_path}")
        parallelism_info = f"world={world_size}, dp={dp_size}"
        if ulysses_sp_size > 1:
            parallelism_info += f", ulysses_sp={ulysses_sp_size}"
        print(parallelism_info)

    try:
        tokenizer = build_tokenizer(QWEN3_MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Tokenizer not available - {e}")
        sys.exit(1)

    baseline_data = None
    if rank == 0:
        try:
            baseline_data = load_training_results(args.baseline_dir, "baseline")
            test_config = {
                "model_name": args.model_name,
                "global_batch_size": args.global_batch_size,
                "max_seq_len": args.max_seq_len,
                "num_train_steps": args.num_train_steps,
            }
            verify_config_compatibility(baseline_data["config"], test_config)
            print("[Rank 0] Baseline loaded")
            for step in range(args.num_train_steps):
                loss = baseline_data["results"]["per_step_losses"][step]
                grad_norm = baseline_data["results"]["per_step_grad_norms"][step]
                print(f"[Rank 0] Baseline Step {step + 1}: loss={loss:.6f}, grad_norm={grad_norm:.6f}")
        except Exception as e:
            print(f"[ERROR] Failed to load baseline: {e}")
            traceback.print_exc()

    baseline_data_list = [baseline_data]
    dist.broadcast_object_list(baseline_data_list, src=0)
    baseline_data = baseline_data_list[0]

    if baseline_data is None:
        if rank == 0:
            print("FAILED: Could not load baseline. Run baseline generation first.")
        sys.exit(1)

    baseline_results = baseline_data["results"]

    if rank == 0:
        print(f"Running {mode_label} + {args.data_format}")

    training_config = FSDP2TrainingConfig(
        config_path=config_path,
        data_path=args.data_path,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        data_format=args.data_format,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        num_train_steps=args.num_train_steps,
        attn_implementation=args.attn_implementation,
        weights_path=weights_path,
        ulysses_sp_size=ulysses_sp_size,
    )

    test_results = run_training_fsdp2_distributed(training_config)

    dist.barrier()

    if rank != 0:
        return

    time.sleep(1.0)
    result_label = f"{mode_label}+{args.data_format}"
    print(f"\nCOMPARISON: HF Baseline vs {result_label}")

    baseline_losses = baseline_results["per_step_losses"]
    test_losses = test_results["per_step_losses"]
    baseline_grad_norms = baseline_results["per_step_grad_norms"]
    test_grad_norms = test_results["per_step_grad_norms"]

    print("\n" + "-" * 100)
    print(f"{'Step':<10} {'Metric':<20} {'HF Baseline':<25} {result_label:<25} {'Diff':<15}")
    print("-" * 100)

    for step in range(args.num_train_steps):
        print(
            f"{step + 1:<10} {'Loss':<20} {baseline_losses[step]:<25.6f} "
            f"{test_losses[step]:<25.6f} {abs(baseline_losses[step] - test_losses[step]):<15.6f}"
        )
        print(
            f"{step + 1:<10} {'Grad Norm':<20} {baseline_grad_norms[step]:<25.6f} "
            f"{test_grad_norms[step]:<25.6f} {abs(baseline_grad_norms[step] - test_grad_norms[step]):<15.6f}"
        )
    print("-" * 100)

    print("\nVerifying metrics...")
    for step in range(args.num_train_steps):
        torch.testing.assert_close(
            torch.tensor(test_losses[step]),
            torch.tensor(baseline_losses[step]),
            rtol=args.rtol_loss,
            atol=1e-4,
            msg=lambda msg: f"Step {step + 1} loss: {msg}",
        )
        print(f"Step {step + 1} loss matches (rtol={args.rtol_loss})")

        torch.testing.assert_close(
            torch.tensor(test_grad_norms[step]),
            torch.tensor(baseline_grad_norms[step]),
            rtol=args.rtol_grad_norm,
            atol=1e-4,
            msg=lambda msg: f"Step {step + 1} grad norm: {msg}",
        )
        print(f"Step {step + 1} grad norm matches (rtol={args.rtol_grad_norm})")

    print("\n" + "=" * 80)
    print("ALL CHECKS PASSED")
    print("=" * 80 + "\n")


def main():
    """Main entry point for training script."""
    args = parse_test_arguments()

    if args.mode == "generate_baseline":
        run_baseline_generation(args)
    elif args.mode == "test":
        run_fsdp2_test(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


# =============================================================================
# Qwen3 Dense Model Test Configuration
# =============================================================================
# Tolerance analysis based on empirical results (GPU):
#   FSDP2+rmpad_with_pos_ids: loss diff max ~0.024%, grad_norm diff max ~0.073%
#   FSDP2+Ulysses+rmpad_with_pos_ids: loss diff max ~0.027%, grad_norm diff max ~0.099%
QWEN3_TEST_CONFIG = {
    "dataset_path": os.path.join(os.path.dirname(__file__), "data", "fineweb_64_qwen3_tokenizer_4096.parquet"),
    "baseline_dir": ".pytest_cache/baseline_outputs",
    "model_name": "Qwen/Qwen3-0.6B",
    "global_batch_size": 32,
    "micro_batch_size": 4,
    "max_seq_len": 512,
    "num_train_steps": 2,
    "data_format": "rmpad_with_pos_ids",
    "attn_implementation": "veomni_flash_attention_2_with_sp",
    "ulysses_sp_size": 4,
    "rtol_loss": 0.001,  # 0.1%
    "rtol_grad_norm": 0.006,  # 0.6%
}


def test_generate_baseline_via_subprocess():
    """Generate baseline via subprocess (single GPU)."""
    baseline_command = [
        "python",
        "-m",
        "tests.distributed.test_fsdp2_equivalence",
        "--mode",
        "generate_baseline",
        "--baseline_dir",
        QWEN3_TEST_CONFIG["baseline_dir"],
        "--model_name",
        QWEN3_TEST_CONFIG["model_name"],
        "--data_path",
        QWEN3_TEST_CONFIG["dataset_path"],
        "--global_batch_size",
        str(QWEN3_TEST_CONFIG["global_batch_size"]),
        "--micro_batch_size",
        str(QWEN3_TEST_CONFIG["micro_batch_size"]),
        "--max_seq_len",
        str(QWEN3_TEST_CONFIG["max_seq_len"]),
        "--num_train_steps",
        str(QWEN3_TEST_CONFIG["num_train_steps"]),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Run from VeOmni root directory
    veomni_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    print(f"\n{'=' * 60}")
    print(f"Baseline generation: {QWEN3_TEST_CONFIG['model_name']}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(baseline_command, env=env, cwd=veomni_dir, check=True)
    assert result.returncode == 0


def test_qwen3_fsdp2_via_subprocess():
    """Test FSDP2 via subprocess (multi-GPU with torchrun)."""
    gpu_count = get_torch_device().device_count()
    if gpu_count < NPROC_PER_NODE:
        raise RuntimeError(f"FSDP2 test requires at least {NPROC_PER_NODE} GPUs, but only {gpu_count} available")

    micro_batch_size = QWEN3_TEST_CONFIG["global_batch_size"] // NPROC_PER_NODE

    fsdp_command = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={NPROC_PER_NODE}",
        "--master_port=4321",
        "-m",
        "tests.distributed.test_fsdp2_equivalence",
        "--mode",
        "test",
        "--baseline_dir",
        QWEN3_TEST_CONFIG["baseline_dir"],
        "--model_name",
        QWEN3_TEST_CONFIG["model_name"],
        "--data_path",
        QWEN3_TEST_CONFIG["dataset_path"],
        "--data_format",
        QWEN3_TEST_CONFIG["data_format"],
        "--attn_implementation",
        QWEN3_TEST_CONFIG["attn_implementation"],
        "--global_batch_size",
        str(QWEN3_TEST_CONFIG["global_batch_size"]),
        "--micro_batch_size",
        str(micro_batch_size),
        "--max_seq_len",
        str(QWEN3_TEST_CONFIG["max_seq_len"]),
        "--num_train_steps",
        str(QWEN3_TEST_CONFIG["num_train_steps"]),
        "--rtol_loss",
        str(QWEN3_TEST_CONFIG["rtol_loss"]),
        "--rtol_grad_norm",
        str(QWEN3_TEST_CONFIG["rtol_grad_norm"]),
    ]

    env = os.environ.copy()
    veomni_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    print(f"\n{'=' * 60}")
    print(f"FSDP2 test: {QWEN3_TEST_CONFIG['model_name']} (GPUs={NPROC_PER_NODE}, {QWEN3_TEST_CONFIG['data_format']})")
    print(f"{'=' * 60}\n")

    result = subprocess.run(fsdp_command, env=env, cwd=veomni_dir, check=True)
    assert result.returncode == 0


def test_qwen3_fsdp2_ulysses_via_subprocess():
    """Test FSDP2 + Ulysses SP via subprocess (multi-GPU with torchrun)."""
    gpu_count = get_torch_device().device_count()
    if gpu_count < NPROC_PER_NODE:
        raise RuntimeError(
            f"FSDP2 Ulysses test requires at least {NPROC_PER_NODE} GPUs, but only {gpu_count} available"
        )

    fsdp_size = NPROC_PER_NODE // QWEN3_TEST_CONFIG["ulysses_sp_size"]

    ulysses_command = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={NPROC_PER_NODE}",
        "--master_port=4321",
        "-m",
        "tests.distributed.test_fsdp2_equivalence",
        "--mode",
        "test",
        "--baseline_dir",
        QWEN3_TEST_CONFIG["baseline_dir"],
        "--model_name",
        QWEN3_TEST_CONFIG["model_name"],
        "--data_path",
        QWEN3_TEST_CONFIG["dataset_path"],
        "--data_format",
        QWEN3_TEST_CONFIG["data_format"],
        "--attn_implementation",
        QWEN3_TEST_CONFIG["attn_implementation"],
        "--global_batch_size",
        str(QWEN3_TEST_CONFIG["global_batch_size"]),
        "--micro_batch_size",
        str(QWEN3_TEST_CONFIG["micro_batch_size"]),
        "--max_seq_len",
        str(QWEN3_TEST_CONFIG["max_seq_len"]),
        "--num_train_steps",
        str(QWEN3_TEST_CONFIG["num_train_steps"]),
        "--ulysses_sp_size",
        str(QWEN3_TEST_CONFIG["ulysses_sp_size"]),
        "--rtol_loss",
        str(QWEN3_TEST_CONFIG["rtol_loss"]),
        "--rtol_grad_norm",
        str(QWEN3_TEST_CONFIG["rtol_grad_norm"]),
    ]

    env = os.environ.copy()
    veomni_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    print(f"\n{'=' * 60}")
    print(
        f"FSDP2+Ulysses test: {QWEN3_TEST_CONFIG['model_name']} "
        f"(GPUs={NPROC_PER_NODE}, fsdp={fsdp_size}, "
        f"ulysses={QWEN3_TEST_CONFIG['ulysses_sp_size']})"
    )
    print(f"{'=' * 60}\n")

    result = subprocess.run(ulysses_command, env=env, cwd=veomni_dir, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
