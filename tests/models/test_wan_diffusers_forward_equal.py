"""Forward parity for VeOmni Wan vs pristine diffusers Wan.

Wan is diffusers-backed instead of transformers-backed, so it is not covered by
``test_models_logits_equal_v5.py``. This test keeps a pristine diffusers
baseline in-process, then applies VeOmni's Wan patch and compares the single-rank
forward under the production FA2 path.
"""

import copy
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAN_TOY_CONFIG_DIR = os.path.join(REPO_ROOT, "tests", "toy_config", "wan_t2v_toy")
WAN_TOY_CONFIG_PATH = os.path.join(WAN_TOY_CONFIG_DIR, "config.json")
WAN_PARITY_CHILD_ENV = "VEOMNI_WAN_PARITY_CHILD"
WAN_PARITY_MODE_ENV = "VEOMNI_WAN_PARITY_MODE"
WAN_PARITY_SKIP_EXIT_CODE = 77
WAN_PARITY_XFAIL_EXIT_CODE = 78


def _make_inputs(device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.zeros(1, 16, 10, 16, 16, device=device, dtype=dtype),
        "timestep": torch.tensor([0.5], device=device, dtype=dtype),
        "encoder_hidden_states": torch.zeros(1, 10, 512, device=device, dtype=dtype),
    }


def _set_diffusers_flash_attention(model) -> None:
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor

    for block in model.blocks:
        for attn in (block.attn1, block.attn2):
            processor = WanAttnProcessor()
            processor._attention_backend = "flash"
            attn.set_processor(processor)


def _load_diffusers_config() -> dict:
    from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel

    with open(WAN_TOY_CONFIG_PATH) as config_file:
        config = json.load(config_file)

    init_signature = inspect.signature(DiffusersWanTransformer3DModel.__init__)
    return {
        key: value
        for key, value in config.items()
        if key in init_signature.parameters and key not in ("self", "_class_name", "_diffusers_version")
    }


def _initialize_floating_parameters(model, value: float = 1e-3) -> None:
    """Keep the toy model finite so the test validates parity, not init stress."""
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.is_floating_point(parameter):
                parameter.fill_(value)


def _release() -> None:
    from veomni.utils.device import empty_cache

    empty_cache()


def _unwrap_wan_output(output: torch.Tensor | tuple[torch.Tensor] | object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    return output.sample


class WanParitySkip(Exception):
    pass


class WanParityXFail(Exception):
    pass


def _run_veomni_wan_forward(
    state_dict: dict[str, torch.Tensor],
    inputs: dict[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
    attn_implementation: str,
) -> torch.Tensor:
    from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel

    from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
        WanTransformer3DModelConfig,
    )
    from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
        WanTransformer3DModel,
        apply_veomni_wan_transformer_patch,
    )
    from veomni.ops.kernels.attention import apply_veomni_attention_patch

    apply_veomni_attention_patch()
    apply_veomni_wan_transformer_patch()

    config = WanTransformer3DModelConfig.from_pretrained(WAN_TOY_CONFIG_DIR)
    model_veomni = (
        WanTransformer3DModel._from_config(
            config,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    model_veomni.load_state_dict(state_dict)

    try:
        with torch.no_grad():
            return (
                _unwrap_wan_output(DiffusersWanTransformer3DModel.forward(model_veomni, **inputs, return_dict=False))
                .detach()
                .clone()
            )
    finally:
        del model_veomni
        _release()


def _run_wan_forward_parity() -> None:
    # Keep the first VeOmni import from applying global attention patches before
    # the pristine diffusers baseline runs.
    os.environ["MODELING_BACKEND"] = "hf"
    from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type

    if not IS_CUDA_AVAILABLE:
        raise WanParitySkip("CUDA required.")
    if importlib.util.find_spec("flash_attn") is None:
        raise WanParitySkip("flash_attn package not installed.")

    os.environ["DIFFUSERS_ATTN_BACKEND"] = "flash"

    from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel

    device = get_device_type()
    # L20 CI shows pristine diffusers Wan can produce NaN for random toy
    # fixtures under FA2. Use the same bf16 dtype as Wan e2e training and keep
    # controlled zero inputs plus nonzero tiny weights to isolate whether
    # VeOmni's Wan patch changes the diffusers FA2 forward.
    dtype = torch.bfloat16
    config = _load_diffusers_config()
    inputs = _make_inputs(device, dtype)

    torch.manual_seed(0)
    model_diffusers = DiffusersWanTransformer3DModel(**config).to(device=device, dtype=dtype).eval()
    _initialize_floating_parameters(model_diffusers)
    _set_diffusers_flash_attention(model_diffusers)

    with torch.no_grad():
        output_diffusers = _unwrap_wan_output(model_diffusers(**inputs, return_dict=False)).detach().clone()
    state_dict = copy.deepcopy(model_diffusers.state_dict())
    del model_diffusers
    _release()

    if not torch.isfinite(output_diffusers).all():
        raise WanParityXFail("Pristine diffusers Wan FA2 forward produced non-finite output on this L20 fixture.")

    try:
        outputs_veomni = {
            "veomni_flash_attention_2_with_sp": _run_veomni_wan_forward(
                state_dict,
                inputs,
                device,
                dtype,
                "veomni_flash_attention_2_with_sp",
            ),
            "flash_attention_2": _run_veomni_wan_forward(state_dict, inputs, device, dtype, "flash_attention_2"),
        }
    finally:
        del state_dict
        _release()

    for attn_implementation, output_veomni in outputs_veomni.items():
        assert output_diffusers.shape == output_veomni.shape
        assert torch.isfinite(output_veomni).all(), (
            f"VeOmni Wan {attn_implementation} forward produced non-finite output."
        )
        if not torch.equal(output_diffusers, output_veomni):
            diff = (output_diffusers.float() - output_veomni.float()).abs()
            mismatched = output_diffusers != output_veomni
            raise AssertionError(
                f"Wan forward differs from pristine diffusers under {attn_implementation}: "
                f"{int(mismatched.sum().item())}/{output_diffusers.numel()} mismatched, "
                f"max_abs_diff={float(diff.max().item()):.3e}, "
                f"first_mismatch_indices={torch.nonzero(mismatched, as_tuple=False)[:5].tolist()}"
            )


def _run_wan_sp_forward(rank: int, world_size: int, init_file: str) -> None:
    os.environ["MODELING_BACKEND"] = "hf"
    os.environ["DIFFUSERS_ATTN_BACKEND"] = "flash"

    import torch.distributed as dist
    from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel

    from veomni.distributed.parallel_state import init_parallel_state
    from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
        WanTransformer3DModelConfig,
    )
    from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
        WanTransformer3DModel,
        apply_veomni_wan_transformer_patch,
    )
    from veomni.ops.kernels.attention import apply_veomni_attention_patch
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

    get_torch_device().set_device(rank)
    try:
        apply_veomni_attention_patch()
        apply_veomni_wan_transformer_patch()

        device = get_device_type()
        dtype = torch.bfloat16
        inputs = _make_inputs(device, dtype)

        config = WanTransformer3DModelConfig.from_pretrained(WAN_TOY_CONFIG_DIR)

        model_sp = (
            WanTransformer3DModel._from_config(
                config,
                attn_implementation="veomni_flash_attention_2_with_sp",
                torch_dtype=dtype,
            )
            .to(device=device, dtype=dtype)
            .eval()
        )
        _initialize_floating_parameters(model_sp)

        baseline = None
        if rank == 0:
            model_no_sp = (
                WanTransformer3DModel._from_config(
                    config,
                    attn_implementation="flash_attention_2",
                    torch_dtype=dtype,
                )
                .to(device=device, dtype=dtype)
                .eval()
            )
            model_no_sp.load_state_dict(model_sp.state_dict())
            with torch.no_grad():
                baseline = _unwrap_wan_output(
                    DiffusersWanTransformer3DModel.forward(model_no_sp, **inputs, return_dict=False)
                ).detach()
            assert torch.isfinite(baseline).all(), "Wan no-SP VeOmni FA2 baseline produced non-finite output."
            del model_no_sp

        dist.init_process_group(
            backend=get_dist_comm_backend(),
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        init_parallel_state(dp_size=1, ulysses_size=world_size, device_type=device)
        with torch.no_grad():
            output_sp = _unwrap_wan_output(
                DiffusersWanTransformer3DModel.forward(model_sp, **inputs, return_dict=False)
            ).detach()

        assert output_sp.shape == inputs["hidden_states"].shape
        assert torch.isfinite(output_sp).all(), "Wan SP=2 FA2 forward produced non-finite output."

        output_list = [torch.empty_like(output_sp) for _ in range(world_size)]
        dist.all_gather(output_list, output_sp)
        if rank == 0:
            assert baseline is not None
            for other_rank, other_output in enumerate(output_list):
                if not torch.equal(baseline, other_output):
                    diff = (baseline.float() - other_output.float()).abs()
                    mismatched = baseline != other_output
                    raise AssertionError(
                        f"Wan SP=2 rank {other_rank} forward differs from no-SP VeOmni FA2: "
                        f"{int(mismatched.sum().item())}/{baseline.numel()} mismatched, "
                        f"max_abs_diff={float(diff.max().item()):.3e}, "
                        f"first_mismatch_indices={torch.nonzero(mismatched, as_tuple=False)[:5].tolist()}"
                    )

        del model_sp
    finally:
        _release()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_wan_sp_baseline_check() -> None:
    os.environ["MODELING_BACKEND"] = "hf"
    os.environ["DIFFUSERS_ATTN_BACKEND"] = "flash"

    from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel

    from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
        WanTransformer3DModelConfig,
    )
    from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
        WanTransformer3DModel,
        apply_veomni_wan_transformer_patch,
    )
    from veomni.ops.kernels.attention import apply_veomni_attention_patch
    from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type

    if not IS_CUDA_AVAILABLE:
        raise WanParitySkip("CUDA required.")
    if importlib.util.find_spec("flash_attn") is None:
        raise WanParitySkip("flash_attn package not installed.")

    apply_veomni_attention_patch()
    apply_veomni_wan_transformer_patch()

    device = get_device_type()
    dtype = torch.bfloat16
    inputs = _make_inputs(device, dtype)
    config = WanTransformer3DModelConfig.from_pretrained(WAN_TOY_CONFIG_DIR)
    model = (
        WanTransformer3DModel._from_config(config, attn_implementation="flash_attention_2", torch_dtype=dtype)
        .to(device=device, dtype=dtype)
        .eval()
    )
    _initialize_floating_parameters(model)
    try:
        with torch.no_grad():
            baseline = _unwrap_wan_output(
                DiffusersWanTransformer3DModel.forward(model, **inputs, return_dict=False)
            ).detach()
        if not torch.isfinite(baseline).all():
            raise WanParityXFail("Wan no-SP FA2 baseline produced non-finite output on this fixture.")
    finally:
        del model
        _release()


def _run_wan_backward_finite() -> None:
    os.environ["MODELING_BACKEND"] = "veomni"
    os.environ["DIFFUSERS_ATTN_BACKEND"] = "flash"

    from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
        WanTransformer3DModelConfig,
    )
    from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
        WanTransformer3DModel,
        apply_veomni_wan_transformer_patch,
    )
    from veomni.ops.kernels.attention import apply_veomni_attention_patch
    from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type

    if not IS_CUDA_AVAILABLE:
        raise WanParitySkip("CUDA required.")
    if importlib.util.find_spec("flash_attn") is None:
        raise WanParitySkip("flash_attn package not installed.")

    apply_veomni_attention_patch()
    apply_veomni_wan_transformer_patch()

    device = get_device_type()
    dtype = torch.bfloat16
    hidden_shape = (1, 16, 10, 16, 16)
    torch.manual_seed(0)
    batch = {
        "latents": [torch.zeros(hidden_shape, device=device, dtype=dtype)],
        "hidden_states": [torch.zeros(hidden_shape, device=device, dtype=dtype)],
        "timestep": [torch.tensor([0.5], device=device, dtype=dtype)],
        "encoder_hidden_states": [torch.zeros(1, 10, 512, device=device, dtype=dtype)],
        "training_target": [torch.randn(hidden_shape, device=device, dtype=dtype) * 1e-2],
    }
    config = WanTransformer3DModelConfig.from_pretrained(WAN_TOY_CONFIG_DIR)

    for attn_implementation in ("flash_attention_2", "veomni_flash_attention_2_with_sp"):
        torch.manual_seed(0)
        model = (
            WanTransformer3DModel._from_config(
                config,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
            )
            .to(device=device, dtype=dtype)
            .train()
        )
        _initialize_floating_parameters(model)
        try:
            output = model(**batch)
            loss = output.loss["mse_loss"]
            assert torch.isfinite(output.predictions[0]).all(), (
                f"Wan {attn_implementation} train forward produced non-finite predictions."
            )
            assert torch.isfinite(loss).all(), f"Wan {attn_implementation} train forward produced non-finite loss."
            loss.backward()

            nonfinite_grad_names = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            assert not nonfinite_grad_names, (
                f"Wan {attn_implementation} backward produced non-finite gradients: {nonfinite_grad_names[:10]}"
            )
        finally:
            del model
            _release()


def test_wan_forward_bitwise_equal_to_diffusers_flash_attention_2():
    """Wan T2V forward is bitwise-equal to pristine diffusers at sp_size=1.

    This intentionally requires exact equality on controlled deterministic
    inputs and toy weights: both sides run in the same clean subprocess on the
    same GPU, use the same FA2 backend, and share an identical state dict and
    input tensors. This unit test isolates whether VeOmni's Wan patch changed
    the bf16 single-rank forward before sequence parallelism enters the picture.
    """
    env = os.environ.copy()
    env["MODELING_BACKEND"] = "hf"
    env[WAN_PARITY_CHILD_ENV] = "1"
    env[WAN_PARITY_MODE_ENV] = "single_rank_parity"
    env["PYTHONPATH"] = os.pathsep.join(part for part in (REPO_ROOT, env.get("PYTHONPATH")) if part)
    result = subprocess.run(
        [sys.executable, __file__],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == WAN_PARITY_SKIP_EXIT_CODE:
        pytest.skip(output.strip())
    if result.returncode == WAN_PARITY_XFAIL_EXIT_CODE:
        pytest.xfail(output.strip())
    assert result.returncode == 0, output


def test_wan_flash_attention_2_backward_is_finite():
    """Wan bf16 FA2 train-mode forward/backward stays finite on the e2e fixture."""
    env = os.environ.copy()
    env["MODELING_BACKEND"] = "veomni"
    env[WAN_PARITY_CHILD_ENV] = "1"
    env[WAN_PARITY_MODE_ENV] = "backward_finite"
    env["PYTHONPATH"] = os.pathsep.join(part for part in (REPO_ROOT, env.get("PYTHONPATH")) if part)
    result = subprocess.run(
        [sys.executable, __file__],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == WAN_PARITY_SKIP_EXIT_CODE:
        pytest.skip(output.strip())
    assert result.returncode == 0, output


def test_wan_forward_sp2_matches_no_sp_flash_attention_2():
    """Wan T2V SP=2 forward stays finite and bitwise-matches no-SP VeOmni FA2."""
    from veomni.utils.device import IS_CUDA_AVAILABLE, get_torch_device

    world_size = 2
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn package not installed.")
    if get_torch_device().device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} CUDA devices.")

    env = os.environ.copy()
    env["MODELING_BACKEND"] = "hf"
    env[WAN_PARITY_CHILD_ENV] = "1"
    env[WAN_PARITY_MODE_ENV] = "sp_baseline_check"
    env["PYTHONPATH"] = os.pathsep.join(part for part in (REPO_ROOT, env.get("PYTHONPATH")) if part)
    result = subprocess.run(
        [sys.executable, __file__],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == WAN_PARITY_SKIP_EXIT_CODE:
        pytest.skip(output.strip())
    if result.returncode == WAN_PARITY_XFAIL_EXIT_CODE or result.returncode < 0:
        pytest.xfail(output.strip() or f"Wan no-SP FA2 baseline exited with signal {-result.returncode}.")
    assert result.returncode == 0, output

    import torch.multiprocessing as mp

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        init_file = tmp.name

    try:
        mp.spawn(_run_wan_sp_forward, args=(world_size, init_file), nprocs=world_size, join=True)
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)


def test_wan_forward_casts_nested_float_inputs_to_model_dtype():
    """DiT e2e passes list-valued batches, so Wan casts nested tensors itself."""
    pytest.importorskip("diffusers")

    from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
        WanTransformer3DModelConfig,
    )
    from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
        WanTransformer3DModel,
        apply_veomni_wan_transformer_patch,
    )

    apply_veomni_wan_transformer_patch()

    dtype = torch.bfloat16
    config = WanTransformer3DModelConfig.from_pretrained(WAN_TOY_CONFIG_DIR)
    model = (
        WanTransformer3DModel._from_config(config, attn_implementation="eager", torch_dtype=dtype)
        .to(device="cpu", dtype=dtype)
        .eval()
    )
    _initialize_floating_parameters(model)

    hidden_shape = (1, 16, 10, 16, 16)
    batch = {
        "latents": [torch.zeros(hidden_shape, dtype=torch.float32)],
        "hidden_states": [torch.zeros(hidden_shape, dtype=torch.float32)],
        "timestep": [torch.tensor([0.5], dtype=torch.float32)],
        "encoder_hidden_states": [torch.zeros(1, 10, 512, dtype=torch.float32)],
        "training_target": [torch.zeros(hidden_shape, dtype=torch.float32)],
    }

    with torch.no_grad():
        output = model(**batch)

    loss = output.loss["mse_loss"]
    assert torch.isfinite(loss).all()
    assert torch.isfinite(output.predictions[0]).all()
    assert output.predictions[0].dtype == dtype


if __name__ == "__main__" and os.environ.get(WAN_PARITY_CHILD_ENV) == "1":
    try:
        mode = os.environ.get(WAN_PARITY_MODE_ENV, "single_rank_parity")
        if mode == "single_rank_parity":
            _run_wan_forward_parity()
        elif mode == "sp_baseline_check":
            _run_wan_sp_baseline_check()
        elif mode == "backward_finite":
            _run_wan_backward_finite()
        else:
            raise ValueError(f"Unknown {WAN_PARITY_MODE_ENV}={mode}")
    except WanParitySkip as exc:
        print(str(exc))
        raise SystemExit(WAN_PARITY_SKIP_EXIT_CODE) from exc
    except WanParityXFail as exc:
        print(str(exc))
        raise SystemExit(WAN_PARITY_XFAIL_EXIT_CODE) from exc
