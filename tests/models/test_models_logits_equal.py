import copy
import gc
import importlib.util
import os
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, empty_cache, get_device_type, get_torch_device

# Importing `utils` now captures pristine HF class attributes (in its
# `_PRISTINE_HF_CLASSES` snapshot) before any veomni import has a chance to
# monkey-patch them. `apply_veomni_hf_unpatch()` restores them; we call it
# before every HF build so leaks from the previous test do not poison the
# current one.
from .utils import apply_veomni_hf_unpatch  # noqa: E402


# Must be set before `import veomni` so GPU kernel patches remain gated off.
# VEOMNI_USE_LIGER_KERNEL=0 disables Liger substitutions in qwen3 / qwen3_moe
# / deepseek_v3 gpu_patch.py. VEOMNI_USE_FUSED_KERNELS=0 additionally disables
# the deepseek_v3 Triton RoPE + batch-invariant RMSNorm path, which is the
# default when Liger is off.
os.environ.setdefault("VEOMNI_USE_LIGER_KERNEL", "0")
os.environ.setdefault("VEOMNI_USE_FUSED_KERNELS", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")
# Required by torch.use_deterministic_algorithms for cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class Case:
    """A test case pairing an HF model config with a veomni build.

    HF is random-initialised from the toy config; its state dict is then
    copied into the veomni model. `sync_weight_key` selects a layout
    adapter from `tests/models/weight_sync_adapters.py`; needed for MoE
    models whose veomni layout stacks experts into a single tensor.

    `attn_implementation` is passed through to both HF and veomni. For
    `"flash_attention_2"` the dtype must be bf16/fp16 (FA2 requirement),
    so each case pairs an attention backend with the appropriate dtype.
    """

    case_id: str
    path: str
    sync_weight_key: Optional[str]
    attn_implementation: str = "eager"
    dtype: str = "float32"


def _toy(name: str) -> str:
    return os.path.join(REPO_ROOT, "tests", "toy_config", name)


CASES = [
    # eager + fp32
    Case("qwen3-toy-eager", _toy("qwen3_toy"), sync_weight_key=None),
    Case("qwen3_moe-toy-eager", _toy("qwen3_moe_toy"), sync_weight_key="qwen3_moe"),
    Case("deepseek_v3-toy-eager", _toy("deepseek_v3_toy"), sync_weight_key="deepseek_v3"),
    # flash_attention_2 + bf16 (FA2 does not support fp32)
    Case(
        "qwen3-toy-fa2",
        _toy("qwen3_toy"),
        sync_weight_key=None,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case(
        "qwen3_moe-toy-fa2",
        _toy("qwen3_moe_toy"),
        sync_weight_key="qwen3_moe",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case(
        "deepseek_v3-toy-fa2",
        _toy("deepseek_v3_toy"),
        sync_weight_key="deepseek_v3",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
]


def _apply_determinism():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _release():
    gc.collect()
    if IS_CUDA_AVAILABLE:
        empty_cache()


def _build_hf_model(case: Case):
    """Return a device-resident, eval-mode HF model randomly initialised from config."""
    from transformers import AutoConfig, AutoModelForCausalLM

    apply_veomni_hf_unpatch()
    config = AutoConfig.from_pretrained(case.path)
    torch.manual_seed(0)
    get_torch_device().manual_seed_all(0)
    # Init directly on device so init-time buffers (e.g. rotary `inv_freq`)
    # use the same arithmetic path as the veomni build, which allocates under
    # `torch.device(get_device_type())` via its CustomizedModelingLoader.
    with torch.device(get_device_type()):
        model_hf = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=_DTYPE_MAP[case.dtype],
            attn_implementation=case.attn_implementation,
        )
    return model_hf.eval()


def _build_veomni_model(case: Case, hf_state_dict):
    """Return a device-resident, eval-mode veomni model with HF weights loaded."""
    from veomni.models.auto import build_foundation_model

    model = build_foundation_model(
        config_path=case.path,
        weights_path=None,
        torch_dtype=case.dtype,
        attn_implementation=case.attn_implementation,
        init_device=get_device_type(),
    )

    if case.sync_weight_key is not None:
        from .weight_sync_adapters import get_sync_weight_func

        sync_func = get_sync_weight_func(case.sync_weight_key)
        assert sync_func is not None, f"no sync func for {case.sync_weight_key}"
        sync_func(model.config, hf_state_dict, model)
    else:
        model.load_state_dict(hf_state_dict)

    return model.eval()


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_logits_bitwise_equal(case: Case):
    """Verify veomni forward logits are bitwise identical to native HF.

    Scope: transformers v4 model definition, single sequence, single GPU,
    no GPU kernel patching (Liger + Triton fused kernels both disabled).
    HF is random-initialised from the toy config; its state dict is synced
    to veomni via the layout adapters.

    Execution order is mandatory: the HF forward must run BEFORE any
    veomni model build, because `build_foundation_model` triggers
    `apply_veomni_*_patch` which monkey-patches HF module classes
    process-wide.
    """
    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if is_transformers_version_greater_or_equal_to("5.0.0"):
        pytest.skip("Scope is transformers v4 model definition only.")
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if not os.path.isdir(case.path):
        pytest.skip(f"Path not found: {case.path}")
    if case.attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn package not installed.")

    _apply_determinism()

    device_type = get_device_type()
    gen = torch.Generator(device=device_type).manual_seed(0)
    # Vocab floor of 32000 dodges special tokens across Qwen3 (151936),
    # Qwen3MoE (151936), and DeepseekV3 (129280).
    input_ids = torch.randint(0, 32000, (1, 32), device=device_type, dtype=torch.long, generator=gen)

    # --- HF phase (must precede any veomni model build) ---
    model_hf = _build_hf_model(case)
    with torch.no_grad():
        logits_hf = model_hf(input_ids=input_ids, use_cache=False).logits.detach().clone()
    hf_state_dict = copy.deepcopy(model_hf.state_dict())
    del model_hf
    _release()

    # --- veomni phase ---
    model_ve = _build_veomni_model(case, hf_state_dict)
    with torch.no_grad():
        logits_ve = model_ve(input_ids=input_ids, use_cache=False).logits.detach().clone()
    del model_ve, hf_state_dict
    _release()

    assert logits_hf.shape == logits_ve.shape, (
        f"[{case.case_id}] shape mismatch: hf={tuple(logits_hf.shape)} ve={tuple(logits_ve.shape)}"
    )

    if not torch.equal(logits_hf, logits_ve):
        diff = (logits_hf.float() - logits_ve.float()).abs()
        ne = logits_hf != logits_ve
        n_mis = int(ne.sum().item())
        total = logits_hf.numel()
        max_abs = float(diff.max().item())
        first_idx = torch.nonzero(ne, as_tuple=False)[:5].tolist()
        raise AssertionError(
            f"[{case.case_id}] logits not bitwise equal: "
            f"{n_mis}/{total} mismatched, max_abs_diff={max_abs:.3e}, "
            f"first_mismatch_indices={first_idx}"
        )
