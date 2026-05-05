import copy
import gc
import importlib.util
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    empty_cache,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)

# Importing `hf_unpatch` here (rather than from `utils`) captures pristine HF
# class attributes before any veomni import has a chance to monkey-patch them,
# without dragging in the heavy `veomni.data` import chain (av/torchcodec).
# `apply_veomni_hf_unpatch()` restores them; we call it before every HF build
# so leaks from the previous test do not poison the current one.
from .hf_unpatch import apply_veomni_hf_unpatch  # noqa: E402


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
    """One bitwise-equal comparison between a native HF build and a VeOmni build.

    `sync_weight_key` -> a layout adapter in `weight_sync_adapters.py`;
    needed for MoE models whose VeOmni layout stacks experts.
    `attn_implementation` is forwarded to both sides; FA2 requires bf16.
    The HF class is resolved from `config.architectures[0]` (no per-case
    override). `forward_attr` targets a submodule (`model.thinker` for
    Omni) while state-dict load still happens at the root, so the talker /
    vision sub-modules get the same random init on both sides.
    """

    case_id: str
    path: str
    sync_weight_key: Optional[str]
    attn_implementation: str = "eager"
    dtype: str = "float32"
    forward_attr: Optional[str] = None


def _toy(name: str) -> str:
    return os.path.join(REPO_ROOT, "tests", "toy_config", name)


# Scope: transformers v4, single CUDA rank.
# Causal-LM cases use text-only input_ids (shape (1, 32), vocab floor
# 32000 to dodge every 151xxx multimodal placeholder id).
# VL / Omni cases add a dummy 2x2 image (1 placeholder token after merger)
# so the visual tower actually runs. Audio is a follow-up: Omni passes
# zero `audio_mask` to satisfy the patched asserts but no `input_features`,
# so the audio tower stays dark.
#
# Two MoE families (qwen3_vl_moe, qwen3_omni_moe) skip the eager+fp32
# variant: HF v4 `_init_weights` hard-codes bf16 for the language tower,
# so an fp32 case there would just be testing a half-cast model.
CASES = [
    # ---- text-only causal-LM ----
    # eager + fp32
    Case("qwen2_5-toy-eager", _toy("qwen25_toy"), sync_weight_key=None),
    Case("llama3_1-toy-eager", _toy("llama31_toy"), sync_weight_key=None),
    Case("seed_oss-toy-eager", _toy("seed_oss_toy"), sync_weight_key=None),
    Case("qwen3-toy-eager", _toy("qwen3_toy"), sync_weight_key=None),
    Case("qwen3_moe-toy-eager", _toy("qwen3_moe_toy"), sync_weight_key="qwen3_moe"),
    Case("deepseek_v3-toy-eager", _toy("deepseek_v3_toy"), sync_weight_key="deepseek_v3"),
    # flash_attention_2 + bf16 (FA2 does not support fp32)
    Case(
        "qwen2_5-toy-fa2",
        _toy("qwen25_toy"),
        sync_weight_key=None,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case(
        "llama3_1-toy-fa2",
        _toy("llama31_toy"),
        sync_weight_key=None,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case(
        "seed_oss-toy-fa2",
        _toy("seed_oss_toy"),
        sync_weight_key=None,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
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
    # ---- VLMs ----
    # eager + fp32
    Case("qwen2_vl-toy-eager", _toy("qwen2vl_toy"), sync_weight_key=None),
    Case("qwen2_5_vl-toy-eager", _toy("qwen25vl_toy"), sync_weight_key=None),
    Case("qwen3_vl-toy-eager", _toy("qwen3vl_toy"), sync_weight_key=None),
    # flash_attention_2 + bf16
    Case(
        "qwen2_vl-toy-fa2",
        _toy("qwen2vl_toy"),
        sync_weight_key=None,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case(
        "qwen2_5_vl-toy-fa2",
        _toy("qwen25vl_toy"),
        sync_weight_key=None,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case(
        "qwen3_vl-toy-fa2",
        _toy("qwen3vl_toy"),
        sync_weight_key=None,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    # qwen3_vl_moe: HF init forces bf16 for the language tower, so only
    # fa2+bf16 exercises a self-consistent dtype assignment. State-dict
    # layout already matches between HF and VeOmni (both stack
    # `experts.gate_up_proj` / `experts.down_proj`), so no sync adapter.
    Case(
        "qwen3_vl_moe-toy-fa2",
        _toy("qwen3vlmoe_toy"),
        sync_weight_key=None,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    # ---- Omni (thinker-only forward) ----
    # The Omni public class is a thinker+talker wrapper; we run forward on
    # `model.thinker` so the comparison stays focused on the thinker stack
    # (text + visual + audio). State-dict load still happens at the root,
    # so talker/audio/visual sub-modules get the same random init on both
    # sides.
    Case(
        "qwen2_5_omni-toy-eager",
        _toy("qwen25omni_toy"),
        sync_weight_key=None,
        forward_attr="thinker",
    ),
    Case(
        "qwen2_5_omni-toy-fa2",
        _toy("qwen25omni_toy"),
        sync_weight_key=None,
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_attr="thinker",
    ),
    # qwen3_omni_moe: HF forces bf16 on the thinker; thinker experts use
    # per-expert HF layout, stacked on the VeOmni side -> needs the
    # `qwen3_omni_moe` sync adapter from `weight_sync_adapters.py`.
    Case(
        "qwen3_omni_moe-toy-fa2",
        _toy("qwen3omni_toy"),
        sync_weight_key="qwen3_omni_moe",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_attr="thinker",
    ),
]


def _apply_determinism():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group():
    """Module-scoped 1-rank process group for the v4 patches' all_gather calls.

    Some patches (qwen2_5_vl) call `dist.all_gather` unconditionally — even
    at sp_size=1 — and crash without a default group. Teardown only fires
    if we created the group, so we don't yank state from unrelated tests.
    Mirrors the test bodies' skip gates so a skip-only environment doesn't
    pay an init cost.
    """
    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if not IS_CUDA_AVAILABLE or is_transformers_version_greater_or_equal_to("5.0.0"):
        yield
        return

    import torch.distributed as dist

    we_initialised = False
    if not dist.is_initialized():
        # Bind the accelerator before init_process_group: on multi-GPU CI
        # hosts the comm library can otherwise pick the wrong visible device
        # and hang/fail. The IS_CUDA_AVAILABLE gate above already ensures
        # we have an accelerator here.
        get_torch_device().set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend=get_dist_comm_backend(), rank=0, world_size=1)
        we_initialised = True
    try:
        yield
    finally:
        if we_initialised and dist.is_initialized():
            dist.destroy_process_group()


def _release():
    gc.collect()
    if IS_CUDA_AVAILABLE:
        empty_cache()


def _resolve_hf_class(config):
    """Look up the concrete HF class from `config.architectures[0]`.

    Every toy config carries the concrete class name there, so one lookup
    covers causal-LMs, VLMs, and Omni wrappers — the latter aren't in any
    `AutoModel*` mapping anyway.
    """
    import transformers

    arch = config.architectures[0]
    return getattr(transformers, arch)


def _build_hf_model(case: Case):
    """Return a device-resident, eval-mode HF model randomly initialised from config."""
    from transformers import AutoConfig

    apply_veomni_hf_unpatch()
    config = AutoConfig.from_pretrained(case.path)
    torch.manual_seed(0)
    get_torch_device().manual_seed_all(0)
    cls = _resolve_hf_class(config)
    target_dtype = _DTYPE_MAP[case.dtype]
    # `_from_config` inits weights directly at `target_dtype` — a
    # `cls(config).to(bf16)` path would init in fp32 first, producing
    # different random bytes than VeOmni's bf16-from-the-start build (RNG
    # output depends on tensor dtype). Allocating on-device matches
    # VeOmni's init path so rotary `inv_freq` and friends share arithmetic.
    with torch.device(get_device_type()):
        model_hf = cls._from_config(
            config,
            torch_dtype=target_dtype,
            attn_implementation=case.attn_implementation,
        )
    return model_hf.eval()


# Causal-LM toy configs that have no `vision_config` / `thinker_config` —
# every forward kwargs construction below stays a no-op for these.
_CAUSAL_LM_MODEL_TYPES = frozenset({"qwen2", "qwen3", "qwen3_moe", "llama", "seed_oss", "deepseek_v3"})


def _vision_section(config):
    """Return `(vision_config, image_token_id, audio_token_id)` or `(None,)*3`.

    Top-level VLs carry `vision_config` on the root; Omni nests it under
    `thinker_config`. The placeholder field was renamed `_index` -> `_id`
    between qwen2_5_omni and qwen3_omni_moe — accept either.
    """
    if config.model_type in _CAUSAL_LM_MODEL_TYPES:
        return None, None, None
    if hasattr(config, "thinker_config"):
        cfg = config.thinker_config
    else:
        cfg = config
    vision_config = getattr(cfg, "vision_config", None)
    if vision_config is None:
        return None, None, None
    # Don't `or` — token id 0 is technically valid and would short-circuit.
    image_token = getattr(cfg, "image_token_index", None)
    if image_token is None:
        image_token = getattr(cfg, "image_token_id", None)
    audio_token = getattr(cfg, "audio_token_index", None)
    if audio_token is None:
        audio_token = getattr(cfg, "audio_token_id", None)
    return vision_config, image_token, audio_token


def _make_inputs(case: Case, config, device, dtype):
    """Build `(input_ids, forward_kwargs)` for the case.

    Causal-LM: text-only `(1, 32)` ids, vocab floor 32000 dodges every
    151xxx multimodal placeholder.

    VL / Omni: same base ids but the first
    `n_tokens = grid_t * grid_h * grid_w // spatial_merge_size**2` positions
    hold `image_token_id`, alongside dummy `pixel_values` +
    `image_grid_thw = [[1, 2, 2]]`. `n_tokens` matches what the merger
    emits, so the patched `masked_scatter` consumes every embedding once.
    `pixel_values` shape comes from `vision_config` (qwen2/2.5: patch_size
    14; qwen3: 16). `image_mask` / `video_mask` (and `audio_mask` for
    Omni) are passed explicitly because the patches otherwise call
    `dist.all_gather` to recompute them. Audio is a follow-up: Omni
    passes zero `audio_mask` for the asserts but no `input_features`.
    """
    base_gen = torch.Generator(device=device).manual_seed(0)
    base_input_ids = torch.randint(0, 32000, (1, 32), device=device, dtype=torch.long, generator=base_gen)

    vision_config, image_token_id, _ = _vision_section(config)
    if vision_config is None:
        return base_input_ids, {}

    patch_size = getattr(vision_config, "patch_size", 14)
    temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)
    in_channels = getattr(vision_config, "in_channels", getattr(vision_config, "in_chans", 3))
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)

    grid_t, grid_h, grid_w = 1, spatial_merge_size, spatial_merge_size
    num_patches = grid_t * grid_h * grid_w
    n_tokens = num_patches // (spatial_merge_size**2)
    feat_dim = in_channels * temporal_patch_size * patch_size * patch_size

    pix_gen = torch.Generator(device=device).manual_seed(1)
    pixel_values = torch.randn(num_patches, feat_dim, dtype=dtype, device=device, generator=pix_gen)
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long, device=device)

    input_ids = base_input_ids.clone()
    input_ids[0, :n_tokens] = image_token_id

    image_mask = input_ids == image_token_id
    video_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    forward_kwargs = {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "image_mask": image_mask,
        "video_mask": video_mask,
    }
    if case.forward_attr == "thinker":
        forward_kwargs["audio_mask"] = torch.zeros_like(input_ids, dtype=torch.bool)
    return input_ids, forward_kwargs


def _forward_target(model, case: Case):
    """Return the submodule we run forward on (`model` itself by default)."""
    if case.forward_attr is None:
        return model
    target = model
    for part in case.forward_attr.split("."):
        target = getattr(target, part)
    return target


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

    from transformers import AutoConfig

    _apply_determinism()
    device_type = get_device_type()
    target_dtype = _DTYPE_MAP[case.dtype]
    config = AutoConfig.from_pretrained(case.path)
    input_ids, fwd_kwargs = _make_inputs(case, config, device_type, target_dtype)

    # --- HF phase (must precede any veomni model build) ---
    model_hf = _build_hf_model(case)
    with torch.no_grad():
        logits_hf = (
            _forward_target(model_hf, case)(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs)
            .logits.detach()
            .clone()
        )
    hf_state_dict = copy.deepcopy(model_hf.state_dict())
    del model_hf
    _release()

    # --- veomni phase ---
    # `input_ids.clone()` — the patched Omni thinker zeroes placeholder
    # positions in-place before embedding, so reuse a fresh tensor.
    model_ve = _build_veomni_model(case, hf_state_dict)
    with torch.no_grad():
        logits_ve = (
            _forward_target(model_ve, case)(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs)
            .logits.detach()
            .clone()
        )
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


# Subset of CASES exercised through the runtime converter — only the MoE
# models, since the converter only fires for them. Mirrors the eager+fp32
# and fa2+bf16 split of CASES so the converter path gets exercised against
# both the attention kernels and the dtypes that real users hit.
_RUNTIME_CONVERTER_CASES = [
    Case("qwen3_moe-toy-runtime-converter", _toy("qwen3_moe_toy"), sync_weight_key="qwen3_moe"),
    Case("deepseek_v3-toy-runtime-converter", _toy("deepseek_v3_toy"), sync_weight_key="deepseek_v3"),
    Case(
        "qwen3_moe-toy-runtime-converter-fa2",
        _toy("qwen3_moe_toy"),
        sync_weight_key="qwen3_moe",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case(
        "deepseek_v3-toy-runtime-converter-fa2",
        _toy("deepseek_v3_toy"),
        sync_weight_key="deepseek_v3",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
]


def _save_hf_checkpoint(state_dict: dict, config, dst_dir: str) -> None:
    """Write an HF-format per-expert checkpoint that build_foundation_model can read."""
    from safetensors.torch import save_file

    config.save_pretrained(dst_dir)
    save_file(
        {k: v.detach().contiguous().cpu() for k, v in state_dict.items()},
        os.path.join(dst_dir, "model.safetensors"),
    )


def _build_veomni_model_via_runtime_converter(case: Case, hf_state_dict, hf_buffers, config, weights_dir: str):
    """Build a VeOmni model by routing load through the runtime converter.

    This is the path real users hit: a per-expert HF checkpoint dir that
    the v4 stacking converter consumes at load time, no offline merge.

    Why we restore HF's non-persistent buffers after load
    -----------------------------------------------------
    The rotary `inv_freq` is registered `persistent=False`, so it isn't in
    the safetensors. On the `weights_path != None` path, the model is built
    under `init_empty_weights()`, which patches `register_parameter` but
    not `register_buffer` — so `inv_freq` is computed on CPU during
    `__init__`, snapshotted, then `.to(device)`-copied to CUDA. That
    CPU-computed-then-moved value differs from a directly-on-CUDA
    `1.0 / (base ** ...)` by one ULP, propagating into a non-zero logits
    delta (~1e-6 qwen3_moe, ~3e-3 deepseek_v3) — enough to fail
    `torch.equal`.

    This is a pre-existing loader-level quirk on path 2, not a converter
    regression: every offline-merged MoE checkpoint loaded through
    `weights_path=<merged_dir>` hits the same CPU->CUDA `inv_freq` move.
    The sibling test (`weights_path is None`) builds under
    `with torch.device(init_device)` and so dodges it.

    Restoring HF's buffers after load keeps this test a bitwise check on
    the converter's *parameter* loading. A proper loader fix
    (`init_empty_weights(include_buffers=True)` + recompute on
    `init_device` after `to_empty`) is tracked separately.
    """
    from veomni.models.auto import build_foundation_model

    _save_hf_checkpoint(hf_state_dict, config, weights_dir)

    model = build_foundation_model(
        config_path=weights_dir,
        weights_path=weights_dir,
        torch_dtype=case.dtype,
        attn_implementation=case.attn_implementation,
        init_device=get_device_type(),
    )

    # Restore non-persistent buffers (e.g. rotary inv_freq) that aren't in the
    # state dict. See the docstring above for the full background. Walking by
    # FQN keeps this independent of how nested they are.
    persistent_keys = set(hf_state_dict.keys())
    for name, buf in hf_buffers.items():
        if name in persistent_keys:
            continue
        parts = name.split(".")
        target_module = model
        for p in parts[:-1]:
            target_module = getattr(target_module, p)
        target = target_module._buffers[parts[-1]]
        target.copy_(buf.to(target.device, dtype=target.dtype))

    return model.eval()


@pytest.mark.parametrize("case", _RUNTIME_CONVERTER_CASES, ids=[c.case_id for c in _RUNTIME_CONVERTER_CASES])
def test_logits_bitwise_equal_via_runtime_converter(case: Case):
    """Smoke test: HF checkpoint -> runtime converter -> bitwise-equal forward.

    Complements `test_logits_bitwise_equal`: that one syncs HF weights via
    the manual adapter; this one writes a real per-expert HF checkpoint and
    routes loading through `MoEV4StackingConverter`, the path users hit
    with a vanilla HF model dir. Bitwise-equal logits prove the converter
    produces the exact same stacked tensors the manual adapter does.

    See `_build_veomni_model_via_runtime_converter` for why HF's
    non-persistent buffers are restored before the forward.
    """
    from transformers import AutoConfig

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
    target_dtype = _DTYPE_MAP[case.dtype]
    hf_config = AutoConfig.from_pretrained(case.path)
    input_ids, fwd_kwargs = _make_inputs(case, hf_config, device_type, target_dtype)

    # --- HF phase (must precede any veomni model build, same as the sibling test) ---
    model_hf = _build_hf_model(case)
    with torch.no_grad():
        logits_hf = (
            _forward_target(model_hf, case)(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs)
            .logits.detach()
            .clone()
        )
    hf_state_dict = copy.deepcopy(model_hf.state_dict())
    hf_buffers = {n: b.detach().clone() for n, b in model_hf.named_buffers()}
    del model_hf
    _release()

    # --- veomni phase: load the HF checkpoint through build_foundation_model ---
    tmp_dir = tempfile.mkdtemp(prefix="veomni_v4_converter_test_")
    try:
        model_ve = _build_veomni_model_via_runtime_converter(case, hf_state_dict, hf_buffers, hf_config, tmp_dir)
        with torch.no_grad():
            logits_ve = (
                _forward_target(model_ve, case)(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs)
                .logits.detach()
                .clone()
            )
        del model_ve
        _release()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        del hf_state_dict
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
            f"[{case.case_id}] logits not bitwise equal via runtime converter: "
            f"{n_mis}/{total} mismatched, max_abs_diff={max_abs:.3e}, "
            f"first_mismatch_indices={first_idx}"
        )
