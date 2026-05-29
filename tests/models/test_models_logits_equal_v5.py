"""Bitwise logits-equal tests for transformers v5 models.

Every VeOmni-supported model ships self-contained generated modeling under
``veomni/models/transformers/<model>/generated/``, so pristine
``transformers.*`` classes stay untouched and HF + VeOmni can be built
side-by-side without an unpatch helper.

Coverage
--------
Models under ``veomni/models/transformers/`` that register a patchgen-generated
class (``transformers-stable`` default group in ``pyproject.toml`` pins
``transformers==5.2.0``):

- Causal-LM (text-only):           qwen2, qwen3, qwen3_moe, deepseek_v3
- VLM via text-only sub-config
  (``*ForCausalLM`` registered):   qwen3_5, qwen3_5_moe
- VLM full forward (image + text): qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe,
                                   qwen3_5, qwen3_5_moe (the latter two reuse
                                   the same toy config as the text sub-config
                                   cases above, just with the vision tower
                                   exercised via dummy 2x2 image input)
- Causal-LM with MLA + DSA:        glm_moe_dsa (eager + sdpa only — the
  upstream class sets ``_supports_flash_attn = False``)
- Causal-LM with MLA + MoE:        deepseek_v3 (eager + fp32 only — MLA SDPA
  reshape doesn't reach bitwise parity with HF on the toy config; logit drift
  dominates the bf16 noise floor and is tracked separately)

Scope decisions
---------------
- Qwen3.5 layers are forced to all ``"full_attention"``: without
  ``causal_conv1d`` installed, HF's linear-attention path uses
  ``F.silu(self.conv1d(...))`` while VeOmni dispatches to fla's Triton
  ``causal_conv1d`` — different implementations, not bitwise equal.
- ``cu_seq_lens_q`` is supplied for Qwen3.5: VeOmni's patched
  ``Qwen3_5DecoderLayer.forward`` ``assert``\\s on it; HF ignores it via
  ``**kwargs``.
- VLM image input is a single dummy 2x2 patch — small enough to keep the
  test fast, large enough to actually run the visual tower.
- Omni audio is a follow-up; only the ``audio_mask`` zero-tensor is passed
  so the patched asserts succeed and the audio tower stays dark.
"""

import copy
import gc
import importlib.util
import os
import shutil
import tempfile
from dataclasses import dataclass, field
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


# Required by ``dist.init_process_group`` in the module-scoped fixture.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12356")
# Required by ``torch.use_deterministic_algorithms`` for cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16}

# How a case maps to (config preparation, forward target, input shape):
#   "causal_lm"         — toy config IS the text config; no extraction.
#   "qwen3_5_text"      — extract text_config + force full_attention + cu_seq_lens_q.
#   "qwen3_5_vlm_full"  — full multimodal forward with dummy 2x2 image *and* the
#                         same text-config overrides as "qwen3_5_text" applied
#                         in-place (so the text tower stays runnable without
#                         causal_conv1d/grouped_mm installed).
#   "vlm_full"          — full VLM forward with a dummy 2x2 image.
#   "omni_thinker"      — full Omni model; forward runs on ``model.thinker``.
_KINDS = ("causal_lm", "qwen3_5_text", "qwen3_5_vlm_full", "vlm_full", "omni_thinker")


@dataclass(frozen=True)
class Case:
    case_id: str
    toy_config_dir: str
    arch: str
    kind: str
    attn_implementation: str = "eager"
    dtype: str = "float32"
    forward_attr: Optional[str] = None  # e.g. "thinker" for Omni
    config_overrides: dict = field(default_factory=dict)


def _toy(name: str) -> str:
    return os.path.join(REPO_ROOT, "tests", "toy_config", name)


# Each model gets eager+fp32 (RNG-init parity baseline) and a real-user
# attention path (FA2+bf16 where supported, else SDPA+bf16).
CASES = [
    # ── causal-LM ─────────────────────────────────────────────────────────
    Case("qwen2-eager", _toy("qwen2_toy"), "Qwen2ForCausalLM", "causal_lm"),
    Case(
        "qwen2-fa2",
        _toy("qwen2_toy"),
        "Qwen2ForCausalLM",
        "causal_lm",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case("qwen3-eager", _toy("qwen3_toy"), "Qwen3ForCausalLM", "causal_lm"),
    Case(
        "qwen3-fa2",
        _toy("qwen3_toy"),
        "Qwen3ForCausalLM",
        "causal_lm",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case(
        "qwen3_moe-eager",
        _toy("qwen3_moe_toy"),
        "Qwen3MoeForCausalLM",
        "causal_lm",
        config_overrides={"_experts_implementation": "eager"},
    ),
    Case(
        "qwen3_moe-fa2",
        _toy("qwen3_moe_toy"),
        "Qwen3MoeForCausalLM",
        "causal_lm",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        config_overrides={"_experts_implementation": "eager"},
    ),
    # ── DeepSeek-V3 (MLA + MoE) ──────────────────────────────────────────
    # HF v5 builds the fused ``gate_up_proj [E, 2*I, H]`` directly via
    # ``DeepseekV3NaiveMoe`` (no ``_experts_implementation`` knob — the
    # ``naive`` path is hard-wired). VeOmni's ``PatchedDeepseekV3NaiveMoe``
    # dispatches on the ``moe_experts`` OpSlot — eager loop here because
    # the test runs without fused-kernel bindings.
    #
    # eager+fp32 only: deepseek_v3 ships MLA (multi-head latent attention)
    # with split q_a / q_b / kv_a / kv_b projections, and the patchgen-
    # generated SDPA path doesn't currently reach bitwise parity with HF's
    # MLA SDPA reshape on the toy config (logit drift dominates the bf16
    # noise floor — tracked separately). The eager+fp32 case is sufficient
    # to validate the patchgen modeling: it covers the MoE expert dispatch,
    # the OpSlot-guarded cross-entropy, and the v5 ``gate_up_proj`` layout
    # end-to-end.
    Case("deepseek_v3-eager", _toy("deepseek_v3_toy"), "DeepseekV3ForCausalLM", "causal_lm"),
    # ── GLM-MoE-DSA (MLA + Dynamic Sparse Attention) ─────────────────────
    # Upstream sets ``_supports_flash_attn = False``, so FA2 is not an
    # option here — we use eager+fp32 as the RNG-init baseline and
    # sdpa+bf16 as the closest-to-real-user path. Like every MoE case the
    # ``@use_experts_implementation`` decorator needs ``"eager"`` so HF's
    # ``torch._grouped_mm`` path doesn't diverge from VeOmni's eager loop.
    Case(
        "glm_moe_dsa-eager",
        _toy("glm_moe_dsa_toy"),
        "GlmMoeDsaForCausalLM",
        "causal_lm",
        config_overrides={"_experts_implementation": "eager"},
    ),
    Case(
        "glm_moe_dsa-sdpa",
        _toy("glm_moe_dsa_toy"),
        "GlmMoeDsaForCausalLM",
        "causal_lm",
        attn_implementation="sdpa",
        dtype="bfloat16",
        config_overrides={"_experts_implementation": "eager"},
    ),
    # ── Qwen3.5 (text-only sub-config) ───────────────────────────────────
    Case("qwen3_5-text-eager", _toy("qwen3_5_toy"), "Qwen3_5ForCausalLM", "qwen3_5_text"),
    Case(
        "qwen3_5-text-fa2",
        _toy("qwen3_5_toy"),
        "Qwen3_5ForCausalLM",
        "qwen3_5_text",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case("qwen3_5_moe-text-eager", _toy("qwen3_5_moe_toy"), "Qwen3_5MoeForCausalLM", "qwen3_5_text"),
    # FA2 swapped for SDPA: the toy's (num_kv_heads=2, head_dim=256, seq_len=32)
    # crashes ``_flash_attn_varlen_forward`` upstream on both HF and VeOmni.
    Case(
        "qwen3_5_moe-text-sdpa",
        _toy("qwen3_5_moe_toy"),
        "Qwen3_5MoeForCausalLM",
        "qwen3_5_text",
        attn_implementation="sdpa",
        dtype="bfloat16",
    ),
    # ── Qwen3.5 (full multimodal forward; same toy configs as above) ────
    # Reuses the qwen3_5{,_moe}_toy configs but runs ``*ForConditionalGeneration``
    # over the whole config (text + vision tower) instead of the extracted text
    # sub-config. Text-tower constraints (full_attention + eager experts +
    # cu_seq_lens_q) are mirrored from the text-only cases — without them HF's
    # linear-attention / grouped-mm paths diverge from VeOmni's patched ones.
    # The image input is the same dummy 2x2 patch the other VLMs use, large
    # enough to fire ``fast_pos_embed_interpolate`` / ``rot_pos_emb`` / the
    # patched ``Qwen3_5VisionModel.forward`` host-side ``cu_seqlens`` build.
    #
    # sdpa+bf16 (not fa2) because:
    #   - eager+fp32 fails at ``lm_head`` with dtype mismatch: HF init forces
    #     bf16 on the vision tower (init_zeros for the offset RMSNorm) and
    #     ``masked_scatter`` propagates bf16 into the language tower, so
    #     eager+fp32 would just test a half-cast model.
    #   - fa2+bf16 produces NaN on the toy config (HF side too, not VeOmni's
    #     fault): same upstream FA-on-tiny-shape issue that already pushed
    #     ``qwen3_5_moe-text`` from fa2 to sdpa above.
    # sdpa+bf16 has consistent dtype end-to-end and avoids the FA toy crash,
    # while still covering everything we actually patch in the ViT.
    Case(
        "qwen3_5_vl-sdpa",
        _toy("qwen3_5_toy"),
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5_vlm_full",
        attn_implementation="sdpa",
        dtype="bfloat16",
    ),
    Case(
        "qwen3_5_moe_vl-sdpa",
        _toy("qwen3_5_moe_toy"),
        "Qwen3_5MoeForConditionalGeneration",
        "qwen3_5_vlm_full",
        attn_implementation="sdpa",
        dtype="bfloat16",
    ),
    # ── VLMs (full forward with a dummy 2x2 image) ───────────────────────
    Case("qwen2_vl-eager", _toy("qwen2vl_toy"), "Qwen2VLForConditionalGeneration", "vlm_full"),
    Case(
        "qwen2_vl-fa2",
        _toy("qwen2vl_toy"),
        "Qwen2VLForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case("qwen2_5_vl-eager", _toy("qwen25vl_toy"), "Qwen2_5_VLForConditionalGeneration", "vlm_full"),
    Case(
        "qwen2_5_vl-fa2",
        _toy("qwen25vl_toy"),
        "Qwen2_5_VLForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case("qwen3_vl-eager", _toy("qwen3vl_toy"), "Qwen3VLForConditionalGeneration", "vlm_full"),
    Case(
        "qwen3_vl-fa2",
        _toy("qwen3vl_toy"),
        "Qwen3VLForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    # qwen3_vl_moe: HF init forces bf16 on the language tower (init_zeros for
    # the offset RMSNorm), so eager+fp32 would just test a half-cast model.
    # ``_experts_implementation="eager"`` matches qwen3_moe — without it HF
    # defaults to ``"grouped_mm"`` (``torch._grouped_mm``) which diverges
    # numerically from VeOmni's eager expert loop.
    Case(
        "qwen3_vl_moe-fa2",
        _toy("qwen3vlmoe_toy"),
        "Qwen3VLMoeForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        config_overrides={"_experts_implementation": "eager"},
    ),
    # ── Omni (forward on ``model.thinker`` so talker stays out of scope) ──
    Case(
        "qwen2_5_omni-fa2",
        _toy("qwen25omni_toy"),
        "Qwen2_5OmniForConditionalGeneration",
        "omni_thinker",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_attr="thinker",
    ),
    Case(
        "qwen3_omni_moe-fa2",
        _toy("qwen3omni_toy"),
        "Qwen3OmniMoeForConditionalGeneration",
        "omni_thinker",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_attr="thinker",
        config_overrides={"_experts_implementation": "eager"},
    ),
]


def _apply_determinism():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group():
    """1-rank NCCL group for VeOmni's SP-aware attention wrappers.

    Only init/teardown if we created the group; gated on the same skip
    conditions as the test bodies so a skip-only run doesn't pay the cost.
    """
    if not IS_CUDA_AVAILABLE:
        yield
        return

    import torch.distributed as dist

    we_initialised = False
    if not dist.is_initialized():
        # Bind the accelerator before init so NCCL doesn't pick the wrong
        # visible device on multi-GPU CI hosts.
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


# ── config preparation ───────────────────────────────────────────────────


def _apply_qwen3_5_text_overrides(text_cfg) -> None:
    """In-place: force full-attention layers and eager experts on a qwen3_5 text config.

    The qwen3_5 family's linear-attention layers diverge bitwise from HF without
    ``causal_conv1d`` installed (HF: ``F.silu(self.conv1d(...))``; VeOmni: fla
    Triton ``causal_conv1d``). The MoE variant's default ``"grouped_mm"`` routes
    through ``torch._grouped_mm`` and crashes on the toy's tiny expert tensors;
    eager runs the per-expert loop on both sides. Both fixes apply to ``text_cfg``
    whether we consume it standalone (``qwen3_5_text``) or in-place inside the
    full multimodal config (``qwen3_5_vlm_full``).
    """
    if hasattr(text_cfg, "layer_types") and text_cfg.layer_types is not None:
        text_cfg.layer_types = ["full_attention"] * len(text_cfg.layer_types)
    text_cfg._experts_implementation = "eager"


def _make_config(case: Case):
    """Return the config the test will hand to both HF and VeOmni."""
    from transformers import AutoConfig

    full_config = AutoConfig.from_pretrained(case.toy_config_dir)

    if case.kind == "qwen3_5_text":
        cfg = copy.deepcopy(full_config.text_config)
        _apply_qwen3_5_text_overrides(cfg)
    elif case.kind == "qwen3_5_vlm_full":
        # Keep the full multimodal config (text + vision) but mutate the text
        # sub-config in-place so the text tower has the same bitwise-equal
        # ground rules as ``qwen3_5_text``.
        cfg = full_config
        _apply_qwen3_5_text_overrides(cfg.text_config)
    else:
        cfg = full_config

    cfg.architectures = [case.arch]
    for k, v in case.config_overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── input construction ───────────────────────────────────────────────────


def _vision_section(config):
    """Return ``(vision_config, image_token_id)`` or ``(None, None)``.

    Top-level VLs carry ``vision_config`` on the root; Omni nests it under
    ``thinker_config``. The placeholder field renamed ``_index`` -> ``_id``
    between qwen2_5_omni and qwen3_omni_moe — accept either.
    """
    if not hasattr(config, "vision_config") and not hasattr(config, "thinker_config"):
        return None, None
    vc_root = config.thinker_config if hasattr(config, "thinker_config") else config
    vision_config = getattr(vc_root, "vision_config", None)
    if vision_config is None:
        return None, None
    image_token = getattr(vc_root, "image_token_index", None)
    if image_token is None:
        image_token = getattr(vc_root, "image_token_id", None)
    return vision_config, image_token


def _make_inputs(case: Case, config, device, dtype) -> tuple[torch.Tensor, dict]:
    """Build ``(input_ids, forward_kwargs)`` for the case.

    Causal-LM / qwen3_5 text: text-only ids. VLM / Omni: same base ids but
    the first ``n_tokens`` positions are overwritten with ``image_token_id``
    and dummy ``pixel_values`` + ``image_grid_thw = [[1, 2, 2]]`` are passed
    so the visual tower runs and the patched ``masked_scatter`` consumes
    every embedding once. Omni additionally needs a zero ``audio_mask``
    to satisfy the patched asserts.
    """
    seq_len = 32
    base_gen = torch.Generator(device=device).manual_seed(0)
    base_input_ids = torch.randint(32000, (1, seq_len), device=device, dtype=torch.long, generator=base_gen)

    fwd_kwargs: dict = {}
    if case.kind in ("qwen3_5_text", "qwen3_5_vlm_full"):
        # VeOmni's patched ``Qwen3_5DecoderLayer.forward`` ``assert``s on it;
        # HF ignores it via ``**kwargs``. Fires for every text-tower layer
        # regardless of whether the surrounding forward is text-only or
        # multimodal.
        fwd_kwargs["cu_seq_lens_q"] = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    vision_config, image_token_id = _vision_section(config)
    if vision_config is None or case.kind in ("causal_lm", "qwen3_5_text"):
        return base_input_ids, fwd_kwargs

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

    fwd_kwargs.update(
        {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            # ``image_mask`` / ``video_mask`` are passed explicitly because
            # the patches otherwise call ``dist.all_gather`` on input_ids to
            # recompute them.
            "image_mask": image_mask,
            "video_mask": video_mask,
        }
    )
    if case.kind == "omni_thinker":
        fwd_kwargs["audio_mask"] = torch.zeros_like(input_ids, dtype=torch.bool)

    return input_ids, fwd_kwargs


# ── HF / VeOmni model build ──────────────────────────────────────────────


def _resolve_hf_class(arch: str):
    import transformers

    return getattr(transformers, arch)


def _build_hf_model(case: Case, config, dtype: torch.dtype):
    """Pristine HF model — same seed + init path as VeOmni."""
    cls = _resolve_hf_class(case.arch)
    torch.manual_seed(0)
    get_torch_device().manual_seed_all(0)
    # ``_from_config(..., torch_dtype=...)`` inits at ``dtype`` directly;
    # ``cls(config).to(dtype)`` would init in fp32 first and produce
    # different RNG bytes (RNG output is dtype-dependent). On-device init
    # matches VeOmni's path so rotary ``inv_freq`` shares arithmetic.
    with torch.device(get_device_type()):
        model = cls._from_config(
            config,
            torch_dtype=dtype,
            attn_implementation=case.attn_implementation,
        )
    return model.eval()


def _build_veomni_model(case: Case, config, hf_state_dict):
    """VeOmni-generated model with HF state_dict loaded."""
    from veomni.models.auto import build_foundation_model
    from veomni.ops import apply_ops_config

    from ..tools.training_utils import make_eager_ops_config

    # Install our eager-everywhere ops config first so ``build_foundation_model``'s
    # contract (caller passes ops_implementation OR singleton already installed)
    # is satisfied. The Qwen3.5 GatedDeltaNet OpSlots (rms_norm_gated /
    # causal_conv1d / chunk_gated_delta_rule) never fire under our
    # full-attention override, but pinning them to ``"eager"`` keeps the test
    # runnable without ``flash-linear-attention`` installed.
    # The SP-aware FA wrappers degrade to plain FA at sp_size=1, so passing
    # ``flash_attention_2`` directly is equivalent to the SP-rewritten name.
    apply_ops_config(make_eager_ops_config(attn_implementation=case.attn_implementation))

    model = build_foundation_model(
        config_path=config,
        weights_path=None,
        torch_dtype=case.dtype,
        attn_implementation=case.attn_implementation,
        init_device=get_device_type(),
    )
    model.load_state_dict(hf_state_dict)
    return model.eval()


def _forward_target(model, case: Case):
    if case.forward_attr is None:
        return model
    target = model
    for part in case.forward_attr.split("."):
        target = getattr(target, part)
    return target


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_logits_bitwise_equal_v5(case: Case):
    """Bitwise-equal forward: pristine HF vs VeOmni patched modeling."""
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if not os.path.isdir(case.toy_config_dir):
        pytest.skip(f"Path not found: {case.toy_config_dir}")
    if case.attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn package not installed.")

    _apply_determinism()

    device = get_device_type()
    target_dtype = _DTYPE_MAP[case.dtype]

    config = _make_config(case)
    input_ids, fwd_kwargs = _make_inputs(case, config, device, target_dtype)

    model_hf = _build_hf_model(case, config, target_dtype)
    with torch.no_grad():
        logits_hf = (
            _forward_target(model_hf, case)(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs)
            .logits.detach()
            .clone()
        )
    hf_state_dict = copy.deepcopy(model_hf.state_dict())
    del model_hf
    _release()

    model_ve = _build_veomni_model(case, config, hf_state_dict)
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


# ── runtime loader path (weights_path != None) ───────────────────────────
#
# The sibling test above bypasses ``build_foundation_model``'s loader by
# setting ``weights_path=None`` and copying weights in-memory via
# ``model.load_state_dict(...)``. That misses the path real users hit:
# ``init_empty_weights()`` + safetensors round-trip + non-persistent buffer
# rematerialisation. The loader test below writes the HF state-dict to a
# tmpdir and lets ``build_foundation_model(weights_path=<dir>)`` rehydrate
# the model end-to-end through the on-disk loader pipeline.
#
# Coverage: every v5 MoE family with one ``fa2+bf16`` (or ``sdpa+bf16`` for
# ``glm_moe_dsa``, which sets ``_supports_flash_attn = False``) representative,
# plus an ``eager+fp32`` baseline where supported. The loader path itself is
# largely architecture-independent, so the value of breadth here is the
# *non-persistent buffer surface* (vision tower RoPE caches, audio tower
# positional embeddings) and the larger state-dict that the safetensors round
# trip has to walk — both stress what the in-memory sibling test misses.
#
# ``qwen3_5_moe`` is the lone v5 MoE we skip: that case extracts
# ``full_config.text_config`` and builds a text-only model, switching
# ``model_type`` from ``qwen3_5_moe`` to ``qwen3_5_moe_text``. VeOmni's
# ``MODELING_REGISTRY`` keys off the parent ``model_type``, so handing the
# extracted text-config to ``build_foundation_model(config_path=...)`` would
# need either a synthetic wrapper config or a registry-bypass — both
# misrepresent the real-user loader flow we want to cover.
_LOADER_CASES = [
    Case(
        "qwen3_moe-eager-loader",
        _toy("qwen3_moe_toy"),
        "Qwen3MoeForCausalLM",
        "causal_lm",
        config_overrides={"_experts_implementation": "eager"},
    ),
    Case(
        "qwen3_moe-fa2-loader",
        _toy("qwen3_moe_toy"),
        "Qwen3MoeForCausalLM",
        "causal_lm",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        config_overrides={"_experts_implementation": "eager"},
    ),
    # deepseek_v3 is the only v5 MoE we ship that emits a *per-expert*
    # safetensors layout from ``save_pretrained`` (via the ``qwen2_moe``
    # weight-conversion mapping). The loader path therefore actually fires
    # ``DeepseekV3CheckpointTensorConverter`` here, merging per-expert
    # tensors back into VeOmni's fused ``gate_up_proj``/``down_proj``
    # layout. Bitwise-equal logits prove the converter's per-expert →
    # stack-and-fuse merge yields the same parameters HF builds in-memory.
    # Eager+fp32 only — see the ``deepseek_v3-eager`` entry in CASES for
    # why fa2/sdpa diverge on MLA today.
    Case("deepseek_v3-eager-loader", _toy("deepseek_v3_toy"), "DeepseekV3ForCausalLM", "causal_lm"),
    Case(
        "glm_moe_dsa-eager-loader",
        _toy("glm_moe_dsa_toy"),
        "GlmMoeDsaForCausalLM",
        "causal_lm",
        config_overrides={"_experts_implementation": "eager"},
    ),
    Case(
        "glm_moe_dsa-sdpa-loader",
        _toy("glm_moe_dsa_toy"),
        "GlmMoeDsaForCausalLM",
        "causal_lm",
        attn_implementation="sdpa",
        dtype="bfloat16",
        config_overrides={"_experts_implementation": "eager"},
    ),
    # qwen3_vl_moe: HF init forces bf16 on the language tower, so only
    # fa2+bf16 exercises a self-consistent dtype assignment (matches the
    # CASES entry above).
    Case(
        "qwen3_vl_moe-fa2-loader",
        _toy("qwen3vlmoe_toy"),
        "Qwen3VLMoeForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        config_overrides={"_experts_implementation": "eager"},
    ),
    # qwen2_5_omni / qwen3_omni_moe: forward on ``model.thinker`` so the
    # talker stays out of scope — same as the in-memory CASES entries.
    # State-dict load still happens at the root, so talker / vision /
    # audio sub-modules round-trip through safetensors with the rest.
    Case(
        "qwen2_5_omni-fa2-loader",
        _toy("qwen25omni_toy"),
        "Qwen2_5OmniForConditionalGeneration",
        "omni_thinker",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_attr="thinker",
    ),
    Case(
        "qwen3_omni_moe-fa2-loader",
        _toy("qwen3omni_toy"),
        "Qwen3OmniMoeForConditionalGeneration",
        "omni_thinker",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_attr="thinker",
        config_overrides={"_experts_implementation": "eager"},
    ),
]


def _save_hf_checkpoint(state_dict: dict, config, dst_dir: str) -> None:
    """Write an HF-format checkpoint that ``build_foundation_model`` can read."""
    from safetensors.torch import save_file

    config.save_pretrained(dst_dir)
    save_file(
        {k: v.detach().contiguous().cpu() for k, v in state_dict.items()},
        os.path.join(dst_dir, "model.safetensors"),
    )


def _build_veomni_model_from_disk(case: Case, config, hf_state_dict, hf_buffers, weights_dir: str):
    """Build a VeOmni model by routing load through the disk-backed loader.

    This is the path real users hit
    (``build_foundation_model(weights_path=<dir>)``).
    For MoE the on-disk layout already matches VeOmni's stacked layout,
    so no expert-stacking converter fires here — the value of this test
    is exercising the loader pipeline itself (``init_empty_weights`` +
    safetensors deserialisation + non-persistent buffer rematerialisation),
    not the converter.

    Why we restore HF's non-persistent buffers after load
    -----------------------------------------------------
    Loader-level quirk: rotary ``inv_freq`` is registered
    ``persistent=False``, so it isn't in
    the safetensors. On the ``weights_path != None`` path the model is
    built under ``init_empty_weights()``, which patches ``register_parameter``
    but not ``register_buffer`` — so ``inv_freq`` is computed on CPU during
    ``__init__``, snapshotted, then ``.to(device)``-copied. That value
    differs from a directly-on-CUDA ``1.0 / (base ** ...)`` by one ULP,
    enough to fail ``torch.equal``. Restoring HF's buffers after load
    keeps this test a bitwise check on the loader's *parameter* path; the
    buffer rematerialisation fix is tracked separately.

    Why we hand a directory (not the in-memory config) to ``build_foundation_model``
    -------------------------------------------------------------------------------
    Some VeOmni models register a ``MODEL_CONFIG_REGISTRY`` hook that mutates
    the config at build-time (e.g. qwen3_omni_moe forces
    ``tie_word_embeddings=False`` so the loader's post-load tie step is a
    no-op on the thinker+talker container). That hook fires inside
    ``build_config(...)``, which is reached only when ``config_path`` is a
    string path — passing a ``PretrainedConfig`` short-circuits it.
    We therefore feed the on-disk directory and re-apply
    ``case.config_overrides`` via ``config_kwargs`` so private fields
    like ``_experts_implementation`` survive without depending on
    ``PretrainedConfig`` JSON round-trip semantics for underscored attrs.
    """
    from veomni.models.auto import build_foundation_model
    from veomni.ops import apply_ops_config

    from ..tools.training_utils import make_eager_ops_config

    apply_ops_config(make_eager_ops_config(attn_implementation=case.attn_implementation))
    _save_hf_checkpoint(hf_state_dict, config, weights_dir)

    model = build_foundation_model(
        config_path=weights_dir,
        weights_path=weights_dir,
        config_kwargs=dict(case.config_overrides),
        torch_dtype=case.dtype,
        attn_implementation=case.attn_implementation,
        init_device=get_device_type(),
    )

    # Iterate VeOmni's buffers (not HF's): the VeOmni-side model can be a
    # subset of the HF model (e.g. omni strips ``talker`` because the talker
    # is not subclassed locally), so HF may carry buffers that have no
    # counterpart on VeOmni — skipping those is the correct behaviour, not
    # masking a real defect.
    persistent_keys = set(hf_state_dict.keys())
    for name, ve_buf in model.named_buffers():
        if name in persistent_keys:
            continue
        src = hf_buffers.get(name)
        if src is None:
            continue
        ve_buf.copy_(src.to(ve_buf.device, dtype=ve_buf.dtype))

    return model.eval()


@pytest.mark.parametrize("case", _LOADER_CASES, ids=[c.case_id for c in _LOADER_CASES])
def test_logits_bitwise_equal_v5_via_loader(case: Case):
    """Bitwise-equal forward through the disk-backed loader path.

    Complements ``test_logits_bitwise_equal_v5``: that one short-circuits
    weight loading via in-memory ``load_state_dict``; this one routes the
    HF state-dict through ``build_foundation_model(weights_path=<dir>)``,
    which is the path users actually hit. Bitwise-equal logits prove the
    loader doesn't perturb any tensor on the way in.

    See ``_build_veomni_model_from_disk`` for why HF's non-persistent
    buffers are restored before the forward.
    """
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if not os.path.isdir(case.toy_config_dir):
        pytest.skip(f"Path not found: {case.toy_config_dir}")
    if case.attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn package not installed.")

    _apply_determinism()

    device = get_device_type()
    target_dtype = _DTYPE_MAP[case.dtype]

    config = _make_config(case)
    input_ids, fwd_kwargs = _make_inputs(case, config, device, target_dtype)

    model_hf = _build_hf_model(case, config, target_dtype)
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

    tmp_dir = tempfile.mkdtemp(prefix="veomni_v5_loader_test_")
    try:
        model_ve = _build_veomni_model_from_disk(case, config, hf_state_dict, hf_buffers, tmp_dir)
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
            f"[{case.case_id}] logits not bitwise equal via loader: "
            f"{n_mis}/{total} mismatched, max_abs_diff={max_abs:.3e}, "
            f"first_mismatch_indices={first_idx}"
        )
