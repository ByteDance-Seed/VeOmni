"""Bitwise logits-equal tests for transformers v5 models.

Sibling to ``test_models_logits_equal.py`` (which targets transformers v4 +
runtime monkey patches). v5 ships generated, self-contained patched modeling
files under ``veomni/models/transformers/<model>/generated/``; the pristine
HF classes in ``transformers`` are therefore never mutated, so this test
needs no ``hf_unpatch.py`` analogue — both ``transformers.Qwen3_5ForCausalLM``
and the VeOmni ``patched_modeling_qwen3_5_gpu.Qwen3_5ForCausalLM`` are built
side-by-side.

Scope decisions
---------------
- Text-only forward via ``Qwen3_5ForCausalLM`` / ``Qwen3_5MoeForCausalLM``.
  The toy configs are full VLMs; we extract ``text_config`` and use the
  text-only architecture so the comparison is not perturbed by VLM-specific
  patches (``fast_pos_embed_interpolate``, ``get_image_features``, etc.).
- ``layer_types`` is overridden to all ``"full_attention"``. Without the
  ``causal_conv1d`` PyPI package, HF's linear-attention path falls back to
  ``F.silu(self.conv1d(...))`` while VeOmni's OpSlot dispatches to
  ``fla.modules.convolution.causal_conv1d`` — different kernels, not bitwise
  equal. Restricting to full-attention layers exercises the rest of the
  modeling stack (RMSNorm + RoPE + GQA attention + MoE experts + MTP head)
  under the same OpSlot-bound kernels on both sides.
- ``cu_seq_lens_q`` is supplied as a single-segment cumulative tensor.
  VeOmni's patched ``Qwen3_5DecoderLayer.forward`` ``assert``\\s on it; HF's
  decoder layer ignores it (consumed via ``**kwargs``).
- Both ``qwen3_5`` and ``qwen3_5_moe`` patched modeling files gate RMSNorm
  behind an ``OpSlot("rms_norm", "qwen3_5")`` guard; with
  ``rms_norm_implementation="eager"`` the slot is unbound and the forward
  falls through to the HF reference, so no module-level monkey-patching is
  needed on either side.
"""

import copy
import gc
import importlib.util
import os
from dataclasses import dataclass

import pytest
import torch

from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    empty_cache,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


# Must be set before ``import veomni`` so OpSlot defaults that map to "eager"
# stay as eager and don't try to bind GPU-only kernels at import time.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12356")
# Required by torch.use_deterministic_algorithms for cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class Case:
    """One bitwise-equal comparison between pristine HF and VeOmni-generated.

    ``toy_config_dir`` points at a v5 VLM toy config (Qwen3_5Config or
    Qwen3_5MoeConfig); the test extracts ``text_config`` and runs forward
    against the matching ``*ForCausalLM`` text-only architecture.
    """

    case_id: str
    toy_config_dir: str
    text_arch: str  # e.g. "Qwen3_5ForCausalLM"
    attn_implementation: str = "eager"
    dtype: str = "float32"


def _toy(name: str) -> str:
    return os.path.join(REPO_ROOT, "tests", "toy_config", name)


# Eager+fp32 covers the determinism / RNG-init parity baseline. fa2+bf16
# covers the dtype path that real users hit (Qwen3.5 hard-codes bf16 in the
# toy config). Two attn implementations (eager / FA2) catch divergence inside
# the attention forward itself.
CASES = [
    Case(
        "qwen3_5-toy-text-eager",
        _toy("qwen3_5_toy"),
        text_arch="Qwen3_5ForCausalLM",
    ),
    Case(
        "qwen3_5-toy-text-fa2",
        _toy("qwen3_5_toy"),
        text_arch="Qwen3_5ForCausalLM",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    Case(
        "qwen3_5_moe-toy-text-eager",
        _toy("qwen3_5_moe_toy"),
        text_arch="Qwen3_5MoeForCausalLM",
    ),
    # Qwen3.5-MoE drops FA2: the toy config (num_kv_heads=2, head_dim=256,
    # seq_len=32) crashes inside ``flash_attn.flash_attn_interface._flash_attn_varlen_forward``
    # with an illegal memory access on both pristine HF and VeOmni. Replaced
    # with SDPA so the bf16 path still gets exercised end-to-end.
    Case(
        "qwen3_5_moe-toy-text-sdpa",
        _toy("qwen3_5_moe_toy"),
        text_arch="Qwen3_5MoeForCausalLM",
        attn_implementation="sdpa",
        dtype="bfloat16",
    ),
]


def _apply_determinism():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group():
    """Module-scoped 1-rank process group for the SP-aware attention wrappers.

    VeOmni's generated modeling reads ``get_parallel_state()`` and the
    SP-aware FA wrappers expect a default group even at sp_size=1. Mirrors
    the v4 sibling test's gates so a skip-only environment doesn't pay an
    init cost.
    """
    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if not IS_CUDA_AVAILABLE or not is_transformers_version_greater_or_equal_to("5.0.0"):
        yield
        return

    import torch.distributed as dist

    we_initialised = False
    if not dist.is_initialized():
        # Bind the accelerator before init_process_group so multi-GPU CI
        # hosts don't have NCCL pick the wrong visible device.
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


def _make_text_config(toy_config_dir: str, text_arch: str, layer_types_override: str = "full_attention"):
    """Load the toy VLM config and return a text-only config ready for build.

    Why ``layer_types`` is overridden — see module docstring.
    Why ``_experts_implementation = "eager"`` is forced — HF's default
    ``"grouped_mm"`` (resolved by ``get_correct_experts_implementation``)
    routes ``Qwen3_5MoeExperts.forward`` through ``torch._grouped_mm`` /
    ``torch.nn.functional.grouped_mm``, which is incompatible with the toy
    config's tiny expert tensors and crashes with an illegal memory access.
    Eager runs the original per-expert loop on both sides.
    """
    from transformers import AutoConfig

    full_config = AutoConfig.from_pretrained(toy_config_dir)
    text_config = copy.deepcopy(full_config.text_config)

    # Strip linear-attention layers — see module docstring for the rationale.
    if hasattr(text_config, "layer_types") and text_config.layer_types is not None:
        text_config.layer_types = [layer_types_override] * len(text_config.layer_types)

    # Force eager MoE experts forward on the HF side; harmless on VeOmni's side
    # (the class_replacement removes the dispatching decorator entirely).
    text_config._experts_implementation = "eager"

    # The HF text-only architecture lookup happens via ``architectures[0]``
    # on both sides; pristine HF goes through ``getattr(transformers, ...)``
    # and VeOmni's ``MODELING_REGISTRY`` keys on ``model_type``. The toy's
    # ``text_config.model_type`` already matches the v5 text-model registry
    # entry (``qwen3_5_text`` / ``qwen3_5_moe_text``).
    text_config.architectures = [text_arch]
    return text_config


def _build_hf_text_model(text_config, attn_implementation: str, dtype: torch.dtype):
    """Pristine HF text-only model — same RNG seed as VeOmni so weights match."""
    import transformers

    cls = getattr(transformers, text_config.architectures[0])
    torch.manual_seed(0)
    get_torch_device().manual_seed_all(0)
    # ``_from_config`` inits weights directly at ``dtype``; ``cls(config).to(bf16)``
    # would init in fp32 first and produce different random bytes than a
    # bf16-from-the-start build (RNG output depends on tensor dtype). Allocating
    # on-device matches VeOmni's path so rotary ``inv_freq`` / friends share
    # arithmetic.
    with torch.device(get_device_type()):
        model = cls._from_config(
            text_config,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
    return model.eval()


def _make_eager_ops_config(attn_implementation: str):
    """OpsImplementationConfig that pins every op to its HF-equivalent path.

    With ``layer_types`` forced to ``full_attention`` the GatedDeltaNet
    OpSlots (rms_norm_gated / causal_conv1d / chunk_gated_delta_rule) never
    fire, but binding them to ``"eager"`` rather than the default ``"fla"``
    avoids loading optional kernels and keeps the test runnable on hosts
    where ``flash-linear-attention`` is missing.
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig

    return OpsImplementationConfig(
        attn_implementation=attn_implementation,
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
    )


def _build_veomni_text_model(text_config, attn_implementation: str, dtype_name: str, hf_state_dict):
    """VeOmni-generated text-only model with HF state_dict loaded."""
    from veomni.models.auto import build_foundation_model

    # ``apply_ops_config`` is normally called *inside* ``build_foundation_model``
    # when ``ops_implementation`` is supplied. We pre-call it so the config's
    # ``__post_init__`` SP-rewriting (under ``MODELING_BACKEND=veomni``) runs
    # before we hand a still-original ``attn_implementation`` to
    # ``build_foundation_model``. The SP wrappers degrade to the underlying FA
    # kernel at ``ulysses_enabled=False`` (sp_size=1), so passing
    # ``flash_attention_2`` directly here is equivalent to the rewritten name.
    from veomni.ops import apply_ops_config

    apply_ops_config(_make_eager_ops_config(attn_implementation))

    model = build_foundation_model(
        config_path=text_config,
        weights_path=None,
        torch_dtype=dtype_name,
        attn_implementation=attn_implementation,
        init_device=get_device_type(),
    )
    model.load_state_dict(hf_state_dict)
    return model.eval()


def _make_input_ids_and_kwargs(text_config, device) -> tuple[torch.Tensor, dict]:
    """Text-only input_ids and forward kwargs.

    Vocab floor 32000 dodges multimodal placeholder ids. ``cu_seq_lens_q``
    is required by VeOmni's patched ``Qwen3_5DecoderLayer.forward``; HF's
    decoder ignores it via ``**kwargs``.
    """
    gen = torch.Generator(device=device).manual_seed(0)
    vocab = max(min(text_config.vocab_size, 200000), 32000)
    seq_len = 32
    input_ids = torch.randint(32000, vocab, (1, seq_len), device=device, dtype=torch.long, generator=gen)
    cu_seq_lens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    return input_ids, {"cu_seq_lens_q": cu_seq_lens_q}


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_logits_bitwise_equal_v5(case: Case):
    """Bitwise-equal forward: pristine HF vs VeOmni patched modeling.

    Scope: transformers v5 (>=5.0.0). Single GPU. No GPU kernel patching.
    ``layer_types`` forced to ``full_attention`` to avoid the
    ``causal_conv1d`` kernel divergence between HF (Dao-AILab / pure-PyTorch
    fallback) and VeOmni (fla Triton). See module docstring.
    """
    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if not is_transformers_version_greater_or_equal_to("5.0.0"):
        pytest.skip("Scope is transformers v5 model definition only.")
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if not os.path.isdir(case.toy_config_dir):
        pytest.skip(f"Path not found: {case.toy_config_dir}")
    if case.attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn package not installed.")

    _apply_determinism()

    device = get_device_type()
    target_dtype = _DTYPE_MAP[case.dtype]

    text_config = _make_text_config(case.toy_config_dir, case.text_arch)
    input_ids, fwd_kwargs = _make_input_ids_and_kwargs(text_config, device)

    # --- HF phase ---
    # No "must precede VeOmni build" ordering constraint here (unlike the v4
    # test): the v5 generated modeling files don't mutate the
    # ``transformers.*`` namespace, so HF stays pristine regardless of order.
    model_hf = _build_hf_text_model(text_config, case.attn_implementation, target_dtype)
    with torch.no_grad():
        logits_hf = model_hf(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs).logits.detach().clone()
    hf_state_dict = copy.deepcopy(model_hf.state_dict())
    del model_hf
    _release()

    # --- VeOmni phase ---
    model_ve = _build_veomni_text_model(text_config, case.attn_implementation, case.dtype, hf_state_dict)
    with torch.no_grad():
        logits_ve = model_ve(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs).logits.detach().clone()
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
