"""Bitwise logits-equal tests for transformers v5 models.

Sibling to ``test_models_logits_equal.py`` (v4 scope). v5 ships self-contained
generated modeling under ``veomni/models/transformers/<model>/generated/``,
so pristine ``transformers.*`` classes stay untouched and HF + VeOmni can be
built side-by-side without an unpatch helper.

Scope decisions
---------------
- Text-only forward via ``*ForCausalLM`` so VLM-specific patches
  (``fast_pos_embed_interpolate``, ``get_image_features``, ...) don't enter
  the comparison.
- ``layer_types`` forced to all ``"full_attention"``: without the
  ``causal_conv1d`` package, HF's linear-attention path falls back to
  ``F.silu(self.conv1d(...))`` while VeOmni dispatches to fla's Triton
  ``causal_conv1d`` — different implementations, not bitwise equal.
- ``cu_seq_lens_q`` supplied as a single segment: VeOmni's patched
  ``Qwen3_5DecoderLayer.forward`` ``assert``\\s on it; HF ignores it via
  ``**kwargs``.
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


@dataclass(frozen=True)
class Case:
    """One HF-vs-VeOmni bitwise comparison.

    ``toy_config_dir`` points at a full VLM toy config; the test extracts
    ``text_config`` and runs the matching ``*ForCausalLM`` text architecture.
    """

    case_id: str
    toy_config_dir: str
    text_arch: str
    attn_implementation: str = "eager"
    dtype: str = "float32"


def _toy(name: str) -> str:
    return os.path.join(REPO_ROOT, "tests", "toy_config", name)


# eager+fp32 = RNG-init parity baseline; fa2/sdpa+bf16 = the dtype real users hit.
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
    # FA2 swapped for SDPA on MoE: the toy's (num_kv_heads=2, head_dim=256,
    # seq_len=32) shape crashes ``_flash_attn_varlen_forward`` upstream on
    # both HF and VeOmni; SDPA still exercises the bf16 path end-to-end.
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
    """1-rank NCCL group for VeOmni's SP-aware attention wrappers.

    Only init/teardown if we created the group; gated on the same skip
    conditions as the test bodies so a skip-only run doesn't pay the cost.
    """
    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if not IS_CUDA_AVAILABLE or not is_transformers_version_greater_or_equal_to("5.0.0"):
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


def _make_text_config(toy_config_dir: str, text_arch: str, layer_types_override: str = "full_attention"):
    """Extract a text-only sub-config from a v5 VLM toy config."""
    from transformers import AutoConfig

    full_config = AutoConfig.from_pretrained(toy_config_dir)
    text_config = copy.deepcopy(full_config.text_config)

    # All-full-attention override — see module docstring.
    if hasattr(text_config, "layer_types") and text_config.layer_types is not None:
        text_config.layer_types = [layer_types_override] * len(text_config.layer_types)

    # HF's default ``"grouped_mm"`` routes through ``torch._grouped_mm`` and
    # crashes on the toy's tiny expert tensors; eager runs the per-expert loop
    # on both sides. Harmless on the non-MoE qwen3_5 path.
    text_config._experts_implementation = "eager"

    # HF resolves the class via ``architectures[0]``; VeOmni's MODELING_REGISTRY
    # keys on ``model_type`` (already correct on the extracted text_config).
    text_config.architectures = [text_arch]
    return text_config


def _build_hf_text_model(text_config, attn_implementation: str, dtype: torch.dtype):
    """Pristine HF text-only model — same seed + init path as VeOmni."""
    import transformers

    cls = getattr(transformers, text_config.architectures[0])
    torch.manual_seed(0)
    get_torch_device().manual_seed_all(0)
    # ``_from_config(..., torch_dtype=...)`` inits at ``dtype`` directly;
    # ``cls(config).to(dtype)`` would init in fp32 first and produce different
    # RNG bytes (RNG output is dtype-dependent). On-device init matches
    # VeOmni's path so rotary ``inv_freq`` shares arithmetic.
    with torch.device(get_device_type()):
        model = cls._from_config(
            text_config,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
    return model.eval()


def _make_eager_ops_config(attn_implementation: str):
    """OpsImplementationConfig that pins every op to its HF-equivalent path.

    The GatedDeltaNet OpSlots (rms_norm_gated / causal_conv1d /
    chunk_gated_delta_rule) never fire under our full-attention override,
    but ``"eager"`` instead of the default ``"fla"`` keeps the test
    runnable without ``flash-linear-attention`` installed.
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
    from veomni.ops import apply_ops_config

    # Install our eager-everywhere ops config first so ``build_foundation_model``'s
    # "no config installed → use defaults" branch doesn't overwrite it (the
    # defaults rebind GatedDeltaNet slots to fla and CE to chunk_loss).
    # The SP-aware FA wrappers degrade to plain FA at sp_size=1, so passing
    # ``flash_attention_2`` directly is equivalent to the SP-rewritten name.
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
    """Text-only ids + forward kwargs. Vocab floor 32000 dodges MM placeholders."""
    gen = torch.Generator(device=device).manual_seed(0)
    vocab = max(min(text_config.vocab_size, 200000), 32000)
    seq_len = 32
    input_ids = torch.randint(32000, vocab, (1, seq_len), device=device, dtype=torch.long, generator=gen)
    cu_seq_lens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    return input_ids, {"cu_seq_lens_q": cu_seq_lens_q}


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_logits_bitwise_equal_v5(case: Case):
    """Bitwise-equal forward: pristine HF vs VeOmni patched modeling.

    transformers v5, single GPU, no GPU kernel patching. See module docstring
    for the ``layer_types`` and ``cu_seq_lens_q`` constraints.
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

    model_hf = _build_hf_text_model(text_config, case.attn_implementation, target_dtype)
    with torch.no_grad():
        logits_hf = model_hf(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs).logits.detach().clone()
    hf_state_dict = copy.deepcopy(model_hf.state_dict())
    del model_hf
    _release()

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
