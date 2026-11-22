"""Janus SeedOmni training-graph parity under FSDP2, with and without SP.

``transformers`` has no ``JanusForConditionalGeneration`` reference to compare
against — and upstream Janus is a different codebase, so a monolithic comparison
would test the port rather than the wiring. What is worth pinning is the one
thing SeedOmni itself adds: **the same graph must produce the same step whether
Ulysses SP is on or off.** A wrong slice/gather does not change the shapes, so
the loss alone is not enough — this compares loss, every parameter gradient, and
the AdamW update that follows.

The backbone is the only Janus module that slices: ``janus_llama`` packs all
carrier embeddings into one flat sequence and splits it across the SP group.
The other three are either replicated per rank (``janus_siglip``, DDP) or
shard over a different axis (``janus_text_encoder``, vocab-parallel ``emb``), so
they are the same on every SP rank by construction. The packed length is odd, so
the SP=2 slice has to pad and the output gather has to unpad.

Run on CPU/Gloo in FP32, and on real NCCL/HCCL collectives in BF16.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.training_utils import make_eager_ops_config
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY
from veomni.models.seed_omni.accelerated.utils.executor import TrainNodeRunner
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.janus.text_encoder.chat_template import JanusChatTemplate
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


_CI_CPU_2 = LaunchSpec(nproc=2, device="cpu")
_CI_ACCELERATOR_2 = LaunchSpec(nproc=2, device="accelerator")

MODULES = ("janus_llama", "janus_text_encoder")

# Backbone + vocabulary half only: the graph under test is
# ``text_encoder.encode -> llama -> text_encoder.decode``, the path SP slices.
# ``janus_siglip`` / ``janus_vqvae`` are covered by ``test_janus_modules.py``;
# including them here would need image fixtures without adding an SP dimension
# (neither slices).
_GRAPH_EDGES = [
    {"from": "janus_text_encoder.encode", "to": "janus_llama"},
    {"from": "janus_llama", "to": "janus_text_encoder.decode"},
    {"from": "janus_text_encoder.decode", "to": "end"},
]

VOCAB = 128
HIDDEN = 32

_TEXT_CONFIG = {
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "max_position_embeddings": 128,
}

# Packed length 3 + 4 = 7 (odd), so an SP=2 slice has to pad before splitting and
# the output gather has to strip that padding back off.
_ALIGN_CONVERSATIONS = [
    [
        (torch.tensor([5, 6], dtype=torch.long), torch.tensor([-100, -100], dtype=torch.long), "user"),
        (torch.tensor([7], dtype=torch.long), torch.tensor([7], dtype=torch.long), "assistant"),
    ],
    [
        (torch.tensor([11], dtype=torch.long), torch.tensor([-100], dtype=torch.long), "user"),
        (torch.tensor([12, 13, 14], dtype=torch.long), torch.tensor([12, 13, 14], dtype=torch.long), "assistant"),
    ],
]


class _FakeTokenizer:
    """Only what ``JanusChatTemplate.__init__`` reads, so no real vocab is needed."""

    bos_token = "<s>"
    eos_token = "</s>"
    boi_token = "<boi>"
    eoi_token = "<eoi>"
    bos_token_id = 1
    eos_token_id = 2
    boi_token_id = 3
    eoi_token_id = 4
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {
            "<s>": 1,
            "</s>": 2,
            "<begin_of_image>": 3,
            "<end_of_image>": 4,
            "<image_placeholder>": 5,
        }.get(token, 0)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _conversation_list() -> list[list[ConversationItem]]:
    """Two samples of pre-tokenised text, so no tokenisation happens in the step.

    ``labels`` / ``attention_mask`` / ``_OMNI_TOKEN_LEN`` are what the chat
    template writes; setting them directly keeps the SP comparison about the
    backbone's slice rather than about rendering. The template object is still
    needed because ``encode`` reads ``pack_input_ids`` off it.
    """
    out: list[list[ConversationItem]] = []
    for sample in _ALIGN_CONVERSATIONS:
        items: list[ConversationItem] = []
        for ids, labels, role in sample:
            items.append(
                ConversationItem(
                    type="text",
                    value=ids.clone(),
                    role=role,
                    meta={
                        "labels": labels.clone(),
                        "attention_mask": torch.ones_like(ids),
                        _OMNI_TOKEN_LEN: int(ids.numel()),
                    },
                )
            )
        out.append(items)
    return out


def _chat_template() -> JanusChatTemplate:
    return JanusChatTemplate(_FakeTokenizer())


def _build_modules(*, attn_implementation: str, device: torch.device, dtype: torch.dtype):
    """The two modules the SP path touches, seeded with the same weights on every rank."""
    torch.manual_seed(2026)
    ops = make_eager_ops_config(attn_implementation=attn_implementation)
    text_encoder = OMNI_ACCELERATED_MODEL_REGISTRY["janus_text_encoder"]()(
        OMNI_CONFIG_REGISTRY["janus_text_encoder"]()(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            tie_word_embeddings=False,
            lm_head_bias=False,
            ops_implementation=ops,
        )
    )
    text_encoder._chat_template = _chat_template()
    # ``OMNI_CONFIG_REGISTRY[key]`` is a factory returning the class, so it takes
    # two calls (the qwen3_moe harness does the same). The backbone owns no wte /
    # lm_head: ``embed_tokens`` is an ``nn.Identity`` over the text encoder's.
    llm = OMNI_ACCELERATED_MODEL_REGISTRY["janus_llama"]()(
        OMNI_CONFIG_REGISTRY["janus_llama"]()(text_config=_TEXT_CONFIG, ops_implementation=ops)
    )
    return {
        "janus_text_encoder": text_encoder.to(device=device, dtype=dtype),
        "janus_llama": llm.to(device=device, dtype=dtype),
    }


def _omni(modules: dict[str, torch.nn.Module]) -> OmniModel:
    config = OmniConfig(
        _module_entries={name: {"model_path": name} for name in MODULES},
        training_graphs={"default": _GRAPH_EDGES},
        generation_graphs={},
    )
    return OmniModel(config, modules)


def _graph_step(modules: dict[str, torch.nn.Module], conversations, *, backward: bool = False):
    """One forward over the graph, optionally backward.

    Gradients are what an SP regression corrupts: a loss can match while a gather
    averages the shards instead of concatenating them, and only the per-parameter
    gradient shows it. Flattened from any DTensor so the two SP settings compare.
    """
    model = _omni(modules)
    out = model.forward({"conversation_list": conversations}, node_runner=TrainNodeRunner())
    loss = out["loss"]
    if not backward:
        return loss

    loss.backward()
    grads: dict[tuple[str, str], torch.Tensor] = {}
    for name, module in modules.items():
        for param_name, param in module.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad
            grad = grad.full_tensor() if isinstance(grad, DTensor) else grad
            grads[(name, param_name)] = grad.detach().float().clone()
    return loss, grads


def _adamw_step(modules: dict[str, torch.nn.Module]) -> None:
    params = [param for module in modules.values() for param in module.parameters()]
    torch.optim.AdamW(params, lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False).step()


def _param_probe(modules: dict[str, torch.nn.Module]) -> dict[str, tuple[float, float]]:
    """``{param: (sum, sum_of_squares)}`` over the whole model, as a float64 pair."""
    probe: dict[str, tuple[float, float]] = {}
    for name, module in modules.items():
        for param_name, param in module.named_parameters():
            flat = param.detach().float().flatten()
            probe[f"{name}.{param_name}"] = (float(flat.sum()), float(flat.pow(2).sum()))
    return probe


def _shard(modules: dict[str, torch.nn.Module], *, mesh, mp_policy=None) -> None:
    """FSDP2 over the composed tree, the way ``OmniModelRuntime`` wraps it.

    Leaf-first: the vocabulary embedding gets its own group (it is vocab-parallel
    in the launcher config), then the module, then each decoder layer, then the
    backbone. ``reshard_after_forward`` is left on for the leaves so a backward
    has something to re-gather.
    """
    extra = {"mp_policy": mp_policy} if mp_policy is not None else {}
    text_encoder = modules["janus_text_encoder"]
    llm = modules["janus_llama"]
    fully_shard(text_encoder.embed_tokens, mesh=mesh, reshard_after_forward=True, **extra)
    fully_shard(text_encoder, mesh=mesh, **extra)
    for layer in llm.language_model.layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=True, **extra)
    fully_shard(llm, mesh=mesh, **extra)


def _run_once(
    *,
    rank: int,
    device: torch.device,
    device_type: str,
    dtype: torch.dtype,
    attn_implementation: str,
    dp: int,
    sp: int,
    mp_policy=None,
) -> dict:
    """One FSDP2 step at a fixed ``(dp, sp)`` split, reported as floats."""
    ps_name = f"align_dp{dp}_sp{sp}"
    ps = _init_parallel_state(
        dp_size=dp,
        dp_shard_size=dp,
        ulysses_size=sp,
        dp_mode="fsdp2",
        device_type=device_type,
        name=ps_name,
    )
    modules = _build_modules(attn_implementation=attn_implementation, device=device, dtype=dtype)

    # Data placement follows the split, exactly as the dataloader does it:
    # at SP=1 the two ranks are two DP replicas, so each takes one sample; at
    # SP=2 there is one DP replica and the SP group replicates the same batch.
    all_conversations = _conversation_list()
    conversations = [all_conversations[rank]] if sp == 1 else all_conversations

    with use_parallel_state(ps_name):
        _shard(modules, mesh=ps.fsdp_mesh, mp_policy=mp_policy)
        loss, grads = _graph_step(modules, conversations, backward=True)
        _adamw_step(modules)
        probe = _param_probe(modules)
    return {
        "loss": float(loss.detach()),
        # A DTensor's ``full_tensor()`` is still the whole parameter, so reduce to
        # a pair of scalars rather than a float() that only works for 1-element tensors.
        "grads": {f"{name}:{param}": (float(g.sum()), float(g.pow(2).sum())) for (name, param), g in grads.items()},
        "param_probe": probe,
    }


def _worker(rank: int, rendezvous: str, out_dir: str, kind: str) -> None:
    world = 2
    if kind == "cpu":
        device_type = "cpu"
        device = torch.device("cpu")
        dtype = torch.float32
        attn_implementation = "veomni_sdpa"
        mp_policy = None
        backend = "gloo"
    else:
        device_type = get_device_type()
        get_torch_device().set_device(rank)
        device = torch.device(f"{device_type}:{rank}")
        dtype = torch.bfloat16
        # ``veomni_sdpa`` is what the qwen3vl / qwen3_moe align tests use on an
        # accelerator too: the eager kernel ignores the SP lengths, so a slice
        # would go untested (or fail the cu_seqlens check for the wrong reason).
        attn_implementation = "veomni_sdpa"
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        backend = get_dist_comm_backend()

    dist.init_process_group(backend, init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        # SP=1: dp=2. SP=2: dp=1, sp=2 — the same world size, so the only thing
        # that changes is whether the backbone sequence is sliced.
        reports = {}
        for name, dp, sp in (("sp1", world, 1), ("sp2", world // 2, 2)):
            reports[name] = _run_once(
                rank=rank,
                device=device,
                device_type=device_type,
                dtype=dtype,
                attn_implementation=attn_implementation,
                dp=dp,
                sp=sp,
                mp_policy=mp_policy,
            )
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(reports), encoding="utf-8")
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


def _assert_matches(sp1: dict, sp2: dict, *, loss_tol: float, grad_tol: float, step_tol: float) -> None:
    """Compare an SP=2 step against the SP=1 one.

    The two runs see the same graph but a different split of it: SP=1 has one
    sample per rank and averages two per-sample gradients, SP=2 runs the packed
    pair on both ranks and slices the sequence. In FP32 that is the same number
    to summation order, which is what the CPU case pins. In BF16 the two
    accumulation orders differ enough that only a gross divergence is
    meaningful, so the accelerator case gets the same per-metric tolerances the
    qwen3 / qwen3_moe align tests use rather than one blanket number.
    """
    assert sp1["loss"] == pytest.approx(sp2["loss"], abs=loss_tol, rel=loss_tol)

    assert sp1["grads"].keys() == sp2["grads"].keys(), "an SP step dropped or added a gradient"
    for name, (sum1, sq1) in sp1["grads"].items():
        sum2, sq2 = sp2["grads"][name]
        assert sum2 == pytest.approx(sum1, abs=grad_tol, rel=grad_tol), name
        assert sq2 == pytest.approx(sq1, abs=grad_tol, rel=grad_tol), name

    # After the AdamW step the weights are the gradient reduction's output, so a
    # slice that never reached the group lands here even if the loss matched.
    assert sp1["param_probe"].keys() == sp2["param_probe"].keys()
    assert len(sp1["param_probe"]) > 8, "expected the whole graph, not a spot check"
    for name, (sum1, sq1) in sp1["param_probe"].items():
        sum2, sq2 = sp2["param_probe"][name]
        assert sum2 == pytest.approx(sum1, abs=step_tol, rel=step_tol), name
        assert sq2 == pytest.approx(sq1, abs=step_tol, rel=step_tol), name


@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_sp_matches_single_rank_on_cpu(tmp_path: Path):
    """FSDP2 at SP=2 must reproduce SP=1 on loss, gradients and the update.

    The tolerance is float32 summation order, not a modelling fudge: the two runs
    compute the same graph, and only the length of the sequence each rank owns
    differs.
    """
    out_dir = tmp_path / "sp_align_cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(_worker, args=(str(out_dir / "rendezvous"), str(out_dir), "cpu"), nprocs=2, join=True)
    for rank in range(2):
        reports = json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        # FP32: the two runs compute the same graph, so the residual really is
        # summation order.
        _assert_matches(reports["sp1"], reports["sp2"], loss_tol=1e-5, grad_tol=1e-4, step_tol=1e-4)


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) or runner_device_count() < 2,
    reason="requires >=2 CUDA or NPU devices",
)
def test_sp_matches_single_rank_on_accelerator(tmp_path: Path):
    """The same comparison over real NCCL/HCCL collectives in BF16."""
    out_dir = tmp_path / "sp_align_acc"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(_worker, args=(str(out_dir / "rendezvous"), str(out_dir), "accelerator"), nprocs=2, join=True)
    for rank in range(2):
        reports = json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        assert reports
        # BF16 over real collectives — the tolerances the qwen3 align tests use.
        _assert_matches(reports["sp1"], reports["sp2"], loss_tol=5e-3, grad_tol=2e-2, step_tol=1e-2)
