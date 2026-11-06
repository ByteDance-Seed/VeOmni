"""Qwen3-VL SeedOmni graph parity across the conversation and packed graphs.

Two things are checked, in one file because they share the fixture:

1. **The two graphs agree.** A Qwen3-VL training step must produce the same loss
   whether it runs the conversation graph (carrier walked item by item) or the
   packed graph (one flat sequence built on the CPU). They encode the same model,
   so a difference means one of the graphs is computing something else.
2. **Sequence parallelism preserves that loss.** The same step under FSDP2 at
   SP=1 and SP=2 (with an odd packed length, so the Ulysses slice has to pad and
   the output gather has to unpad) must match, for both graphs.

The reference is the SP=1 run of the same graph rather than a monolithic HF model:
the monolithic comparison for this family is already covered by
`docs/seed_omni/models/qwen3vl.md` §6 (vision + backbone hidden states match
`Qwen3VLForConditionalGeneration` bit-for-bit), and re-deriving it here would test
the ViT rather than the graph wiring this file is about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.seed_omni.collator import SeedOmniCollator
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY
from veomni.models.seed_omni.accelerated.utils.executor import TrainNodeRunner
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.processing import Qwen3VLTextEncoderPreprocessor
from veomni.models.seed_omni.modules.qwen3vl.vision.processing import _OMNI_GRID
from veomni.models.seed_omni.processing_omni import OmniProcessor
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


_CI_CPU_2 = LaunchSpec(nproc=2, device="cpu")
_CI_ACCELERATOR_2 = LaunchSpec(nproc=2, device="accelerator")

VOCAB = 64
HIDDEN = 16
# in_channels * temporal_patch_size * patch_size**2 — one ViT patch row.
PATCH_ROW = 3 * 2 * 16 * 16
# One 4x4 image grid -> 16 patch rows -> 4 merged tokens at merge size 2.
_GRID = [1, 4, 4]

_TEXT_CONFIG = {
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 64,
}
_VISION_CONFIG = {
    "depth": 1,
    "hidden_size": HIDDEN,
    "intermediate_size": 32,
    "num_heads": 2,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": HIDDEN,
    "num_position_embeddings": 256,  # 16 per side; the grid must fit inside
    "deepstack_visual_indexes": [],
}

_CONVERSATION_EDGES = [
    {"from": "qwen3vl_vision", "to": "qwen3vl_llm"},
    {"from": "qwen3vl_text_encoder.encode", "to": "qwen3vl_llm"},
    {"from": "qwen3vl_llm", "to": "qwen3vl_text_encoder.decode"},
    {"from": "qwen3vl_text_encoder.decode", "to": "end"},
]
_PACKED_EDGES = [
    {"from": "qwen3vl_text_encoder.pack_encode", "to": "qwen3vl_vision.pack_encode"},
    {"from": "qwen3vl_vision.pack_encode", "to": "qwen3vl_llm.pack_forward"},
    {"from": "qwen3vl_llm.pack_forward", "to": "qwen3vl_text_encoder.pack_decode"},
    {"from": "qwen3vl_text_encoder.pack_decode", "to": "end"},
]
_MODULES = ("qwen3vl_vision", "qwen3vl_text_encoder", "qwen3vl_llm")


class _SmallTokenizer:
    """Qwen3-VL chat markers mapped into a tiny vocabulary.

    Only the surface `Qwen3VLChatTemplate` reads: the marker ids plus a callable
    that turns text into in-range ids.
    """

    eos_token = "<|im_end|>"
    eos_token_id = 2
    pad_token_id = 0
    special_tokens = {
        "<|im_start|>": 3,
        "<|im_end|>": 2,
        "<|vision_start|>": 4,
        "<|vision_end|>": 5,
    }

    def convert_tokens_to_ids(self, token):
        return self.special_tokens.get(token, -1)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        ids, index = [], 0
        while index < len(text):
            for token, token_id in sorted(self.special_tokens.items(), key=lambda kv: -len(kv[0])):
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    break
            else:
                ids.append(ord(text[index]) % 20 + 8)
                index += 1
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        del ids, kwargs
        return ""


def _samples(n: int = 2) -> list[list[ConversationItem]]:
    """`n` conversations, each one image plus a supervised assistant turn.

    Deliberately uneven text so the packed length is not a power of two, which is
    what makes an SP=2 slice need padding.
    """
    texts = ["hi", "hello there"][:n]
    out = []
    for text in texts:
        out.append(
            [
                ConversationItem(type="text", value=text, role="user"),
                ConversationItem(
                    type="image",
                    value=torch.rand(16, PATCH_ROW),
                    role="user",
                    meta={_OMNI_GRID: list(_GRID)},
                ),
                ConversationItem(type="text", value="ok", role="assistant"),
            ]
        )
    return out


def _build_modules(*, attn_implementation: str, device: torch.device, dtype: torch.dtype):
    """Three Qwen3-VL modules with a seeded, tiny architecture."""
    torch.manual_seed(2026)
    ops = make_eager_ops_config(attn_implementation=attn_implementation)
    text_encoder = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3vl_text_encoder"]()(
        OMNI_CONFIG_REGISTRY["qwen3vl_text_encoder"]()(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            tie_word_embeddings=False,
            lm_head_bias=False,
            ops_implementation=ops,
        )
    )
    text_encoder._chat_template = Qwen3VLChatTemplate(_SmallTokenizer())
    llm = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3vl_llm"]()(
        OMNI_CONFIG_REGISTRY["qwen3vl_llm"]()(
            text_config=_TEXT_CONFIG,
            spatial_merge_size=2,
            ops_implementation=ops,
        )
    )
    vision = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3vl_vision"]()(
        OMNI_CONFIG_REGISTRY["qwen3vl_vision"]()(
            vision_config=_VISION_CONFIG,
            ops_implementation=ops,
        )
    )
    return {
        "qwen3vl_vision": vision.to(device=device, dtype=dtype),
        "qwen3vl_text_encoder": text_encoder.to(device=device, dtype=dtype),
        "qwen3vl_llm": llm.to(device=device, dtype=dtype),
    }


def _omni(modules: dict[str, torch.nn.Module], edges: list[dict[str, str]]) -> OmniModel:
    config = OmniConfig(
        _module_entries={name: {"model_path": name} for name in _MODULES},
        training_graphs={"default": edges},
        generation_graphs={},
    )
    return OmniModel(config, modules)


def _conversation_batch(samples: list[list[ConversationItem]]) -> dict:
    template = Qwen3VLChatTemplate(_SmallTokenizer())
    return {"conversation_list": [template.tokenize_conversation(sample) for sample in samples]}


def _packed_batch(samples: list[list[ConversationItem]]) -> dict:
    preprocessor = Qwen3VLTextEncoderPreprocessor(
        Qwen3VLChatTemplate(_SmallTokenizer()), packed_preprocess=True, spatial_merge_size=2
    )
    processor = OmniProcessor({"qwen3vl_text_encoder": preprocessor})
    return SeedOmniCollator()([processor.preprocess(sample) for sample in samples])


def _graph_loss(modules: dict[str, torch.nn.Module], edges: list[dict[str, str]], batch: dict) -> torch.Tensor:
    model = _omni(modules, edges)
    first = next(iter(modules.values()))
    device, dtype = first.device, first.dtype
    moved = {
        key: (value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device))
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
    out = model.forward(moved, node_runner=TrainNodeRunner())
    return out["loss"]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_conversation_and_packed_graphs_agree_on_cpu():
    """The two graphs encode the same model, so a one-step loss must match.

    Each graph gets its own modules *and its own samples*. A conversation run is
    destructive to the carrier: `scatter_qwen3vl_visual_embeds` writes the vision
    output back onto the item (`slot.item.value = embeds_split[idx]`), so an image
    item that entered as `(16, 1536)` patches leaves as `(4, 16)` merged embeds.
    Feeding that item to the packed graph makes the ViT slice it as raw patches
    and fail on the shape. Nothing is wrong with the design — one carrier is meant
    to be consumed once — the test just must not share one.
    """
    conversation_loss = _graph_loss(
        _build_modules(attn_implementation="veomni_sdpa", device=torch.device("cpu"), dtype=torch.float32),
        _CONVERSATION_EDGES,
        _conversation_batch(_samples(2)),
    )
    packed_loss = _graph_loss(
        _build_modules(attn_implementation="veomni_sdpa", device=torch.device("cpu"), dtype=torch.float32),
        _PACKED_EDGES,
        _packed_batch(_samples(2)),
    )

    assert torch.allclose(conversation_loss.float(), packed_loss.float(), atol=1e-6, rtol=1e-6), (
        f"conversation {float(conversation_loss)} != packed {float(packed_loss)}"
    )


def _sp_worker(rank: int, rendezvous: str, out_dir: str, device_type: str) -> None:
    """Run both graphs at SP=1 and SP=2 and record each loss."""
    world = 2
    device = torch.device(f"{device_type}:{rank}")
    if device_type != "cpu":
        get_torch_device().set_device(rank)
    dist.init_process_group(
        "gloo" if device_type == "cpu" else get_dist_comm_backend(),
        init_method=f"file://{rendezvous}",
        world_size=world,
        rank=rank,
    )
    try:
        dtype = torch.float32 if device_type == "cpu" else torch.bfloat16
        attn = "veomni_sdpa"
        reports: dict[str, float] = {}
        for name, edges, build_batch in (
            ("conversation", _CONVERSATION_EDGES, _conversation_batch),
            ("packed", _PACKED_EDGES, _packed_batch),
        ):
            for sp_size in (1, 2):
                ps = _init_parallel_state(
                    dp_size=world if sp_size == 1 else 1,
                    dp_shard_size=world if sp_size == 1 else 1,
                    ulysses_size=sp_size,
                    dp_mode="fsdp2",
                    device_type=device_type,
                    name=f"{name}_sp{sp_size}",
                )
                with use_parallel_state(f"{name}_sp{sp_size}"):
                    modules = _build_modules(attn_implementation=attn, device=device, dtype=dtype)
                    if device_type != "cpu":
                        from torch.distributed.fsdp import fully_shard

                        for module in modules.values():
                            fully_shard(module, mesh=ps.fsdp_mesh)
                    samples = _samples(2)
                    # dp=2 takes one conversation per rank; sp=2 replicates the batch.
                    batch = build_batch([samples[rank]] if sp_size == 1 else samples)
                    loss = _graph_loss(modules, edges, batch)
                    reports[f"{name}_sp{sp_size}"] = float(loss.detach().float())
                clear_parallel_state()
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(reports), encoding="utf-8")
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_sp_matches_single_rank_on_cpu(tmp_path: Path):
    """FSDP2 at SP=2 (with SP padding) must reproduce the SP=1 loss for both graphs."""
    out_dir = tmp_path / "sp_align_cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _sp_worker,
        args=(str(out_dir / "rendezvous"), str(out_dir), "cpu"),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        reports = json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        assert set(reports) == {"conversation_sp1", "conversation_sp2", "packed_sp1", "packed_sp2"}
        for name in ("conversation", "packed"):
            assert reports[f"{name}_sp2"] == pytest.approx(reports[f"{name}_sp1"], abs=1e-5), (rank, name, reports)
        assert reports["conversation_sp1"] == pytest.approx(reports["packed_sp1"], abs=1e-5), (rank, reports)


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) or runner_device_count() < 2,
    reason="requires >=2 CUDA or NPU devices",
)
def test_sp_matches_single_rank_on_accelerator(tmp_path: Path):
    """Same comparison over real NCCL/HCCL collectives in BF16."""
    out_dir = tmp_path / "sp_align_acc"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _sp_worker,
        args=(str(out_dir / "rendezvous"), str(out_dir), get_device_type()),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        reports = json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        assert reports
        for name in ("conversation", "packed"):
            assert reports[f"{name}_sp2"] == pytest.approx(reports[f"{name}_sp1"], rel=5e-3), (rank, name, reports)
        assert reports["conversation_sp1"] == pytest.approx(reports["packed_sp1"], rel=5e-3), (rank, reports)
