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
from torch.distributed.tensor import DTensor

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
from veomni.models.seed_omni.modules.qwen3vl.vision.processing import Qwen3VLVisionPreprocessor
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
    # Two layers so `deepstack_visual_indexes` can point at a real interior layer
    # (index 0), which is what makes the DeepStack path run at all.
    "depth": 2,
    "hidden_size": HIDDEN,
    "intermediate_size": 32,
    "num_heads": 2,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": HIDDEN,
    "num_position_embeddings": 256,  # 16 per side; the grid must fit inside
    "deepstack_visual_indexes": [0],
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


def _vision_preprocessor():
    """The real vision preprocessor, so media are patchified by the same code as
    production rather than by hand-built tensors."""
    from transformers import Qwen2VLImageProcessor, Qwen3VLVideoProcessor

    image_processor = Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=16 * 16 * 4,
        max_pixels=16 * 16 * 256,
        size={"shortest_edge": 64, "longest_edge": 256},
    )
    video_processor = Qwen3VLVideoProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=16 * 16 * 4,
        max_pixels=16 * 16 * 256,
    )
    preprocessor = Qwen3VLVisionPreprocessor(
        image_processor, video_processor, dtype=torch.float32, spatial_merge_size=2
    )
    preprocessor.bind_dummy_inputs(
        OMNI_CONFIG_REGISTRY["qwen3vl_vision"]()(vision_config=dict(_VISION_CONFIG)), dtype=torch.float32
    )
    return preprocessor


def _samples() -> list[list[ConversationItem]]:
    """One batch covering every modality the tower must handle in a single step:

    * sample 0 — text + **image**
    * sample 1 — text + **video** (4 frames -> 2 temporal patches)
    * sample 2 — **text only**, so the vision preprocessor injects an ``is_dummy``
      FSDP-anchor placeholder for it

    The dummy matters beyond FSDP alignment: it is the only path where the packed
    and conversation graphs disagree about *where* the visual rows sit, and where
    `visual_is_real` has to be right for the scatter.

    Text lengths are deliberately uneven and the packed total is not a power of
    two, so an SP=2 slice has to pad and the gather has to unpad.
    """
    from veomni.data.seed_omni.utils.media_metadata import VIDEO_METADATA_KEY, VideoMetadata
    from veomni.data.seed_omni.utils.video import VideoInputs

    # Fixed seed: `_samples` draws the image and video pixels at random, and two
    # runs must see the *same* batch for their losses and gradients to be comparable.
    torch.manual_seed(1234)
    video_metadata = VideoMetadata(fps=2.0, total_num_frames=4, frames_indices=list(range(4)), requested_fps=2.0)
    samples: list[list[ConversationItem]] = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(
                type="image",
                value=torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8),
                role="user",
            ),
            ConversationItem(type="text", value="ok", role="assistant"),
        ],
        [
            ConversationItem(type="text", value="hello there", role="user"),
            ConversationItem(
                type="video",
                value=VideoInputs(video=torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8)),
                role="user",
                meta={VIDEO_METADATA_KEY: video_metadata},
            ),
            ConversationItem(type="text", value="sure", role="assistant"),
        ],
        [
            ConversationItem(type="text", value="just text", role="user"),
            ConversationItem(type="text", value="ok then", role="assistant"),
        ],
    ]
    # Patchify + inject the per-sample dummy, exactly as the dataset transform does.
    _vision_preprocessor()({"conversation_list": samples}, inference=False)
    return samples


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


def _graph_loss(
    modules: dict[str, torch.nn.Module],
    edges: list[dict[str, str]],
    batch: dict,
    *,
    count_sp: bool = False,
    backward: bool = False,
):
    """One forward over the graph, optionally backward, optionally counting SP calls.

    `backward` returns `(loss, grads, sp_calls)` where ``grads`` maps ``(module,
    param)`` to a host copy of its gradient, flattened from any DTensor. Gradients
    are what an SP regression actually corrupts — a loss can match while a gather
    silently averages the shards instead of concatenating them, and only the
    per-parameter gradient shows it.

    The SP counter is what keeps the SP test honest: a loss that matches because
    neither run took the SP branch proves nothing, and that is exactly what a
    regression in the parallelism guards looks like. Every module that slices
    imports the helper by name, so the counter wraps each of those bindings.
    """
    calls = {"slice": 0, "pad": 0}
    restores = []
    if count_sp:
        from veomni.models.seed_omni.modules.qwen3vl.llm.accelerated import accelerated as llm_acc
        from veomni.models.seed_omni.modules.qwen3vl.llm.accelerated import packed as llm_packed
        from veomni.models.seed_omni.modules.qwen3vl.text_encoder.accelerated import packed as te_packed
        from veomni.models.seed_omni.modules.qwen3vl.vision.accelerated import packed as vision_packed

        targets = [llm_acc, llm_packed, te_packed, vision_packed]
        for module in targets:
            for attr, key in (
                ("slice_input_tensor", "slice"),
                ("sp_pad", "pad"),
                ("sp_pad_and_slice", "pad"),
            ):
                original = getattr(module, attr, None)
                if original is None:
                    continue

                def wrapper(*args, _original=original, _key=key, **kwargs):
                    calls[_key] += 1
                    return _original(*args, **kwargs)

                restores.append((module, attr, original))
                setattr(module, attr, wrapper)

    try:
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
        loss = out["loss"]
        grads: dict[tuple[str, str], torch.Tensor] = {}
        if backward:
            loss.backward()
            for name, module in modules.items():
                for param_name, param in module.named_parameters():
                    if param.grad is None:
                        continue
                    grad = param.grad
                    grad = grad.full_tensor() if isinstance(grad, DTensor) else grad
                    grads[(name, param_name)] = grad.detach().float().clone()
    finally:
        for target, attr, original in restores:
            setattr(target, attr, original)

    if backward:
        return loss, grads, dict(calls)
    return (loss, dict(calls)) if count_sp else loss


def _adamw_step(modules: dict[str, torch.nn.Module]) -> None:
    """One AdamW step over the modules, using the `.grad` the backward left."""
    params = [param for module in modules.values() for param in module.parameters()]
    optimizer = torch.optim.AdamW(params, lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False)
    optimizer.step()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_conversation_and_packed_graphs_agree_on_cpu():
    """The two graphs encode the same model, so a step must match on loss,
    every parameter gradient, and the AdamW update that follows.

    Loss alone is not enough: a graph could reach the same scalar while routing
    the gradient of the dummy anchor, the video frames or the image differently,
    and only the per-parameter comparison shows it.

    Each graph gets its own modules *and its own samples*. A conversation run is
    destructive to the carrier: `scatter_qwen3vl_visual_embeds` writes the vision
    output back onto the item (`slot.item.value = embeds_split[idx]`), so an image
    item that entered as `(16, 1536)` patches leaves as `(4, 16)` merged embeds.
    Feeding that item to the packed graph makes the ViT slice it as raw patches
    and fail on the shape. Nothing is wrong with the design — one carrier is meant
    to be consumed once — the test just must not share one.
    """
    conversation_modules = _build_modules(
        attn_implementation="veomni_sdpa", device=torch.device("cpu"), dtype=torch.float32
    )
    packed_modules = _build_modules(attn_implementation="veomni_sdpa", device=torch.device("cpu"), dtype=torch.float32)
    conversation_loss, conversation_grads, _ = _graph_loss(
        conversation_modules, _CONVERSATION_EDGES, _conversation_batch(_samples()), backward=True
    )
    packed_loss, packed_grads, _ = _graph_loss(packed_modules, _PACKED_EDGES, _packed_batch(_samples()), backward=True)

    assert torch.allclose(conversation_loss.float(), packed_loss.float(), atol=1e-6, rtol=1e-6), (
        f"conversation {float(conversation_loss)} != packed {float(packed_loss)}"
    )

    # The dummy anchor only exists to keep the vision tower in the graph, so a
    # gradient reaching every vision parameter is the evidence it did.
    assert conversation_grads and packed_grads
    assert set(conversation_grads) == set(packed_grads)
    vision_grads = [key for key in packed_grads if key[0] == "qwen3vl_vision"]
    assert len(vision_grads) > 1, "the dummy anchor must give the vision tower a gradient"

    for key in conversation_grads:
        torch.testing.assert_close(
            packed_grads[key], conversation_grads[key], atol=1e-6, rtol=1e-5, msg=lambda m, k=key: f"{k}: {m}"
        )

    # AdamW is a deterministic function of the gradients, so a matching step is
    # implied by matching gradients — but it is the thing a trainer applies, and
    # comparing post-step parameters catches a gradient that agrees while some
    # other input to the optimizer (weight decay over the wrong tensor set) does not.
    _adamw_step(conversation_modules)
    _adamw_step(packed_modules)
    for name, module in conversation_modules.items():
        reference = dict(module.named_parameters())
        actual = dict(packed_modules[name].named_parameters())
        for param_name in reference:
            torch.testing.assert_close(
                actual[param_name].detach(),
                reference[param_name].detach(),
                atol=1e-6,
                rtol=1e-5,
                msg=lambda m, k=(name, param_name): f"{k}: {m}",
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
                    # Shard on CPU too. FSDP2 is what reduces gradient across the
                    # group; without the wrap, SP=2 ranks each keep their own
                    # un-reduced share and step to different weights. Skipping the
                    # wrap on CPU (as an earlier version did) hides exactly the
                    # behaviour this test is meant to check.
                    from torch.distributed.fsdp import fully_shard

                    for module in modules.values():
                        fully_shard(module, mesh=ps.fsdp_mesh)
                    samples = _samples()
                    # Both settings get the same batch: at dp=2 every rank must see
                    # identical data for the mean to be comparable, and at sp=2 the
                    # ranks are slices of one replicated batch.
                    batch = build_batch(samples)
                    loss, _, sp_calls = _graph_loss(modules, edges, batch, count_sp=True, backward=True)
                    reports[f"{name}_sp{sp_size}"] = float(loss.detach().float())
                    reports[f"{name}_sp{sp_size}_sp_calls"] = sp_calls
                    # Compare *parameters after an optimizer step*, not pre-reduce
                    # gradients. Under FSDP2 the cross-rank reduction of `.grad`
                    # happens when the optimizer consumes it, so reading `param.grad`
                    # right after `backward()` shows this rank's un-reduced share —
                    # which legitimately differs between SP ranks (each holds half
                    # the sequence). The stepped weights are the reduced result.
                    _adamw_step(modules)
                    # Every parameter, not a sample: two fingerprints per tensor
                    # catch both an offset and a scale difference, so a single
                    # divergent tensor cannot hide behind the ones not probed.
                    reports[f"{name}_sp{sp_size}_param_probe"] = {
                        f"{mod_name}.{param_name}": [
                            float(full.sum()),
                            float((full * full).sum()),
                        ]
                        for mod_name, module in sorted(modules.items())
                        for param_name, param in sorted(module.named_parameters())
                        for full in [(param.full_tensor() if isinstance(param, DTensor) else param).detach().float()]
                    }
                clear_parallel_state()
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(reports), encoding="utf-8")
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_sp_matches_single_rank_on_cpu(tmp_path: Path):
    """FSDP2 at SP=2 (with SP padding) must reproduce SP=1 on both graphs.

    Checks the loss and, after one AdamW step, every parameter of every module:
    the stepped weights are the gradient reduction's output, so a slice that
    never reached the group would land here as a diverged tensor.
    """
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
        for name in ("conversation", "packed"):
            # The SP=2 run must have actually sliced; otherwise "same loss" only
            # proves the same code path ran twice.
            calls = reports[f"{name}_sp2_sp_calls"]
            assert calls["slice"] and calls["pad"], (rank, name, calls)
            assert not reports[f"{name}_sp1_sp_calls"]["slice"], (rank, name, "SP=1 must not slice")
            assert reports[f"{name}_sp2"] == pytest.approx(reports[f"{name}_sp1"], abs=1e-5), (rank, name, reports)
            # Parameters after one AdamW step: FSDP2 is what reduces `.grad`
            # across the group, so this is the first point where an SP that
            # failed to reduce shows up. Both fingerprints agree across the
            # whole model; the residual is float32 summation order, worst case
            # ~3e-6 relative on the smallest-magnitude bias.
            probe1 = reports[f"{name}_sp1_param_probe"]
            probe2 = reports[f"{name}_sp2_param_probe"]
            # Every parameter is probed, so this is a whole-model comparison and
            # not a spot check.
            assert probe1.keys() == probe2.keys()
            assert len(probe1) > 40, (name, "expected the whole model", len(probe1))
            for param_name, (sum1, sq1) in probe1.items():
                sum2, sq2 = probe2[param_name]
                assert sum2 == pytest.approx(sum1, abs=1e-5), (rank, name, param_name, sum1, sum2)
                assert sq2 == pytest.approx(sq1, abs=1e-5), (rank, name, param_name, sq1, sq2)

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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_every_module_meters_its_own_tokens_on_both_graphs():
    """Per-module token accounting has to exist on both graphs, and the modules that
    see one row per sample must agree on how many samples there are.

    `OmniEnvironMeter` pairs a module's `seqlens` against the batch's `ds_idx` when
    it decides which module drives multi-source accounting, and warns (reporting
    zero tokens) when no module aligns. Before this, Qwen3-VL registered no meter at
    all: every `trace/<module>/...` metric read zero.
    """
    for name, edges, build_batch in (
        ("conversation", _CONVERSATION_EDGES, _conversation_batch),
        ("packed", _PACKED_EDGES, _packed_batch),
    ):
        modules = _build_modules(attn_implementation="veomni_sdpa", device=torch.device("cpu"), dtype=torch.float32)
        _graph_loss(modules, edges, build_batch(_samples()))

        # `metric_meter_collect` drains the buffer, so collect once per module.
        collected = {module_name: module.metric_meter_collect() for module_name, module in modules.items()}
        for module_name, (flops, seqlens) in collected.items():
            assert flops > 0, (name, module_name, "no FLOPs reported")
            assert seqlens and any(s > 0 for s in seqlens), (name, module_name, seqlens)

        # The backbone sees one row per sample on both graphs, so its lengths must
        # be per-sample: a batch-total entry makes `estimate_flops` square the sum
        # and leaves `OmniEnvironMeter` nothing to align against `ds_idx`.
        llm_lens = collected["qwen3vl_llm"][1]
        assert len(llm_lens) == 3, (name, "backbone must report one length per sample", llm_lens)

        if name == "packed":
            # One packed row per sample, so the text encoder and the backbone that
            # both consume that packed sequence must report the same sample count.
            te_lens = collected["qwen3vl_text_encoder"][1]
            assert len(te_lens) == len(llm_lens) == 3, (te_lens, llm_lens)

        # The tower's blocks and patch_embed run on raw patches — `merge²` rows per
        # merged token — so those two terms must be `merge²` times what the same
        # merged count would give. Check the ratio directly rather than a floor:
        # dropping the expansion is the mistake this guards.
        if name == "conversation":
            vision_flops = collected["qwen3vl_vision"][0]
            vision_lens = collected["qwen3vl_vision"][1]
            cfg = modules["qwen3vl_vision"].config.vision_config
            merge_area = cfg.spatial_merge_size**2
            merged = sum(vision_lens)
            raw = merged * merge_area

            mlp_n = cfg.hidden_size * cfg.intermediate_size * 3
            attn_n = cfg.hidden_size * (3 * cfg.hidden_size + cfg.hidden_size)
            block_n = (mlp_n + attn_n) * cfg.depth
            patch_n = cfg.hidden_size * cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2
            # These two terms are the ones that scale with raw rows.
            at_merged = 6 * (block_n + patch_n) * merged / 1e12
            at_raw = 6 * (block_n + patch_n) * raw / 1e12
            # Reported FLOPs must sit closer to the raw reading than the merged one.
            assert vision_flops > (at_merged + at_raw) / 2, (vision_flops, at_merged, at_raw)
