"""Composing Qwen3 + the Qwen3-VL ViT into a visual-instruction-tuning graph.

This is the acceptance test for `configs/seed_omni/Qwen/qwen3/visual_instruction_tuning/`:
two modules contributed by two different PRs — the Qwen3-VL vision tower and the
Qwen3 text encoder / backbone — composed into one training job by YAML alone.

Nothing here is a new module class. The whole recipe is `model_config:`
overrides:

* ``qwen3vl_vision``: ``out_hidden_size`` retargets the ViT's own patch merger at
  the LLM hidden size, ``disable_deepstack`` drops the features a plain backbone
  cannot consume, ``freeze`` keeps only that merger trainable;
* ``qwen3_text_encoder``: ``enable_image`` switches it to the Qwen3-VL chat
  template and turns ``freeze_model`` into "vision special-token rows only".

The three properties worth pinning are that the composed graph actually trains the
overrides it was configured with, that it converges on the toy batch rather than
merely running, and that it survives Ulysses SP at the issue's acceptance shape —
the SP slice crosses three modules with three different shard layouts (patch rows,
token ids, packed sequence), so a wrong-ranks all-to-all shows up as a loss
mismatch at SP=2 and again at SP=4.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.training_utils import make_eager_ops_config
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY
from veomni.models.seed_omni.accelerated.utils.executor import TrainNodeRunner
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.qwen3.text_encoder.accelerated import Qwen3TextEncoderAccelerated
from veomni.models.seed_omni.modules.qwen3.text_encoder.processing import Qwen3TextEncoderPreprocessor
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
_CI_CPU_4 = LaunchSpec(nproc=4, device="cpu")
_CI_ACCELERATOR_4 = LaunchSpec(nproc=4, device="accelerator")

VOCAB = 64
HIDDEN = 16

# 4 query / 2 KV heads: divisible by SP=2 *and* SP=4, so the same config carries the
# parity cases up to the SP=4 shape from #1072. SP=4 also puts the KV heads below the
# Ulysses size, which is the GQA repeat path rather than the plain head split.
_TEXT_CONFIG = {
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "max_position_embeddings": 64,
}
# `depth: 2` so a DeepStack index could point at a real interior layer; the recipe
# disables DeepStack, and this test asserts that.
_VISION_CONFIG = {
    "depth": 2,
    "hidden_size": HIDDEN,
    "intermediate_size": 32,
    "num_heads": 4,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": HIDDEN,
    "num_position_embeddings": 256,
    "deepstack_visual_indexes": [0],
}

_MODULES = ("qwen3vl_vision", "qwen3_text_encoder", "qwen3_llm")

_GRAPH_EDGES = [
    {"from": "qwen3vl_vision", "to": "qwen3_llm"},
    {"from": "qwen3_text_encoder.encode", "to": "qwen3_llm"},
    {"from": "qwen3_llm", "to": "qwen3_text_encoder.decode"},
    {"from": "qwen3_text_encoder.decode", "to": "end"},
]

_VISION_SPECIAL_TOKENS = {
    "<|im_start|>": 3,
    "<|im_end|>": 2,
    "<|vision_start|>": 4,
    "<|vision_end|>": 5,
    "<|image_pad|>": 6,
}


class _SmallTokenizer:
    """Qwen3 chat markers mapped into a tiny vocabulary.

    Only the surface the templates read: marker ids plus a callable that turns
    text into in-range ids.
    """

    eos_token = "<|im_end|>"
    eos_token_id = 2
    pad_token_id = 0
    bos_token = None
    bos_token_id = None

    def convert_tokens_to_ids(self, token):
        """Map a special token to its id, or ``-1`` for an unknown token."""
        return _VISION_SPECIAL_TOKENS.get(token, -1)

    def __call__(self, text, add_special_tokens=False):
        """Greedy longest-match tokenize ``text`` against the special tokens."""
        del add_special_tokens
        ids, index = [], 0
        while index < len(text):
            for token, token_id in sorted(_VISION_SPECIAL_TOKENS.items(), key=lambda kv: -len(kv[0])):
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    break
            else:
                ids.append(ord(text[index]) % 20 + 8)
                index += 1
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        """Decode to text; the FSM only checks for an empty result."""
        del ids, kwargs
        return ""


def _vision_preprocessor() -> Qwen3VLVisionPreprocessor:
    """The real tower preprocessor, so media are patchified by production code."""
    from transformers import Qwen2VLImageProcessor, Qwen3VLVideoProcessor

    preprocessor = Qwen3VLVisionPreprocessor(
        Qwen2VLImageProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=16 * 16 * 4,
            max_pixels=16 * 16 * 256,
            size={"shortest_edge": 64, "longest_edge": 256},
        ),
        Qwen3VLVideoProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=16 * 16 * 4,
            max_pixels=16 * 16 * 256,
        ),
        dtype=torch.float32,
        spatial_merge_size=2,
    )
    preprocessor.bind_dummy_inputs(
        OMNI_CONFIG_REGISTRY["qwen3vl_vision"]()(vision_config=dict(_VISION_CONFIG)), dtype=torch.float32
    )
    return preprocessor


def _text_preprocessor() -> Qwen3TextEncoderPreprocessor:
    """The Qwen3 worker in image mode, through the real config-driven selector.

    ``build_chat_template`` needs a checkpoint on disk; a ``config.json`` carrying
    ``enable_image: true`` is all it reads, so this exercises the same branch the
    dataset worker takes instead of hand-building the template.
    """
    from unittest.mock import patch

    import veomni.models.seed_omni.modules.qwen3.text_encoder.processing as processing

    config_dir = Path(tempfile.mkdtemp()) / "qwen3_text_encoder"
    config_dir.mkdir(parents=True)
    OMNI_CONFIG_REGISTRY["qwen3_text_encoder"]()(
        vocab_size=VOCAB, hidden_size=HIDDEN, tie_word_embeddings=True, enable_image=True
    ).save_pretrained(config_dir)
    # `build_tokenizer` would read a real tokenizer; the tiny stub is enough here.
    with patch.object(processing, "build_tokenizer", lambda module_path: _SmallTokenizer()):
        return Qwen3TextEncoderPreprocessor.from_pretrained(str(config_dir))


def _samples() -> list[list[ConversationItem]]:
    """One caption sample and one text-only sample.

    The text-only sample is what forces the vision preprocessor's per-sample
    ``is_dummy`` anchor, so the tower runs on a step where it has no real image.
    """
    torch.manual_seed(7)
    return [
        [
            ConversationItem(type="text", value="describe", role="user"),
            ConversationItem(type="image", value=torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8), role="user"),
            ConversationItem(type="text", value="a cat", role="assistant"),
        ],
        [
            ConversationItem(type="text", value="just text", role="user"),
            ConversationItem(type="text", value="ok then", role="assistant"),
        ],
    ]


def _build_modules(*, device: torch.device, dtype: torch.dtype) -> dict[str, torch.nn.Module]:
    """The three modules with the recipe's ``model_config`` overrides applied."""
    torch.manual_seed(2026)
    ops = make_eager_ops_config(attn_implementation="veomni_sdpa")

    vision = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3vl_vision"]()(
        OMNI_CONFIG_REGISTRY["qwen3vl_vision"]()(
            vision_config=dict(_VISION_CONFIG),
            out_hidden_size=HIDDEN,
            disable_deepstack=True,
            freeze=True,
            ops_implementation=ops,
        )
    )
    text_encoder = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_text_encoder"]()(
        OMNI_CONFIG_REGISTRY["qwen3_text_encoder"]()(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            tie_word_embeddings=True,
            enable_image=True,
            ops_implementation=ops,
        )
    )
    text_encoder._tokenizer = _SmallTokenizer()
    text_encoder._chat_template = _text_preprocessor()._chat_template
    llm = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_llm"]()(
        OMNI_CONFIG_REGISTRY["qwen3_llm"]()(text_config=dict(_TEXT_CONFIG), ops_implementation=ops)
    )
    vision.freeze_model()
    text_encoder.freeze_model()
    return {
        "qwen3vl_vision": vision.to(device=device, dtype=dtype),
        "qwen3_text_encoder": text_encoder.to(device=device, dtype=dtype),
        "qwen3_llm": llm.to(device=device, dtype=dtype),
    }


def _batch() -> dict:
    """Preprocess the samples exactly as the dataset transform does."""
    processor = OmniProcessor({"qwen3vl_vision": _vision_preprocessor(), "qwen3_text_encoder": _text_preprocessor()})
    samples = [processor.preprocess(sample, inference=False)["conversation_list"] for sample in _samples()]
    return {"conversation_list": samples}


def _omni(modules: dict[str, torch.nn.Module]) -> OmniModel:
    """Wrap ``modules`` in the recipe's training graph as an :class:`OmniModel`."""
    config = OmniConfig(
        _module_entries={name: {"model_path": name} for name in _MODULES},
        training_graphs={"default": _GRAPH_EDGES},
        generation_graphs={},
    )
    return OmniModel(config, modules)


def _graph_loss(modules: dict[str, torch.nn.Module], batch: dict, *, backward: bool = False):
    """Run one forward through the composed graph and return the scalar loss."""
    model = _omni(modules)
    first = next(iter(modules.values()))
    device, dtype = first.device, first.dtype
    moved = {
        key: (
            value.to(device=device, dtype=dtype)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value.to(device=device)
        )
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
    out = model.forward(moved, node_runner=TrainNodeRunner())
    loss = out["loss"]
    if not backward:
        return loss
    loss.backward()
    grads: dict[tuple[str, str], torch.Tensor] = {}
    for name, module in modules.items():
        for param_name, param in module.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            grads[(name, param_name)] = grad.detach().float().clone()
    return loss, grads


def test_the_recipe_configs_resolve_to_the_three_modules_from_two_families():
    """`base.yaml` + `modules_train.yaml` produce the composed model, no Python."""
    from veomni.arguments.omni_arguments_types import (
        OmniArguments,
        OmniDataArguments,
        OmniInferArguments,
        OmniModelRuntimeArguments,
        build_omni_model_runtime_args,
    )

    config_dir = (
        Path(__file__).resolve().parents[4] / "configs" / "seed_omni" / "Qwen" / "qwen3" / "visual_instruction_tuning"
    )
    args = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=".",
            model_config={
                "modules": str(config_dir / "modules_train.yaml"),
                "train_graph": str(config_dir / "graph_train.yaml"),
                "infer_graph": {"understanding": str(config_dir / "graph_infer.yaml")},
            },
        ),
        data=OmniDataArguments(train_path="."),
        infer=OmniInferArguments(),
    )
    config = build_omni_model_runtime_args(args).to_hf_config()

    assert set(config.module_names) == set(_MODULES)
    # A module from another checkpoint keeps its own absolute path; the Qwen3
    # modules resolve under `model.model_path` (here ".").
    vision_path = config._module_entries["qwen3vl_vision"]["model_path"]
    assert os.path.isabs(vision_path), vision_path
    assert vision_path.endswith("qwen3vl_vision")
    assert not os.path.isabs(config._module_entries["qwen3_llm"]["model_path"])
    # The two overrides that make this a vision recipe, straight from the YAML.
    assert config._module_entries["qwen3vl_vision"]["model_config"] == {
        "out_hidden_size": 1024,
        "disable_deepstack": True,
        "freeze": True,
    }
    assert config._module_entries["qwen3_text_encoder"]["model_config"] == {"enable_image": True}
    endpoints = {edge["from"] for edge in config.training_graph} | {edge["to"] for edge in config.training_graph}
    assert {"qwen3vl_vision", "qwen3_text_encoder.encode", "qwen3_llm"} <= endpoints
    assert "understanding" in config.generation_graphs


def test_image_mode_retargets_the_merger_and_freezes_the_tower():
    """`out_hidden_size` / `disable_deepstack` / `freeze` do what the recipe claims."""
    modules = _build_modules(device=torch.device("cpu"), dtype=torch.float32)
    vision = modules["qwen3vl_vision"]

    # The merger is the projection into the LLM space: no separate module.
    assert vision.config.vision_config.out_hidden_size == HIDDEN
    assert vision.config.vision_config.deepstack_visual_indexes == []

    trainable = {name for name, param in vision.named_parameters() if param.requires_grad}
    assert trainable, "the patch merger must stay trainable"
    # ViT blocks are frozen; nothing outside `visual.merger` trains.
    assert all("merger" in name for name in trainable), sorted(trainable)

    text_encoder = modules["qwen3_text_encoder"]
    # The tied table stays trainable at the parameter level (the tied head needs
    # it) but a row mask restricts which rows a gradient can reach.
    assert text_encoder.embed_tokens.weight.requires_grad
    assert text_encoder._trainable_row_mask is not None
    assert int(text_encoder._trainable_row_mask.sum()) == len(type(text_encoder)._VISION_SPECIAL_TOKENS)


def test_eager_inference_can_start_from_a_stock_qwen3vl_checkpoint(tmp_path: Path):
    """The recipe's inference path loads the retargeted merger instead of raising.

    Eager inference (`fsdp_mode: eager`, `OmniModel.from_pretrained`) goes through HF's
    own `from_pretrained`, not VeOmni's loader — so `_MergerProjectionConverter`,
    which drops the stock-shape `linear_fc2` during training, never runs. Without
    the module declaring that one mismatch, the documented convert-then-infer flow
    dies at startup with a size-mismatch error. A second case pins the guard: a
    mismatch the module did *not* declare still raises.
    """
    Vision = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3vl_vision"]()
    VisionConfig = OMNI_CONFIG_REGISTRY["qwen3vl_vision"]()

    # A stock checkpoint at the ViT's own width (`out_hidden_size` unretargeted).
    stock_config = VisionConfig(vision_config=dict(_VISION_CONFIG), ops_implementation=make_eager_ops_config())
    source = tmp_path / "qwen3vl_vision"
    source.mkdir(parents=True)
    Vision(stock_config).save_pretrained(source)
    from transformers import Qwen2VLImageProcessor, Qwen3VLVideoProcessor

    Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2).save_pretrained(source)
    Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2).save_pretrained(source)

    retargeted = VisionConfig(
        vision_config=dict(_VISION_CONFIG), out_hidden_size=HIDDEN, ops_implementation=make_eager_ops_config()
    )
    model = Vision.from_pretrained(str(source), config=retargeted)
    assert model.visual.merger.linear_fc2.weight.shape[0] == HIDDEN

    # The guard: retargeting the ViT's hidden size is not a declared mismatch.
    stray = VisionConfig(
        vision_config={**_VISION_CONFIG, "hidden_size": HIDDEN * 2, "num_heads": 2},
        out_hidden_size=HIDDEN,
        ops_implementation=make_eager_ops_config(),
    )
    with pytest.raises(RuntimeError, match="not the ones this module re-initialises"):
        Vision.from_pretrained(str(source), config=stray)


def test_a_composed_step_trains_only_what_the_overrides_selected():
    """One graph step backprops into the merger, the vision rows and the backbone."""
    modules = _build_modules(device=torch.device("cpu"), dtype=torch.float32)
    loss, grads = _graph_loss(modules, _batch(), backward=True)

    assert torch.isfinite(loss), float(loss.detach())
    assert float(loss.detach()) > 0

    # The LLM is the trainable body.
    assert any(name[0] == "qwen3_llm" for name in grads)
    # The retargeted merger carries the image gradient into the LLM space.
    assert ("qwen3vl_vision", "visual.merger.linear_fc2.weight") in grads
    # The vision special-token rows are the only place the text encoder trains.
    table_grad = grads[("qwen3_text_encoder", "embed_tokens.weight")]
    touched = (table_grad.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
    expected_rows = sorted(
        _VISION_SPECIAL_TOKENS[token] for token in Qwen3TextEncoderAccelerated._VISION_SPECIAL_TOKENS
    )
    assert touched == expected_rows, (touched, expected_rows)


def test_the_frozen_rows_survive_an_adamw_step():
    """`configure_optimizer` keeps weight decay off the frozen rows and the tied head.

    AdamW decays a whole parameter outside the gradient, so the row mask alone is not
    enough: without this hook the 61 frozen rows — and the tied output head that reads
    them — would shrink every step. This drives the real ``configure_optimizer`` and
    compares against a dense AdamW reference, so it fails if the hook is dropped, if
    it stops finding the table's group, or if the row slice is wrong.
    """
    modules = _build_modules(device=torch.device("cpu"), dtype=torch.float32)
    text_encoder = modules["qwen3_text_encoder"]
    table = text_encoder.embed_tokens.weight
    before = table.detach().clone()

    loss, _ = _graph_loss(modules, _batch(), backward=True)

    lr, weight_decay = 1e-2, 0.5
    optimizer = torch.optim.AdamW([table], lr=lr, weight_decay=weight_decay, foreach=False)
    # The hook takes the table's decay over, so AdamW must not apply it to any row.
    text_encoder.configure_optimizer(optimizer)
    assert all(group["weight_decay"] == 0.0 for group in optimizer.param_groups)
    optimizer.step()

    mask = text_encoder._trainable_row_mask
    after = table.detach()
    # Frozen rows are bit-identical: no gradient, and no decay either.
    assert torch.equal(after[~mask], before[~mask])
    # Trainable rows moved, and moved the way AdamW-with-decay would have moved them.
    assert not torch.equal(after[mask], before[mask])
    reference = torch.optim.AdamW([before[mask].clone()], lr=lr, weight_decay=weight_decay, foreach=False)
    reference.param_groups[0]["params"][0].grad = table.grad.detach()[mask].clone()
    reference.step()
    torch.testing.assert_close(after[mask], reference.param_groups[0]["params"][0].detach(), atol=1e-7, rtol=1e-6)
    assert torch.isfinite(loss)


def test_the_toy_recipe_converges():
    """#1072's acceptance check: a few dozen steps on the toy batch actually descend.

    "The graph runs" is not convergence — a freeze that leaves the head untrained, a
    detached merger, or a vision prefix the LLM never sees all produce a flat loss
    curve from a valid forward. So this drives the recipe's optimizer path
    (AdamW + ``configure_optimizer``) and requires the loss to come down, which is
    only possible if the gradient reaches the three blocks the overrides selected.
    """
    steps = 30
    modules = _build_modules(device=torch.device("cpu"), dtype=torch.float32)
    text_encoder = modules["qwen3_text_encoder"]
    params = [param for module in modules.values() for param in module.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-2, weight_decay=0.01, foreach=False)
    text_encoder.configure_optimizer(optimizer)

    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # The graph writes hidden states back onto the carrier it is given, so each
        # step needs a fresh batch rather than the consumed one.
        loss = _graph_loss(modules, _batch())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        assert torch.isfinite(loss), losses

    head = sum(losses[:3]) / 3
    tail = sum(losses[-3:]) / 3
    assert tail < 0.95 * head, (head, tail, losses)
    # Not just "lower on average": the loss has to descend window by window, so a
    # curve that only drifts down by luck cannot pass. Window means rather than
    # per-step values, and 1% relative slack rather than a strict inequality, because
    # one step's rounding order differs across CPU kernels / torch builds and the late
    # windows are where that noise is largest. The slack only permits a 1% *rise*; the
    # real curve drops 1.2-3.9% per window, and the first/last comparison below still
    # requires the whole run to trend down even with it.
    windows = [sum(losses[i : i + 5]) / 5 for i in range(0, steps - 4, 5)]
    assert all(later < earlier * 1.01 for earlier, later in zip(windows, windows[1:])), windows
    assert windows[-1] < windows[0], windows


def test_image_mode_rejects_an_untied_head_and_unknown_vision_tokens():
    """The two silent-failure modes are errors, not wrong training."""
    TextEncoderAccelerated = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_text_encoder"]()
    TextEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3_text_encoder"]()

    untied = TextEncoderAccelerated(
        TextEncoderConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            tie_word_embeddings=False,
            enable_image=True,
            ops_implementation=make_eager_ops_config(),
        )
    )
    untied._tokenizer = _SmallTokenizer()
    with pytest.raises(ValueError, match="needs the head tied"):
        untied.freeze_model()

    class _UnknownTokenizer(_SmallTokenizer):
        def convert_tokens_to_ids(self, token):
            """Simulate a tokenizer that has no id for the token at all."""
            return None

    unknown = TextEncoderAccelerated(
        TextEncoderConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            tie_word_embeddings=True,
            enable_image=True,
            ops_implementation=make_eager_ops_config(),
        )
    )
    unknown._tokenizer = _UnknownTokenizer()
    with pytest.raises(ValueError, match="does not know"):
        unknown.freeze_model()

    class _NegativeTokenizer(_SmallTokenizer):
        def convert_tokens_to_ids(self, token):
            """Simulate a tokenizer that reports an unk id of ``-1``."""
            return -1

    wrapping = TextEncoderAccelerated(
        TextEncoderConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            tie_word_embeddings=True,
            enable_image=True,
            ops_implementation=make_eager_ops_config(),
        )
    )
    wrapping._tokenizer = _NegativeTokenizer()
    with pytest.raises(ValueError, match="outside the embedding table"):
        wrapping.freeze_model()


def _sp_worker(rank: int, rendezvous: str, out_dir: str, world: int, device_type: str) -> None:
    """Run the composed graph at SP=1 and SP=``world`` and record loss + stepped params.

    ``device_type="cpu"`` uses Gloo and float32; anything else is the accelerator
    path the same worker takes over NCCL/HCCL in bfloat16, so the CPU case pins the
    composition and the accelerator case pins the real collectives.
    """
    on_accelerator = device_type != "cpu"
    device = torch.device(f"{device_type}:{rank}" if on_accelerator else "cpu")
    if on_accelerator:
        get_torch_device().set_device(rank)
    dist.init_process_group(
        "gloo" if not on_accelerator else get_dist_comm_backend(),
        init_method=f"file://{rendezvous}",
        world_size=world,
        rank=rank,
    )
    try:
        reports: dict[str, object] = {}
        for sp_size in (1, world):
            name = f"sp{sp_size}"
            ps = _init_parallel_state(
                dp_size=world if sp_size == 1 else 1,
                dp_shard_size=world if sp_size == 1 else 1,
                ulysses_size=sp_size,
                dp_mode="fsdp2",
                device_type=device_type,
                name=name,
            )
            with use_parallel_state(name):
                from torch.distributed.fsdp import fully_shard

                modules = _build_modules(device=device, dtype=torch.float32 if not on_accelerator else torch.bfloat16)
                for module in modules.values():
                    fully_shard(module, mesh=ps.fsdp_mesh)
                loss, _ = _graph_loss(modules, _batch(), backward=True)
                reports[f"{name}_loss"] = float(loss.detach().float())
                # Compare weights after one step: FSDP2 reduces `.grad` when the
                # optimizer consumes it, so this is the first point where an SP
                # slice that failed to reduce shows up.
                params = [param for module in modules.values() for param in module.parameters()]
                torch.optim.AdamW(params, lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False).step()
                reports[f"{name}_probe"] = {
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


def _assert_sp_parity(out_dir: Path, world: int, *, rtol: float = 0.0, atol: float = 1e-5) -> None:
    """Every rank saw the same loss, and every param left the step with the SP=1 value."""
    for rank in range(world):
        reports = json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        assert reports[f"sp{world}_loss"] == pytest.approx(reports["sp1_loss"], rel=rtol, abs=atol), (
            rank,
            reports["sp1_loss"],
        )
        probe1, probe_other = reports["sp1_probe"], reports[f"sp{world}_probe"]
        assert probe1.keys() == probe_other.keys()
        for name, (total1, square1) in probe1.items():
            total_other, square_other = probe_other[name]
            assert total_other == pytest.approx(total1, rel=rtol, abs=atol), (rank, name, total1, total_other)
            assert square_other == pytest.approx(square1, rel=rtol, abs=atol), (rank, name, square1, square_other)


@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_sp2_matches_sp1_on_the_composed_graph(tmp_path: Path):
    """Ulysses SP across ViT + text encoder + backbone must not change the result.

    The three modules shard three different things (patch rows, token ids, the
    packed sequence), so this is the composition-level check that each module's
    slice/gather lands on its own group.
    """
    out_dir = tmp_path / "sp_compose"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _sp_worker,
        args=(str(out_dir / "rendezvous"), str(out_dir), _CI_CPU_2.nproc, "cpu"),
        nprocs=_CI_CPU_2.nproc,
        join=True,
    )

    _assert_sp_parity(out_dir, world=_CI_CPU_2.nproc)


@pytest.mark.launch_plan(provider=_CI_CPU_4)
def test_sp4_matches_sp1_on_the_composed_graph(tmp_path: Path):
    """The #1072 acceptance shape: uniform SP=4 over ViT + text encoder + backbone.

    SP=4 is the size that validated the composition invariant, and it is the first
    size where the tower's 4 attention heads are fully split *and* the LLM's 2 KV
    heads sit below the Ulysses size — so this also covers the GQA head-repeat path
    that SP=2 never reaches.
    """
    out_dir = tmp_path / "sp_compose4"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _sp_worker,
        args=(str(out_dir / "rendezvous"), str(out_dir), _CI_CPU_4.nproc, "cpu"),
        nprocs=_CI_CPU_4.nproc,
        join=True,
    )

    _assert_sp_parity(out_dir, world=_CI_CPU_4.nproc)


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_4)
@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) or runner_device_count() < _CI_ACCELERATOR_4.nproc,
    reason="requires >=4 CUDA or NPU devices",
)
def test_sp4_matches_sp1_on_accelerator(tmp_path: Path):
    """Same SP=1 vs SP=4 comparison over real NCCL/HCCL collectives in bfloat16.

    The CPU case proves the composition; this one proves it over the collective
    implementation CI actually trains with. BF16 all-to-all/gather rounding is not
    bit-exact against an SP=1 run, so this compares with the same relative
    tolerance the per-family alignment tests use.
    """
    out_dir = tmp_path / "sp_compose4_acc"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _sp_worker,
        args=(str(out_dir / "rendezvous"), str(out_dir), _CI_ACCELERATOR_4.nproc, get_device_type()),
        nprocs=_CI_ACCELERATOR_4.nproc,
        join=True,
    )

    _assert_sp_parity(out_dir, world=_CI_ACCELERATOR_4.nproc, rtol=5e-3, atol=1e-4)
