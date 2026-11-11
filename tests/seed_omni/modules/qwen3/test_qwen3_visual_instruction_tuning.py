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
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from scripts.ci.accelerator.launch_plan import LaunchSpec, from_parameter, runner_device_count
from tests.tools.launch_utils import spawn_with_rendezvous
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


_CI_ACCELERATOR_4 = LaunchSpec(nproc=4, device="accelerator")

# Everything here builds and runs on CPU unless a case says otherwise; the two SP
# cases declare their own multi-process plans and the accelerator case below
# overrides this. Without the module-level plan the single-process cases fall back
# to `default_plan`, which claims an accelerator card each on a GPU runner.
pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

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


def test_the_hook_rebuilds_the_table_group_and_survives_a_restore():
    """The param-group change and the restore rule, both from the contract.

    `configure_optimizer` moves the table into its own decay-free group when it
    shares one, which changes the group count — part of the optimizer-state key
    space. This pins that layout, then simulates what loading a checkpoint does to
    it (param groups rebuilt from their saved hyperparameters, decay back at 0.5)
    and requires the next step to keep the frozen rows out of the decay anyway.
    """
    modules = _build_modules(device=torch.device("cpu"), dtype=torch.float32)
    text_encoder = modules["qwen3_text_encoder"]
    table = text_encoder.embed_tokens.weight
    # A group the table shares with something else, so the hook has to split it.
    shared = [table, torch.nn.Parameter(torch.zeros(4))]
    optimizer = torch.optim.AdamW([{"params": shared, "weight_decay": 0.5}], lr=1e-2, foreach=False)

    text_encoder.configure_optimizer(optimizer)
    assert len(optimizer.param_groups) == 2, [len(g["params"]) for g in optimizer.param_groups]
    table_group = next(g for g in optimizer.param_groups if any(p is table for p in g["params"]))
    assert len(table_group["params"]) == 1, "the table must be alone in its group"
    assert table_group["weight_decay"] == 0.0
    untouched = next(g for g in optimizer.param_groups if any(p is not table for p in g["params"]))
    assert untouched["weight_decay"] == 0.5, "other parameters keep their decay"

    # A restore rebuilds groups with their saved hyperparameters: decay is back.
    table_group["weight_decay"] = 0.5
    mask = text_encoder._trainable_row_mask
    before = table.detach().clone()
    _graph_loss(modules, _batch(), backward=True)
    optimizer.step()

    assert torch.equal(table.detach()[~mask], before[~mask]), "frozen rows decayed after a restore"
    assert table_group["weight_decay"] == 0.0, "the step hook must re-zero the group decay"


@pytest.mark.parametrize("kind", ["master", "full", "offload"])
@pytest.mark.parametrize("multi", [False, True])
def test_image_row_decay_rejects_fp32_master_before_optimizer_mutation(kind, multi):
    """Do not silently decay a compute copy that an FP32 master will overwrite."""
    from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW
    from veomni.optim.master_fp32_adamw import FullFP32AdamW, MasterFP32AdamW
    from veomni.optim.optimizer import MultiOptimizer

    text_encoder = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    table = text_encoder.embed_tokens.weight
    other = torch.nn.Parameter(torch.ones(1))
    classes = {"master": MasterFP32AdamW, "full": FullFP32AdamW, "offload": CPUOffloadAdamW}
    group = {"params": [table, other], "weight_decay": 0.5}
    if kind == "offload":
        group["cpu_offload"] = True
    leaf = classes[kind]([group], lr=1e-2)
    optimizer = MultiOptimizer(text_encoder, {"leaf": leaf}, ["leaf"]) if multi else leaf
    before = table.detach().clone()
    with pytest.raises(RuntimeError, match="row-wise weight decay.*FP32 master"):
        text_encoder.configure_optimizer(optimizer)
    assert len(leaf.param_groups) == 1
    assert leaf.param_groups[0]["params"] == [table, other]
    assert leaf.param_groups[0]["weight_decay"] == 0.5
    assert not leaf._optimizer_step_pre_hooks
    assert not leaf.state
    assert torch.equal(table, before)


def test_image_zero_row_decay_remains_supported_by_fp32_master():
    """The unsupported combination requires nonzero row-specific decay."""
    from veomni.optim.master_fp32_adamw import FullFP32AdamW

    text_encoder = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    table = text_encoder.embed_tokens.weight
    optimizer = FullFP32AdamW([table], weight_decay=0.0)
    text_encoder.configure_optimizer(optimizer)
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert not optimizer._optimizer_step_pre_hooks


def test_image_row_mask_rejects_initialized_fp32_master_optimizer():
    from veomni.optim.master_fp32_adamw import FullFP32AdamW

    encoder = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    table = encoder.embed_tokens.weight
    optimizer = FullFP32AdamW([table], lr=1e-2, weight_decay=0.0)
    table.grad = torch.ones_like(table)
    optimizer.step()
    before = table.detach().clone()
    with pytest.raises(RuntimeError, match="requires a fresh FP32 master optimizer"):
        encoder.configure_optimizer(optimizer)
    assert "qwen3_image_row_mask" not in optimizer.param_groups[0]
    assert torch.equal(table, before)


def test_image_row_mask_rejects_multiple_optimizer_owners():
    from veomni.optim.master_fp32_adamw import FullFP32AdamW
    from veomni.optim.optimizer import MultiOptimizer

    encoder = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    table = encoder.embed_tokens.weight
    first = FullFP32AdamW([table], weight_decay=0.0)
    second = FullFP32AdamW([table], weight_decay=0.5)
    optimizer = MultiOptimizer(encoder, {"first": first, "second": second}, ["first", "second"])
    with pytest.raises(RuntimeError, match="multiple optimizer owners"):
        encoder.configure_optimizer(optimizer)
    assert "qwen3_image_row_mask" not in first.param_groups[0]
    assert first.param_groups[0]["weight_decay"] == 0.0
    assert second.param_groups[0]["weight_decay"] == 0.5
    assert not first.state and not second.state


@pytest.mark.parametrize("kind", ["adamw", "muon"])
@pytest.mark.parametrize("mode", ["same_mask", "text_to_image", "image_to_text", "changed_mask"])
def test_image_row_mask_fp32_master_dcp_resume_contract(tmp_path, mode, kind):
    """Frozen-row history must not be restored across different row recipes."""
    from types import SimpleNamespace

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim.master_fp32_adamw import FullFP32AdamW
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    def build(text_only=False, change_mask=False):
        encoder = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
        if text_only:
            encoder._trainable_row_mask = None
        elif change_mask:
            encoder._trainable_row_mask = encoder._trainable_row_mask.clone()
            encoder._trainable_row_mask[0] = ~encoder._trainable_row_mask[0]
        table = encoder.embed_tokens.weight
        if kind == "muon":
            groups = add_qwen38_plans(encoder, [{"params": [table], "weight_decay": 0.0}])
            leaf = Qwen38MuonAdamW(groups, lr=1e-2)
        else:
            leaf = FullFP32AdamW([table], lr=1e-2, weight_decay=0.0)
        optimizer = MultiOptimizer(encoder, {"leaf": leaf}, ["leaf"])
        encoder.configure_optimizer(optimizer)
        return encoder, table, leaf, optimizer

    source, source_table, source_leaf, source_opt = build(text_only=mode == "text_to_image")
    source_table.grad = torch.ones_like(source_table)
    if source._trainable_row_mask is not None:
        source_table.grad[~source._trainable_row_mask] = 0
    source_opt.step()
    source_opt.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": source, "optimizer": source_opt, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, target_table, target_leaf, target_opt = build(
        text_only=mode == "image_to_text", change_mask=mode == "changed_mask"
    )
    before = {name: value.detach().clone() for name, value in target.state_dict().items()}

    def load():
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": target_opt, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )

    if mode != "same_mask":
        with pytest.raises(RuntimeError, match="layout|identity|group|recipe"):
            load()
        assert not target_leaf.state
        for name, value in target.state_dict().items():
            assert torch.equal(value, before[name])
        return
    load()
    torch.testing.assert_close(target_table, source_table, rtol=0, atol=0)
    for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            target_leaf.state[target_table][key], source_leaf.state[source_table][key], rtol=0, atol=0
        )
    grad = torch.ones_like(source_table)
    grad[~source._trainable_row_mask] = 0
    source_table.grad, target_table.grad = grad, grad.clone()
    source_opt.step()
    target_opt.step()
    torch.testing.assert_close(target_table, source_table, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["adamw", "muon"])
@pytest.mark.parametrize("mode", ["text_to_image", "image_to_text", "changed_mask"])
def test_image_row_mask_fp32_master_direct_load_rejects_unproven_history(mode, kind):
    import copy

    from veomni.optim.master_fp32_adamw import FullFP32AdamW
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    source = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    target = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    if mode == "text_to_image":
        source._trainable_row_mask = None
    if mode == "image_to_text":
        target._trainable_row_mask = None
    if mode == "changed_mask":
        target._trainable_row_mask = target._trainable_row_mask.clone()
        target._trainable_row_mask[0] = ~target._trainable_row_mask[0]
    source_table, target_table = source.embed_tokens.weight, target.embed_tokens.weight
    if kind == "muon":
        old = Qwen38MuonAdamW(add_qwen38_plans(source, [{"params": [source_table], "weight_decay": 0.0}]), lr=1e-2)
        new = Qwen38MuonAdamW(add_qwen38_plans(target, [{"params": [target_table], "weight_decay": 0.0}]), lr=1e-2)
    else:
        old = FullFP32AdamW([source_table], lr=1e-2, weight_decay=0.0)
        new = FullFP32AdamW([target_table], lr=1e-2, weight_decay=0.0)
    source.configure_optimizer(old)
    target.configure_optimizer(new)
    source_table.grad = torch.ones_like(source_table)
    old.step()
    before = target_table.detach().clone()
    with pytest.raises(RuntimeError, match="row-mask identity"):
        new.load_state_dict(copy.deepcopy(old.state_dict()))
    assert not new.state
    assert torch.equal(target_table, before)


@pytest.mark.parametrize("kind", ["adamw", "muon"])
def test_image_row_mask_fp32_master_direct_load_rejects_nonzero_decay(kind):
    import copy

    from veomni.optim.master_fp32_adamw import FullFP32AdamW
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    source = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    target = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    if kind == "muon":
        old = Qwen38MuonAdamW(
            add_qwen38_plans(source, [{"params": [source.embed_tokens.weight], "weight_decay": 0.0}]), lr=1e-2
        )
        new = Qwen38MuonAdamW(
            add_qwen38_plans(target, [{"params": [target.embed_tokens.weight], "weight_decay": 0.0}]), lr=1e-2
        )
    else:
        old = FullFP32AdamW([source.embed_tokens.weight], lr=1e-2, weight_decay=0.0)
        new = FullFP32AdamW([target.embed_tokens.weight], lr=1e-2, weight_decay=0.0)
    source.configure_optimizer(old)
    target.configure_optimizer(new)
    source.embed_tokens.weight.grad = torch.ones_like(source.embed_tokens.weight)
    old.step()
    state = copy.deepcopy(old.state_dict())
    state["param_groups"][0]["weight_decay"] = 0.5
    before = target.embed_tokens.weight.detach().clone()
    with pytest.raises(RuntimeError, match="row-mask requires zero weight decay"):
        new.load_state_dict(state)
    assert not new.state and new.param_groups[0]["weight_decay"] == 0.0
    assert torch.equal(target.embed_tokens.weight, before)


def test_image_row_mask_fp32_master_direct_load_rejects_reordered_shared_group():
    import copy

    from veomni.optim.master_fp32_adamw import FullFP32AdamW

    source = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    target = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    source_table, target_table = source.embed_tokens.weight, target.embed_tokens.weight
    source_other = torch.nn.Parameter(torch.ones_like(source_table))
    target_other = torch.nn.Parameter(torch.ones_like(target_table))
    old = FullFP32AdamW([source_table, source_other], lr=1e-2, weight_decay=0.0)
    new = FullFP32AdamW([target_other, target_table], lr=1e-2, weight_decay=0.0)
    source.configure_optimizer(old)
    target.configure_optimizer(new)
    source_other.grad = torch.ones_like(source_other)
    old.step()
    before = target_table.detach().clone()
    with pytest.raises(RuntimeError, match="row-mask identity"):
        new.load_state_dict(copy.deepcopy(old.state_dict()))
    assert not new.state and torch.equal(target_table, before)


def _row_decay_guard_worker(rank, world_size, scenario, native, endpoint):
    """A local build failure must be coordinated by the real runtime wrapper."""
    from types import SimpleNamespace

    from veomni.models.model_runtime import VeOmniModelRuntime
    from veomni.optim.master_fp32_adamw import FullFP32AdamW

    device_type = get_device_type() if native else "cpu"
    backend = get_dist_comm_backend() if native else "gloo"
    if native:
        get_torch_device().set_device(rank)
    dist.init_process_group(
        backend, store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        _init_parallel_state(dp_size=world_size, dp_shard_size=world_size, device_type=device_type)
        text_encoder = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
        table = text_encoder.embed_tokens.weight
        cls = FullFP32AdamW if rank == 0 else torch.optim.AdamW
        optimizer = cls([table], lr=1e-2, weight_decay=0.5)
        before = table.detach().clone()

        def build():
            if rank == 0 and scenario == "build_failure":
                raise RuntimeError("injected optimizer construction failure")
            text_encoder.configure_optimizer(optimizer)

        def unexpected_validate():
            raise AssertionError("a failed optimizer build must not reach validation")

        runtime = SimpleNamespace(_build_optimizer=build, _validate_optimizer_checkpoint_mode=unexpected_validate)
        message = (
            (
                "injected optimizer construction failure"
                if scenario == "build_failure"
                else "row-wise weight decay.*FP32 master"
            )
            if rank == 0
            else "another rank"
        )
        with pytest.raises(RuntimeError, match=message):
            VeOmniModelRuntime._build_optimizer_checked(runtime)
        if rank == 0:
            assert optimizer.param_groups[0]["weight_decay"] == 0.5
            assert not optimizer._optimizer_step_pre_hooks
        assert not optimizer.state
        assert torch.equal(table, before)
        result = torch.ones((), dtype=torch.int32, device=f"{device_type}:{rank}" if native else "cpu")
        dist.all_reduce(result)
        assert result.item() == world_size
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("scenario", ["guard_mismatch", "build_failure"])
def test_image_row_decay_guard_coordinates_across_ranks(scenario):
    spawn_with_rendezvous(_row_decay_guard_worker, args=(2, scenario, False), nprocs=2)


@pytest.mark.launch_plan(
    provider=lambda context: LaunchSpec(nproc=2, device="accelerator")
    if context.capabilities.backend == "npu" and len(context.capabilities.devices) >= 2
    else LaunchSpec(device="cpu")
)
@pytest.mark.parametrize("scenario", ["guard_mismatch", "build_failure"])
def test_native_image_row_decay_guard_coordinates_across_ranks(scenario):
    if not IS_NPU_AVAILABLE or get_device_type() != "npu" or get_torch_device().device_count() < 2:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("Two usable NPU devices are required for this native test")
        pytest.skip("Requires two NPU devices")
    spawn_with_rendezvous(_row_decay_guard_worker, args=(2, scenario, True), nprocs=2)


def test_required_native_row_decay_guard_rejects_missing_npu(monkeypatch):
    monkeypatch.setenv("VEOMNI_REQUIRE_NPU_TESTS", "1")
    monkeypatch.setitem(
        test_native_image_row_decay_guard_coordinates_across_ranks.__globals__, "IS_NPU_AVAILABLE", False
    )
    with pytest.raises(pytest.fail.Exception, match="Two usable NPU devices are required"):
        test_native_image_row_decay_guard_coordinates_across_ranks("guard_mismatch")


def test_native_row_decay_guard_forwards_rendezvous_last(monkeypatch):
    import inspect
    from types import SimpleNamespace

    scope = test_native_image_row_decay_guard_coordinates_across_ranks.__globals__
    monkeypatch.setitem(scope, "IS_NPU_AVAILABLE", True)
    monkeypatch.setitem(scope, "get_device_type", lambda: "npu")
    monkeypatch.setitem(scope, "get_torch_device", lambda: SimpleNamespace(device_count=lambda: 2))
    calls = []

    def record(worker, args, *, nprocs):
        endpoint = object()
        bound = inspect.signature(worker).bind(0, *args, endpoint)
        assert nprocs == 2 and bound.arguments["native"] is True and bound.arguments["endpoint"] is endpoint
        calls.append(True)

    monkeypatch.setitem(scope, "spawn_with_rendezvous", record)
    test_native_image_row_decay_guard_coordinates_across_ranks("guard_mismatch")
    assert calls == [True]


@pytest.mark.parametrize("multi", [False, True])
def test_image_row_decay_rejects_qwen38_muon_adam_fallback(multi):
    """The mixed optimizer's Adam fallback also has an authoritative master."""
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    text_encoder = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    table = text_encoder.embed_tokens.weight
    groups = add_qwen38_plans(text_encoder, [{"params": [table], "weight_decay": 0.5}])
    leaf = Qwen38MuonAdamW(groups, lr=1e-2)
    optimizer = MultiOptimizer(text_encoder, {"leaf": leaf}, ["leaf"]) if multi else leaf
    with pytest.raises(RuntimeError, match="row-wise weight decay.*FP32 master"):
        text_encoder.configure_optimizer(optimizer)
    assert leaf.param_groups[0]["weight_decay"] == 0.5
    assert not leaf._optimizer_step_pre_hooks
    assert not leaf.state


def test_image_zero_row_decay_remains_supported_by_qwen38_muon():
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    text_encoder = _build_modules(device=torch.device("cpu"), dtype=torch.float32)["qwen3_text_encoder"]
    table = text_encoder.embed_tokens.weight
    groups = add_qwen38_plans(text_encoder, [{"params": [table], "weight_decay": 0.0}])
    optimizer = Qwen38MuonAdamW(groups, lr=1e-2)
    text_encoder.configure_optimizer(optimizer)
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert not optimizer._optimizer_step_pre_hooks
    table.grad = torch.zeros_like(table)
    optimizer.step()
    assert optimizer.state[table]["master_weight"].dtype == torch.float32


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
    """The silent-failure modes are errors, not wrong training."""
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


def test_a_typo_in_the_worker_model_config_is_an_error(tmp_path: Path):
    """A misspelled `model_config` key must not silently leave the mode off.

    The worker selects its chat template from `enable_image`, applying overrides
    with a permissive `update` — so `enable_imag: true` would land as a stray
    attribute and the run would quietly train the text-only template.

    The check must not over-fire either: transformers v5 stores several public
    config properties under a leading underscore (`output_attentions` ->
    `_output_attentions`), so a key that is only a near-miss of the *stored* name
    is still a valid key. And the same dict carries HF load options.
    """
    from unittest.mock import patch

    import veomni.models.seed_omni.modules.qwen3.text_encoder.processing as processing

    config_dir = tmp_path / "qwen3_text_encoder"
    config_dir.mkdir(parents=True)
    OMNI_CONFIG_REGISTRY["qwen3_text_encoder"]()(
        vocab_size=VOCAB, hidden_size=HIDDEN, tie_word_embeddings=True
    ).save_pretrained(config_dir)

    with patch.object(processing, "build_tokenizer", lambda module_path: _SmallTokenizer()):
        # The correct spelling still selects the VL template.
        worker = Qwen3TextEncoderPreprocessor.from_pretrained(str(config_dir), config_overrides={"enable_image": True})
        assert type(worker._chat_template).__name__ == "Qwen3VLChatTemplate"

        with pytest.raises(ValueError, match="look misspelled"):
            Qwen3TextEncoderPreprocessor.from_pretrained(str(config_dir), config_overrides={"enable_imag": True})

        # Valid keys that a naive near-miss check would reject.
        for accepted in (
            {"output_attentions": False},
            {"output_hidden_states": True},
            {"device_map": "cpu", "torch_dtype": "auto"},
        ):
            Qwen3TextEncoderPreprocessor.from_pretrained(str(config_dir), config_overrides=accepted)


def _table_rows(weight: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """``{trainable, frozen}`` sums over the embedding table's two row groups.

    ``weight`` is a DTensor under FSDP2, so it is gathered first: the row groups are
    defined on the full table, and gathering also exercises the offset the module
    computed for this rank's slice. Sums (not tensors) keep the report JSON small.
    """
    full = (weight.full_tensor() if isinstance(weight, DTensor) else weight).detach().float()
    return {"trainable": float(full[mask].sum()), "frozen": float(full[~mask].sum())}


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
                optimizer = torch.optim.AdamW(params, lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False)
                # The recipe's own hook, so the row-wise decay runs under the same
                # FSDP2/SP wrapping the trainer uses — the single-process tests call
                # it on an unsharded table and cannot see a DTensor-offset mistake.
                modules["qwen3_text_encoder"].configure_optimizer(optimizer)
                table_module = modules["qwen3_text_encoder"]
                before = _table_rows(table_module.embed_tokens.weight, table_module._trainable_row_mask)
                optimizer.step()
                after = _table_rows(table_module.embed_tokens.weight, table_module._trainable_row_mask)
                reports[f"{name}_rows"] = {
                    "trainable": [before["trainable"], after["trainable"]],
                    "frozen": [before["frozen"], after["frozen"]],
                    "groups": len(optimizer.param_groups),
                }
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

        # The row-wise decay under this wrapping: the frozen rows must be untouched
        # by the step (no gradient, and no group decay leaking onto them), the
        # vision rows must have moved, and SP=4 must agree with SP=1 on both. The
        # single-process test cannot show this — it never shards the table.
        rows1, rows_other = reports["sp1_rows"], reports[f"sp{world}_rows"]
        assert rows1["groups"] == rows_other["groups"], (rank, rows1["groups"], rows_other["groups"])
        for label in ("frozen", "trainable"):
            before1, after1 = rows1[label]
            before_other, after_other = rows_other[label]
            assert before_other == pytest.approx(before1, rel=rtol, abs=atol), (rank, label, "before")
            assert after_other == pytest.approx(after1, rel=rtol, abs=atol), (rank, label, "after")
        assert rows1["frozen"][0] == rows1["frozen"][1], (rank, "frozen rows must not move", rows1["frozen"])
        assert rows1["trainable"][0] != rows1["trainable"][1], (rank, "vision rows must move", rows1["trainable"])


@pytest.mark.launch_plan(provider=from_parameter("nproc", device="cpu"))
@pytest.mark.parametrize("nproc", [2, 4])
def test_sp_matches_sp1_on_the_composed_graph(nproc: int, tmp_path: Path):
    """Ulysses SP across ViT + text encoder + backbone must not change the result.

    The three modules shard three different things (patch rows, token ids, the
    packed sequence), so this is the composition-level check that each module's
    slice/gather lands on its own group. SP=4 also puts the LLM's KV heads below
    the Ulysses size, covering the GQA head-repeat path SP=2 never reaches.
    """
    out_dir = tmp_path / f"sp_compose{nproc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _sp_worker,
        args=(str(out_dir / "rendezvous"), str(out_dir), nproc, "cpu"),
        nprocs=nproc,
        join=True,
    )

    _assert_sp_parity(out_dir, world=nproc)


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_4)
@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) or runner_device_count() < _CI_ACCELERATOR_4.nproc,
    reason="requires >=4 CUDA or NPU devices",
)
def test_sp_matches_sp1_on_accelerator(tmp_path: Path):
    """Same SP=1 vs SP=4 comparison over real NCCL/HCCL collectives in bfloat16.

    The CPU cases prove the composition; this one proves it over the collective
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
