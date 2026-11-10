"""The Qwen3 visual-instruction-tuning recipe through its real entry points.

``tests/seed_omni/modules/qwen3/test_qwen3_visual_instruction_tuning.py``
builds the composed graph in-process, which pins the wiring but not the launcher:
config resolution, checkpoint conversion, the data pipeline and the per-module
optimizers are all bypassed there. This file takes the user-facing path instead —
``scripts/seed_omni/convert_model.py`` on two upstream checkpoints, then
``tasks/omni/train_omni.py``'s trainer on the recipe's YAML — and checks the two
things #1072 asks for that only an end-to-end run can show: the toy recipe
converges, and SP=1 and SP=4 agree on the loss.

Toy checkpoints are synthesized from the same tiny configs the module tests use,
so the run fits on four 24 GiB cards and still walks every real code path
(conversion, graph resolution, FSDP2/SP wrapping, per-module optimizers,
``configure_optimizer``).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch
import yaml

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.launch_utils import run_bounded_process_group
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


_CI_LAUNCH = LaunchSpec(nproc=4)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "tests/train_scripts/omni_qwen3_visual_instruction_tuning_runner.py"
RECIPE = REPO_ROOT / "configs" / "seed_omni" / "Qwen" / "qwen3" / "visual_instruction_tuning"
STEPS = 30
# The measured curve drops 1.2-3.9% per step on this config, so 0.98 leaves the
# check meaningful while absorbing bf16 reduction-order differences.
CONVERGENCE_RATIO = 0.98
SP_LOSS_RTOL = 5e-3

_DEVICE_COUNT = runner_device_count() if (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) else 0

pytestmark = pytest.mark.launch_plan(provider=_CI_LAUNCH)

requires_four_accelerators = pytest.mark.skipif(
    _DEVICE_COUNT < _CI_LAUNCH.nproc,
    reason=f"needs {_CI_LAUNCH.nproc} accelerators for the SP=1 vs SP=4 comparison",
)

VOCAB = 64
HIDDEN = 16
# Both modules of the recipe have to agree on the token ids: the text encoder's
# image mode resolves the vision rows from its own tokenizer, and the VLM
# checkpoint's config points at the same ids so preprocessing agrees.
_SPECIAL_TOKENS = {
    "<unk>": 0,
    "<pad>": 1,
    "<s>": 2,
    "<|im_start|>": 3,
    "<|im_end|>": 4,
    "<|vision_start|>": 5,
    "<|vision_end|>": 6,
    "<|image_pad|>": 7,
}

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


def _save_tokenizer(root: Path) -> None:
    """A tiny fast tokenizer whose special ids match the two checkpoints' configs."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    vocab = dict(_SPECIAL_TOKENS, **{f"t{i}": len(_SPECIAL_TOKENS) + i for i in range(VOCAB - len(_SPECIAL_TOKENS))})
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"],
    ).save_pretrained(root)


def _write_qwen3_source(root: Path) -> None:
    """An upstream Qwen3 checkpoint (text encoder + backbone in one file)."""
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    root.mkdir(parents=True)
    config = Qwen3Config(**_TEXT_CONFIG, tie_word_embeddings=True, dtype="bfloat16")
    config.save_pretrained(root)
    _save_tokenizer(root)
    torch.manual_seed(0)
    tensors = {
        key: value.detach().clone().contiguous()
        for key, value in Qwen3ForCausalLM(config).to(torch.bfloat16).state_dict().items()
    }
    # Upstream keeps the tied head on disk beside the embedding.
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})


def _write_qwen3vl_source(root: Path) -> None:
    """An upstream Qwen3-VL checkpoint — the recipe takes only its vision tower."""
    from safetensors.torch import save_file
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLVideoProcessor,
    )
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    root.mkdir(parents=True)
    config = Qwen3VLConfig(
        vision_config=Qwen3VLVisionConfig(**_VISION_CONFIG),
        text_config=dict(_TEXT_CONFIG),
        tie_word_embeddings=True,
        image_token_id=_SPECIAL_TOKENS["<|image_pad|>"],
        video_token_id=_SPECIAL_TOKENS["<|image_pad|>"],
        vision_start_token_id=_SPECIAL_TOKENS["<|vision_start|>"],
        vision_end_token_id=_SPECIAL_TOKENS["<|vision_end|>"],
    )
    config.save_pretrained(root)
    _save_tokenizer(root)
    Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=16 * 16 * 4,
        max_pixels=16 * 16 * 64,
        size={"shortest_edge": 64, "longest_edge": 128},
    ).save_pretrained(root)
    Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2).save_pretrained(root)
    torch.manual_seed(0)
    tensors = {
        key: value.detach().clone().contiguous()
        for key, value in Qwen3VLForConditionalGeneration(config).to(torch.bfloat16).state_dict().items()
    }
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})


def _write_png(path: Path, rgb: tuple[int, int, int], size: int = 32) -> None:
    """A solid-colour PNG, so the sample carries a real decoded image (no PIL round-trip deps)."""
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        """One PNG chunk: length, type, payload, CRC over type+payload."""
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _run(command: list[str], *, env: dict[str, str], timeout: float) -> None:
    """Run ``command`` to completion, failing the test with its tail on a non-zero exit."""
    run_bounded_process_group(command, env=env, timeout=timeout)


def _convert(source: Path, output: Path, env: dict[str, str]) -> None:
    """Split ``source`` through the user-facing converter."""
    _run(
        [
            sys.executable,
            "scripts/seed_omni/convert_model.py",
            "--model_path",
            str(source),
            "--output_dir",
            str(output),
        ],
        env=env,
        timeout=300.0,
    )


def _train(root: Path, *, ulysses_size: int, env: dict[str, str]) -> list[float]:
    """Train the recipe once at ``ulysses_size`` and return the per-step losses."""
    run_dir = root / f"sp{ulysses_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_env = {**env, "OMNI_VIT_RUNNER_DIR": str(run_dir)}
    _run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={_CI_LAUNCH.nproc}",
            "--standalone",
            RUNNER,
            str(root / "base.yaml"),
            f"--model.accelerator.ulysses_size={ulysses_size}",
            f"--train.checkpoint.output_dir={run_dir / 'out'}",
        ],
        env=run_env,
        timeout=1800.0,
    )
    return json.loads((run_dir / "losses.json").read_text(encoding="utf-8"))["losses"]


@pytest.fixture(scope="module")
def recipe_root(tmp_path_factory) -> Path:
    """Two upstream toy checkpoints, their splits, and a launcher YAML for this run.

    Built once for both cases: converting twice would double the file count for no
    extra coverage, and the two SP runs must start from the same weights anyway —
    that common start is what makes the loss comparison meaningful.
    """
    root = tmp_path_factory.mktemp("qwen3_vit_recipe")
    env = {**os.environ, "NCCL_DEBUG": "OFF"}
    _write_qwen3_source(root / "Qwen3-tiny")
    _write_qwen3vl_source(root / "Qwen3VL-tiny")
    _convert(root / "Qwen3-tiny", root / "Qwen3-split", env)
    _convert(root / "Qwen3VL-tiny", root / "Qwen3VL-split", env)

    # A caption dataset: one image turn plus a text answer, in the ShareGPT4V
    # layout the recipe's `data.yaml` names.
    image = root / "cat.png"
    _write_png(image, (200, 80, 40))
    rows = [
        {
            "source_name": "sharegpt4v_cap_100k",
            "conversations": [{"from": "human", "value": "<image>\ndescribe"}, {"from": "gpt", "value": "a cat"}],
            "images": [str(image)],
        }
        for _ in range(STEPS * _CI_LAUNCH.nproc * 2)
    ]
    (root / "data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    data_yaml = {
        "sources": [str(root / "data.jsonl")],
        "names": ["sharegpt4v_cap_100k"],
        "schedule": [{"schedule_type": "const", "weights": [1.0]}],
        "level": "token",
        "stopping_strategy": "all_exhausted",
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    # The recipe's own modules/graphs, with only the two checkpoint paths and the
    # toy sizes overridden — the overrides that make it a vision recipe stay as
    # written, since they are what this test is about.
    modules = yaml.safe_load((RECIPE / "modules_train.yaml").read_text(encoding="utf-8"))
    modules["qwen3vl_vision"]["model_path"] = str(root / "Qwen3VL-split" / "qwen3vl_vision")
    modules["qwen3vl_vision"]["model_config"]["out_hidden_size"] = HIDDEN
    modules["qwen3_text_encoder"]["model_config"]["enable_image"] = True
    (root / "modules.yaml").write_text(yaml.safe_dump(modules, sort_keys=False), encoding="utf-8")

    base = {
        "model": {
            "model_path": str(root / "Qwen3-split"),
            "model_config": {
                "modules": str(root / "modules.yaml"),
                "train_graph": str(RECIPE / "graph_train.yaml"),
                "infer_graph": {"understanding": str(RECIPE / "graph_infer.yaml")},
            },
            "ops_implementation": {"attn_implementation": "veomni_sdpa"},
            "accelerator": {
                "init_device": "meta",
                "gradient_checkpointing": {"enable": False},
                "fsdp_config": {"fsdp_mode": "fsdp2", "mixed_precision": {"enable": False}},
            },
            "optimizer": {
                "type": "adamw",
                "lr": 3.0e-2,
                "lr_warmup_ratio": 0.0,
                "lr_decay_style": "constant",
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
            },
        },
        "data": {
            "data_type": "seedomni",
            "max_seq_len": 512,
            "train_path": str(root / "data.yaml"),
            "datasets_type": "iterable",
            "multisource_datasets_type": "veomni_weighted_multisource",
            "train_sample": 1_000_000,
            "dataloader": {"type": "native", "num_workers": 0, "drop_last": True},
        },
        "train": {
            "seed": 42,
            "num_train_epochs": 1,
            "global_batch_size": _CI_LAUNCH.nproc * 2,
            "micro_batch_size": 1,
            "dyn_bsz": False,
            "checkpoint": {"output_dir": str(root / "out"), "save_steps": 10_000},
            "wandb": {"enable": False},
        },
    }
    (root / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return root


@requires_four_accelerators
def test_the_toy_recipe_converges_through_the_training_entry_point(recipe_root: Path):
    """#1072 item 1, on the real launcher rather than the in-process graph.

    A run that only "works" — a frozen head, a detached merger, a vision prefix the
    LLM never reads — produces a flat curve from a valid forward, so the assertion
    is that the loss comes down. 30 steps at this size is enough to see it and
    cheap enough for the GPU e2e suite.
    """
    losses = _train(recipe_root, ulysses_size=1, env={**os.environ, "NCCL_DEBUG": "OFF"})

    assert len(losses) == STEPS, len(losses)
    assert all(torch.isfinite(torch.tensor(losses))), losses
    head = sum(losses[:3]) / 3
    tail = sum(losses[-3:]) / 3
    assert tail < head * CONVERGENCE_RATIO, (head, tail, losses)


@requires_four_accelerators
def test_sp4_matches_sp1_through_the_training_entry_point(recipe_root: Path):
    """#1072 item 2: the same recipe at SP=1 and SP=4 must see the same loss.

    This is the shape that validated the composition invariant — uniform Ulysses SP
    across the ViT, the text encoder and the backbone — and the one where a
    wrong-ranks all-to-all shows up as a drifting loss rather than a crash.

    Both runs read the checkpoint and data the fixture wrote, so a difference is
    the parallelism and nothing else. `bf16` reduction order is not bit-exact
    across SP sizes, hence the relative tolerance the per-family alignment tests
    use; a genuine all-to-all or gather bug moves the loss by far more.
    """
    sp1 = _train(recipe_root, ulysses_size=1, env={**os.environ, "NCCL_DEBUG": "OFF"})
    sp4 = _train(recipe_root, ulysses_size=4, env={**os.environ, "NCCL_DEBUG": "OFF"})

    assert len(sp1) == len(sp4) == STEPS, (len(sp1), len(sp4))
    for step, (one, four) in enumerate(zip(sp1, sp4)):
        assert four == pytest.approx(one, rel=SP_LOSS_RTOL), (step, one, four, sp1, sp4)


@pytest.fixture(scope="module", autouse=True)
def _cleanup(recipe_root: Path):
    """Drop the checkpoints and run directories once both cases finish."""
    yield
    shutil.rmtree(recipe_root, ignore_errors=True)
