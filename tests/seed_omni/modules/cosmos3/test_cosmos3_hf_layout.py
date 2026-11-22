"""CPU tests against a miniature release tree, independent of the layout's key map."""

import json
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLWan
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.cosmos3.hf_layout import FLOW, MOT, TEXT, VAE
from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint
from veomni.models.seed_omni.utils.hf_layout import (
    OMNI_HF_LAYOUT_DETECTORS,
    read_hf_source,
    resolve_omni_checkpoint_root,
    save_hf_source_checkpoint,
)

from .test_cosmos3 import TINY_MOT, _modules


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _release(root: Path, indexed=True):
    tr = root / "transformer"
    tr.mkdir(parents=True)
    # Older component-only downloads don't include the pipeline config at the root.
    if indexed:
        _json(root / "config.json", {"model_type": "cosmos3_omni"})
    config = dict(
        TINY_MOT,
        model_type="qwen3_vl_text",
        vocab_size=128,
        latent_patch_size=2,
        patch_latent_dim=8,
        qk_norm_for_diffusion=True,
        dtype="bfloat16",
    )
    config["rope_scaling"] = {"mrope_section": list(config.pop("mrope_section")), "mrope_interleaved": True}
    _json(tr / "config.json", config)
    vocab = {"<unk>": 0, "<|im_end|>": 1, "<|vision_start|>": 2}
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")), unk_token="<unk>", eos_token="<|im_end|>"
    )
    tok.save_pretrained(root / "text_tokenizer")
    modules = _modules(accelerated=False)
    expected = {name: mod.to(torch.bfloat16).state_dict() for name, mod in modules.items()}
    tensors = {k: v for state in expected.values() for k, v in state.items()}
    tensors["blocks.0.unused.weight"] = torch.arange(4)
    # Deliberately use the same basename in both component directories.
    filename = "diffusion_pytorch_model.safetensors"
    if indexed:
        second = {k: tensors.pop(k) for k in list(tensors)[::2]}
        save_file(second, tr / "second.safetensors")
        weight_map = {**dict.fromkeys(second, "second.safetensors"), **dict.fromkeys(tensors, filename)}
        _json(tr / "diffusion_pytorch_model.safetensors.index.json", {"weight_map": weight_map})
    save_file(tensors, tr / filename)
    vae = AutoencoderKLWan(base_dim=4, z_dim=2, dim_mult=[1, 2], num_res_blocks=1, temperal_downsample=[False])
    vae.save_pretrained(root / "vae")
    expected[VAE] = {f"vae.{k}": v for k, v in vae.state_dict().items()}
    return expected


def _load(root):
    return OmniModel.from_pretrained(str(root), dtype="auto", attn_implementation="eager")


def _assert_weights(model, expected):
    assert set(model.modules_dict) == {TEXT, VAE, MOT, FLOW}
    for name, ref in expected.items():
        actual = model.modules_dict[name].state_dict()
        assert actual.keys() == ref.keys()
        for key in ref:
            torch.testing.assert_close(actual[key], ref[key], rtol=0, atol=0)
    text = model.modules_dict[TEXT]
    assert text.config.generation_marker_ids == [1, 2]
    assert text._tokenizer.convert_tokens_to_ids("<|vision_start|>") == 2
    assert model.config.train_type == "online_training"
    assert not model.config.generation_graphs


@pytest.mark.parametrize("indexed", [False, True])
def test_load_and_convert_match_release(tmp_path, indexed):
    root = tmp_path / "release"
    expected = _release(root, indexed)
    assert resolve_omni_checkpoint_root(root) == resolve_omni_checkpoint_root(root / "transformer")
    _assert_weights(_load(root), expected)
    split = tmp_path / "split"
    convert_checkpoint(str(root / "transformer" if indexed else root), str(split), attn_implementation="eager")
    _assert_weights(_load(split), expected)


def test_export_reloads_updated_weights_and_preserves_unused_tensors(tmp_path):
    root = tmp_path / "release"
    expected = _release(root, indexed=True)
    (root / "scheduler").mkdir()
    _json(root / "scheduler/scheduler_config.json", {"shift": 3})
    index_file = root / "model.safetensors.index.json"
    # Some full releases also expose a root index referring into the component directories.
    component_index = root / "transformer/diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(component_index.read_text())["weight_map"]
    _json(index_file, {"weight_map": {k: "transformer/" + v for k, v in weight_map.items()}})
    model = _load(root)
    connector = model.modules_dict[FLOW]
    with torch.no_grad():
        connector.proj_in.bias.add_(1)
    expected[FLOW]["proj_in.bias"] = connector.proj_in.bias.detach().clone()
    out = tmp_path / "exported"
    source = read_hf_source(resolve_omni_checkpoint_root(root / "transformer"))
    save_hf_source_checkpoint(source, {FLOW: (connector, None)}, str(out))
    assert (out / index_file.name).read_bytes() == index_file.read_bytes()
    scheduler = "scheduler/scheduler_config.json"
    assert (out / scheduler).read_bytes() == (root / scheduler).read_bytes()
    index = json.loads((out / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    key = "blocks.0.unused.weight"
    assert torch.equal(load_file(out / "transformer" / index["weight_map"][key])[key], torch.arange(4))
    _assert_weights(_load(out), expected)


def test_missing_tensor_rejected_before_training(tmp_path):
    root = tmp_path / "release"
    _release(root, indexed=False)
    path = root / "transformer/diffusion_pytorch_model.safetensors"
    tensors = load_file(path)
    del tensors["proj_in.bias"]
    save_file(tensors, path)
    with pytest.raises(ValueError, match=r"missing=.*proj_in\.bias"):
        _load(root)


@pytest.mark.parametrize("task", ["online_training", "offline_embedding", "offline_training"])
def test_recipe_resolves_direct_checkpoint(task, tmp_path):
    from veomni.arguments.omni_arguments_types import (
        OmniArguments,
        OmniDataArguments,
        OmniModelRuntimeArguments,
        build_omni_model_runtime_args,
    )

    root = tmp_path / "release"
    _release(root)
    recipe_dir = Path(__file__).resolve().parents[4] / "configs/seed_omni/Cosmos3/cosmos3_nano/train"
    vae_only = task == "offline_embedding"
    graph = "online" if task == "online_training" else task
    args = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=str(root),
            model_config={
                "modules": str(recipe_dir / ("modules_offline_embedding.yaml" if vae_only else "modules_train.yaml")),
                "train_graph": str(recipe_dir / f"graph_{graph}.yaml"),
            },
        ),
        data=OmniDataArguments(train_path=""),
    )
    args.train.training_task = task
    runtime = build_omni_model_runtime_args(args)
    assert read_hf_source(runtime.seed_omni_model_path).model_type == "cosmos3_omni"
    assert set(runtime.modules) == ({VAE} if vae_only else {TEXT, MOT, VAE, FLOW})


def test_cosmos3_detection_is_registered_not_hardcoded():
    assert "cosmos3_omni" in OMNI_HF_LAYOUT_DETECTORS.valid_keys()


def test_plain_qwen_text_is_not_a_cosmos3_release(tmp_path):
    tr = tmp_path / "transformer"
    tr.mkdir()
    (tmp_path / "vae").mkdir()
    _json(tmp_path / "vae/config.json", {})
    _json(tr / "config.json", {"model_type": "qwen3_vl_text"})
    assert resolve_omni_checkpoint_root(tmp_path) == tmp_path
    with pytest.raises(ValueError, match="no SeedOmni HF layout"):
        resolve_omni_checkpoint_root(tr)
