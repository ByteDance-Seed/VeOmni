"""Running a SeedOmni family straight off its upstream HF checkpoint.

A test-only ``fake_hf_omni`` :class:`OmniHFLayout` cuts a tiny upstream
checkpoint into the in-tree ``fake_module_a`` / ``fake_module_b`` stand-ins:
``encoder.*`` feeds module A, ``decoder.*`` feeds module B, and ``head.weight``
is a tied copy of ``encoder.proj.weight`` that no module stores. The tests
check that the layout routes every HF key onto the module that owns it, that
``OmniModel.from_pretrained`` reads the HF root as if it were a split
checkpoint, and that :func:`save_hf_source_checkpoint` writes the source's own
keys, dtypes and shards back.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PretrainedConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.omni_arguments_types import OmniModuleRuntimeArguments, _validate_hf_view_modules
from veomni.models.checkpoint.weights import _load_state_dict
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
from veomni.models.seed_omni.modules.fake_model.fake_module_b.configuration import FakeModuleBConfig
from veomni.models.seed_omni.utils import hf_layout
from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint
from veomni.models.seed_omni.utils.hf_layout import (
    OMNI_HF_LAYOUT_REGISTRY,
    OmniHFLayout,
    OmniHFModuleLayout,
    read_hf_source,
    resolve_hf_source_device_map,
    resolve_omni_checkpoint_root,
    save_hf_source_checkpoint,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


MODEL_TYPE = "fake_hf_omni"
FAKE_A = "fake_module_a"
FAKE_B = "fake_module_b"
HIDDEN = 8
FAKE_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "seed_omni" / "fake_model"


def _load_tokenizer(path: str) -> dict:
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(path)}


FAKE_HF_LAYOUT = OmniHFLayout(
    modules={
        FAKE_A: OmniHFModuleLayout(
            key_prefixes=(("encoder.", ""), ("head.", "head.")),
            build_config=lambda c: FakeModuleAConfig(hidden_size=c.hidden_size),
            build_assets=_load_tokenizer,
        ),
        FAKE_B: OmniHFModuleLayout(
            key_prefixes=(("decoder.", ""),),
            build_config=lambda c: FakeModuleBConfig(hidden_size=c.hidden_size),
        ),
    },
    tied_source_keys={"head.weight": "encoder.proj.weight"},
    graph_dir="configs/seed_omni/fake_model",
    training_graphs="graph_train.yaml",
    generation_graphs="graph_infer.yaml",
    infer_type="infer_gen",
    load_hf_config=PretrainedConfig.from_pretrained,
)


@pytest.fixture(autouse=True)
def _register_fake_layout():
    # ``Registry.__setitem__`` is a local override that ``del`` removes again.
    OMNI_HF_LAYOUT_REGISTRY[MODEL_TYPE] = lambda source_path=None: FAKE_HF_LAYOUT
    yield
    del OMNI_HF_LAYOUT_REGISTRY[MODEL_TYPE]


def _source_tensors() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    encoder = torch.randn(HIDDEN, HIDDEN).to(torch.bfloat16)
    return {
        "encoder.proj.weight": encoder,
        "decoder.proj.weight": torch.randn(HIDDEN, HIDDEN).to(torch.bfloat16),
        "head.weight": encoder.clone(),
    }


def _save_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "a": 4, "b": 5}
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    ).save_pretrained(path)


def _write_hf_checkpoint(root: Path, *, sharded: bool) -> dict[str, torch.Tensor]:
    root.mkdir(parents=True)
    config = {"model_type": MODEL_TYPE, "hidden_size": HIDDEN, "dtype": "bfloat16"}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _save_tokenizer(root)
    tensors = _source_tensors()
    if not sharded:
        save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
        return tensors
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    weight_map = {key: shards[0] if key.startswith("encoder.") else shards[1] for key in tensors}
    for shard in shards:
        save_file(
            {key: value for key, value in tensors.items() if weight_map[key] == shard},
            root / shard,
            metadata={"format": "pt"},
        )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    return tensors


@pytest.fixture
def hf_root(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    root = tmp_path / "fake-hf"
    return root, _write_hf_checkpoint(root, sharded=False)


def _read_safetensors(root: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt") as f:
            weight_map = dict.fromkeys(f.keys(), "model.safetensors")
    tensors = {}
    for filename in sorted(set(weight_map.values())):
        with safe_open(root / filename, framework="pt") as f:
            tensors.update({key: f.get_tensor(key) for key in f.keys()})
    return tensors, weight_map


def test_layout_routes_each_key_to_the_longest_claiming_prefix_and_back():
    layout = OmniHFLayout(
        modules={
            "backbone": OmniHFModuleLayout(key_prefixes=(("model.", "decoder."),), build_config=lambda c: c),
            "embed": OmniHFModuleLayout(
                key_prefixes=(("model.embed_tokens.", "embed_tokens."), ("lm_head.", "lm_head.")),
                build_config=lambda c: c,
            ),
        }
    )
    assert layout.route("model.embed_tokens.weight") == ("embed", "embed_tokens.weight")
    assert layout.route("lm_head.weight") == ("embed", "lm_head.weight")
    assert layout.route("model.layers.1.mlp.weight") == ("backbone", "decoder.layers.1.mlp.weight")
    assert layout.route("visual.patch_embed.weight") is None

    assert layout.source_key("embed", "embed_tokens.weight") == "model.embed_tokens.weight"
    assert layout.source_key("backbone", "decoder.norm.weight") == "model.norm.weight"
    # Routes back to the embedding module, so it is not a backbone key.
    assert layout.source_key("backbone", "decoder.embed_tokens.weight") is None
    assert layout.source_key("backbone", "rotary_emb.inv_freq") is None


def test_layout_rejects_a_source_prefix_claimed_by_two_modules():
    layout = OmniHFLayout(
        modules={
            "a": OmniHFModuleLayout(key_prefixes=(("model.", "a."),), build_config=lambda c: c),
            "b": OmniHFModuleLayout(key_prefixes=(("model.", "b."),), build_config=lambda c: c),
        }
    )
    with pytest.raises(ValueError, match="more than one module"):
        layout.route("model.x")


def test_route_picks_the_longest_prefix_among_many_full_key_renames():
    """A layout may rename thousands of keys one by one beside a few prefixes;
    the length-indexed lookup must keep longest-prefix semantics."""
    full_keys = tuple((f"layers.{i}.attn.wq.weight", f"model.layers.{i}.self_attn.wq.weight") for i in range(50))
    layout = OmniHFLayout(
        modules={
            "llm": OmniHFModuleLayout(
                key_prefixes=full_keys + (("layers.3.ffn.experts.", "model.layers.3.mlp.experts."),),
                build_config=lambda c: c,
            ),
            "vision": OmniHFModuleLayout(key_prefixes=(("image_", "image_"), ("", "rest.")), build_config=lambda c: c),
        }
    )
    assert layout.route("layers.7.attn.wq.weight") == ("llm", "model.layers.7.self_attn.wq.weight")
    assert layout.route("layers.3.ffn.experts.5.w1.weight") == ("llm", "model.layers.3.mlp.experts.5.w1.weight")
    assert layout.route("image_newline") == ("vision", "image_newline")
    # Only the empty prefix claims these: a full key plus a suffix, and an unrelated key.
    assert layout.route("layers.7.attn.wq.weight.extra") == ("llm", "model.layers.7.self_attn.wq.weight.extra")
    assert layout.route("norm.weight") == ("vision", "rest.norm.weight")
    assert layout.source_key("llm", "model.layers.7.self_attn.wq.weight") == "layers.7.attn.wq.weight"
    assert layout.source_key("vision", "rest.norm.weight") == "norm.weight"
    assert layout.source_key("vision", "rest.layers.7.attn.wq.weight") is None


def _with_module_hooks(monkeypatch, module: str, **hooks) -> None:
    """Swap the fake layout for one whose ``module`` declares ``hooks``."""
    patched = dataclasses.replace(FAKE_HF_LAYOUT.modules[module], **hooks)
    layout = dataclasses.replace(FAKE_HF_LAYOUT, modules={**FAKE_HF_LAYOUT.modules, module: patched})
    monkeypatch.setattr(hf_layout, "get_hf_layout", lambda model_type, source_path=None: layout)


def test_the_layout_factory_is_handed_the_checkpoint_directory(hf_root, monkeypatch):
    root, _ = hf_root
    seen = []

    def factory(source_path=None):
        seen.append(source_path)
        return FAKE_HF_LAYOUT

    monkeypatch.setitem(OMNI_HF_LAYOUT_REGISTRY, MODEL_TYPE, factory)
    hf_layout.get_hf_layout.cache_clear()
    OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    assert os.path.realpath(root) in seen
    hf_layout.get_hf_layout.cache_clear()


def test_generated_tensors_fill_what_the_checkpoint_does_not_store(hf_root, monkeypatch):
    root, tensors = hf_root
    del tensors["decoder.proj.weight"]
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    seen = []

    def generated(source_path, config):
        seen.append((source_path, config.hidden_size))
        return {"proj.weight": torch.full((config.hidden_size, config.hidden_size), 3.0)}

    _with_module_hooks(monkeypatch, FAKE_B, generated_tensors=generated)
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    assert seen == [(os.path.realpath(root), HIDDEN)]
    assert (model.modules_dict[FAKE_B].proj.weight == 3.0).all()


def test_export_state_shapes_what_the_export_writes(hf_root, tmp_path, monkeypatch):
    root, tensors = hf_root

    def export_state(module, state):
        return {"proj.weight": state["proj.weight"] + 2.0}

    _with_module_hooks(monkeypatch, FAKE_A, export_state=export_state)
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    out = tmp_path / "export"
    save_hf_source_checkpoint(model.config._hf_source, {FAKE_A: (model.modules_dict[FAKE_A], None)}, str(out))
    exported, _ = _read_safetensors(out)
    assert torch.equal(exported["encoder.proj.weight"], tensors["encoder.proj.weight"] + 2.0)
    assert torch.equal(exported["head.weight"], exported["encoder.proj.weight"])


def test_export_refuses_a_cast_that_would_drop_quantization(hf_root, tmp_path, monkeypatch):
    root, _ = hf_root

    def export_state(module, state):
        return {"proj.weight": state["proj.weight"].to(torch.float8_e4m3fn)}

    _with_module_hooks(monkeypatch, FAKE_A, export_state=export_state)
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    with pytest.raises(TypeError, match="must declare export_state"):
        save_hf_source_checkpoint(
            model.config._hf_source, {FAKE_A: (model.modules_dict[FAKE_A], None)}, str(tmp_path / "export")
        )


def test_export_dtype_refusal_propagates_across_ranks(hf_root, tmp_path, monkeypatch):
    """When another rank's ``export_state`` slice hits a forbidden cast, every rank raises together."""
    import veomni.models.seed_omni.utils.hf_layout as hf_layout_mod

    root, _ = hf_root
    calls = []

    def fake_union(items):
        items = list(items)
        calls.append(items)
        if len(calls) == 1:
            return ["proj.weight"]
        if len(calls) == 2:
            return items
        return ["save_hf_source_checkpoint: peer rank hit a forbidden cast; must declare export_state."]

    monkeypatch.setattr(hf_layout_mod, "_all_ranks_union", fake_union)
    _with_module_hooks(monkeypatch, FAKE_A, export_state=lambda module, state: {})
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    with pytest.raises(TypeError, match="peer rank hit a forbidden cast"):
        save_hf_source_checkpoint(
            model.config._hf_source, {FAKE_A: (model.modules_dict[FAKE_A], None)}, str(tmp_path / "export")
        )
    assert len(calls) == 3


def test_an_hf_root_resolves_to_a_weight_free_split_view(hf_root):
    root, _ = hf_root
    view = resolve_omni_checkpoint_root(root)
    assert view != str(root)
    assert resolve_omni_checkpoint_root(root) == view
    assert resolve_omni_checkpoint_root(view) == view

    source = read_hf_source(view)
    assert source.model_type == MODEL_TYPE and source.path == os.path.realpath(root)
    assert json.loads(Path(view, "config.json").read_text(encoding="utf-8"))["model_type"] == "omni"
    for name in (FAKE_A, FAKE_B):
        assert Path(view, name, "config.json").is_file()
    assert Path(view, FAKE_A, "tokenizer.json").is_file()
    assert not list(Path(view).rglob("*.safetensors"))
    assert set(OmniConfig.from_pretrained(str(root)).module_names) == {FAKE_A, FAKE_B}


def test_split_roots_pass_through_and_unregistered_hf_types_fail(tmp_path):
    omni = tmp_path / "omni"
    omni.mkdir()
    (omni / "config.json").write_text(json.dumps({"model_type": "omni"}), encoding="utf-8")
    assert resolve_omni_checkpoint_root(omni) == omni
    assert read_hf_source(omni) is None

    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "not_a_registered_family"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no SeedOmni HF layout is registered"):
        resolve_omni_checkpoint_root(other)


def test_omni_model_loads_every_module_weight_from_the_hf_root(hf_root):
    root, tensors = hf_root
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    layout = model.config._hf_source.layout

    compared = set()
    for name, module in model.modules_dict.items():
        for key, value in module.state_dict().items():
            source_key = layout.source_key(name, key)
            if source_key in tensors:
                assert torch.equal(value, tensors[source_key]), (name, key)
                compared.add(source_key)
    assert compared == set(tensors) - {"head.weight"}
    assert "infer_gen" in model.config.generation_graphs


def test_a_wrapped_module_reads_only_its_own_keys(hf_root):
    root, _ = hf_root
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    module_a = model.modules_dict[FAKE_A]

    # A LoRA wrapper prefixes every name; the module's own keys must still be read.
    wrapper = torch.nn.Module()
    wrapper.base_model = module_a
    converter = module_a._create_checkpoint_tensor_converter(wrapper)
    assert not converter.should_skip_without_loading("encoder.proj.weight")
    assert converter.should_skip_without_loading("decoder.proj.weight")
    # A tied duplicate the module does not store is skipped, not read.
    assert converter.should_skip_without_loading("head.weight")


def test_auto_dtype_is_the_checkpoint_dtype(hf_root):
    root, _ = hf_root
    model = OmniModel.from_pretrained(str(root), torch_dtype="auto", device_map="cpu")
    for module in model.modules_dict.values():
        assert {param.dtype for param in module.parameters()} == {torch.bfloat16}


def test_device_map_takes_its_from_pretrained_meaning(hf_root):
    root, _ = hf_root
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    module_b = model.modules_dict[FAKE_B]
    assert resolve_hf_source_device_map(module_b, None) == {"": "cpu"}
    assert resolve_hf_source_device_map(module_b, 1) == {"": 1}
    assert resolve_hf_source_device_map(module_b, {"proj": "cpu"}) == {"proj": "cpu"}
    for strategy in ("auto", "sequential"):
        plan = resolve_hf_source_device_map(module_b, strategy, max_memory={"cpu": "1GiB"})
        assert set(plan.values()) == {"cpu"}


def test_a_multi_device_map_loads_on_cpu_then_dispatches(hf_root, monkeypatch):
    import accelerate

    root, tensors = hf_root
    dispatched = []

    def fake_dispatch(model, device_map, **kwargs):
        assert {param.device.type for param in model.parameters()} == {"cpu"}
        dispatched.append((model, device_map))
        return model

    monkeypatch.setattr(accelerate, "dispatch_model", fake_dispatch)
    device_map = {"first": "cpu", "second": "cuda:0"}
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map=device_map)

    assert [plan for _, plan in dispatched] == [device_map] * len(model.modules_dict)
    layout = model.config._hf_source.layout
    for name, module in model.modules_dict.items():
        for key, value in module.state_dict().items():
            assert torch.equal(value, tensors[layout.source_key(name, key)]), (name, key)

    with pytest.raises(ValueError, match="cannot offload to disk"):
        OmniModel.from_pretrained(str(root), device_map={"first": "cpu", "second": "disk"})


@pytest.mark.parametrize("sharded", [False, True], ids=["single_file", "sharded"])
def test_export_writes_back_the_source_layout(tmp_path, sharded):
    root = tmp_path / "fake-hf"
    tensors = _write_hf_checkpoint(root, sharded=sharded)
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    module_a = model.modules_dict[FAKE_A]
    with torch.no_grad():
        module_a.proj.weight.add_(1.0)

    out = tmp_path / "export"
    save_hf_source_checkpoint(model.config._hf_source, {FAKE_A: (module_a, None)}, str(out))

    exported, weight_map = _read_safetensors(root=out)
    _, source_map = _read_safetensors(root=root)
    assert weight_map == source_map
    assert sorted(path.name for path in out.iterdir()) == sorted(path.name for path in root.iterdir())
    for key, value in tensors.items():
        assert exported[key].dtype == value.dtype, key
    trained = tensors["encoder.proj.weight"] + 1.0
    assert torch.equal(exported["encoder.proj.weight"], trained)
    # The tied duplicate follows its trained original, not the stale source copy.
    assert torch.equal(exported["head.weight"], trained)
    # Module B was not handed in, so its weights are carried from the source.
    assert torch.equal(exported["decoder.proj.weight"], tensors["decoder.proj.weight"])


def test_export_refuses_weights_the_source_layout_cannot_hold(hf_root, tmp_path):
    root, _ = hf_root
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    source = model.config._hf_source
    module_a = model.modules_dict[FAKE_A]

    module_a.extra = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match=r"Keys with no source key: \['extra'\]"):
        save_hf_source_checkpoint(source, {FAKE_A: (module_a, None)}, str(tmp_path / "unmapped"))
    del module_a.extra

    module_a.proj.weight = torch.nn.Parameter(torch.zeros(HIDDEN + 1, HIDDEN, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="shape differs from the source"):
        save_hf_source_checkpoint(source, {FAKE_A: (module_a, None)}, str(tmp_path / "reshaped"))


def test_export_writes_a_tied_alias_the_source_omits_through_its_original(hf_root, tmp_path):
    root, tensors = hf_root
    # A ``tie_word_embeddings`` checkpoint stores no copy of the tied head.
    stored = {key: value for key, value in tensors.items() if key != "head.weight"}
    save_file(stored, root / "model.safetensors", metadata={"format": "pt"})
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    module_a = model.modules_dict[FAKE_A]
    module_a.head = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    module_a.head.weight = module_a.proj.weight

    out = tmp_path / "export"
    save_hf_source_checkpoint(model.config._hf_source, {FAKE_A: (module_a, None)}, str(out))

    exported, _ = _read_safetensors(root=out)
    assert set(exported) == set(stored)
    assert torch.equal(exported["encoder.proj.weight"], stored["encoder.proj.weight"])


def test_offline_convert_matches_the_runtime_load(hf_root, tmp_path):
    root, _ = hf_root
    out = tmp_path / "split"
    convert_checkpoint(str(root), str(out))
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["model_type"] == "omni"
    assert read_hf_source(out) is None

    split = OmniModel.from_pretrained(str(out), torch_dtype=torch.bfloat16, device_map="cpu")
    runtime = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    for name in (FAKE_A, FAKE_B):
        expected = runtime.modules_dict[name].state_dict()
        actual = split.modules_dict[name].state_dict()
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.equal(expected[key], actual[key]), (name, key)


def test_state_dict_loader_skips_declared_keys_without_reading_them(hf_root):
    root, tensors = hf_root
    skip = lambda key: key.startswith("decoder.")  # noqa: E731
    loaded = {}
    for iterator in _load_state_dict(str(root), skip_key=skip):
        for key, value in iterator:
            loaded[key] = value
    assert set(loaded) == {key for key in tensors if not skip(key)}


def test_hf_view_modules_must_all_come_from_the_layout(hf_root):
    root, _ = hf_root
    view = resolve_omni_checkpoint_root(root)
    ok = {name: OmniModuleRuntimeArguments(model_path=os.path.join(view, name)) for name in (FAKE_A, FAKE_B)}
    _validate_hf_view_modules(view, ok)

    with pytest.raises(ValueError, match=f"not part of the `{MODEL_TYPE}` HF layout"):
        _validate_hf_view_modules(view, {**ok, "other_module": OmniModuleRuntimeArguments(model_path="/x")})
    with pytest.raises(ValueError, match="sets its own model_path"):
        _validate_hf_view_modules(view, {**ok, FAKE_B: OmniModuleRuntimeArguments(model_path="/elsewhere/b")})


def test_from_pretrained_rejects_a_module_read_from_elsewhere(hf_root, tmp_path):
    root, _ = hf_root
    split = tmp_path / "split"
    convert_checkpoint(str(root), str(split))

    overlay = {f"{FAKE_B}.model_path": str(split / FAKE_B)}
    with pytest.raises(ValueError, match="view of the HuggingFace"):
        OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu", **overlay)


def test_build_runtime_args_keeps_model_path_and_sets_seed_omni_model_path(hf_root):
    from veomni.arguments.omni_arguments_types import (
        OmniArguments,
        OmniDataArguments,
        OmniInferArguments,
        OmniModelRuntimeArguments,
        build_omni_model_runtime_args,
    )

    root, _ = hf_root
    args = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=str(root),
            model_config={
                "modules": str(FAKE_CONFIGS / "train" / "modules_train.yaml"),
                "train_graph": str(FAKE_CONFIGS / "train" / "graph_train.yaml"),
                "infer_graph": {"infer_gen": str(FAKE_CONFIGS / "infer" / "graph_infer_gen.yaml")},
            },
        ),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )
    runtime = build_omni_model_runtime_args(args)

    assert args.model.model_path == str(root)
    assert runtime.model_path == str(root)
    assert args.model.seed_omni_model_path == runtime.seed_omni_model_path
    assert runtime.seed_omni_model_path != str(root)
    assert os.path.isfile(os.path.join(runtime.seed_omni_model_path, "hf_source.json"))
    assert runtime.resolved_model_path == runtime.seed_omni_model_path
    assert all(module.model_path.startswith(runtime.seed_omni_model_path) for module in runtime.modules.values())


def test_a_training_export_needs_safetensors_in_the_source(tmp_path):
    from veomni.arguments.omni_arguments_types import _check_hf_source_exportable

    root = tmp_path / "fake-hf-bin"
    tensors = _write_hf_checkpoint(root, sharded=False)
    (root / "model.safetensors").unlink()
    torch.save(tensors, root / "pytorch_model.bin")
    view = resolve_omni_checkpoint_root(root)

    with pytest.raises(ValueError, match="needs safetensors weights"):
        _check_hf_source_exportable(view)


def test_an_offline_embedding_run_skips_the_export_check(tmp_path, monkeypatch):
    from veomni.arguments import omni_arguments_types
    from veomni.arguments.omni_arguments_types import (
        OmniArguments,
        OmniDataArguments,
        OmniInferArguments,
        OmniModelRuntimeArguments,
        build_omni_model_runtime_args,
    )

    root = tmp_path / "fake-hf"
    _write_hf_checkpoint(root, sharded=False)
    checked = []
    monkeypatch.setattr(omni_arguments_types, "_check_hf_source_exportable", checked.append)
    args = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=str(root),
            model_config={
                "modules": str(FAKE_CONFIGS / "train" / "modules_train.yaml"),
                "train_graph": str(FAKE_CONFIGS / "train" / "graph_train.yaml"),
            },
        ),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )
    assert args.train.checkpoint.save_hf_weights

    args.train.training_task = "offline_embedding"
    build_omni_model_runtime_args(args)
    assert checked == []

    args.train.training_task = "online_training"
    build_omni_model_runtime_args(args)
    assert len(checked) == 1


def test_large_carried_tensors_are_sliced_across_ranks_and_consolidated_on_every_rank(hf_root, tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.distributed.checkpoint._consolidate_hf_safetensors as consolidate_mod
    from torch.distributed.tensor import DTensor, Shard
    from torch.testing._internal.distributed.fake_pg import FakeStore

    import veomni.models.seed_omni.utils.hf_layout as hf_layout_mod
    import veomni.utils.save_safetensor_utils as save_utils_mod

    root, tensors = hf_root
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    source = model.config._hf_source
    module_a = model.modules_dict[FAKE_A]

    # Lower the threshold so the 4x4 carried tensor ``decoder.proj.weight`` is sliced across ranks.
    monkeypatch.setattr(hf_layout_mod, "_SHARDED_CARRY_BYTES", 8)
    group = object()
    monkeypatch.setattr(save_utils_mod, "_consolidation_process_group", lambda: group)

    saved_states = []
    saved_groups = []
    writers = []

    class FakeWriter:
        def __init__(self, **kwargs):
            writers.append(kwargs)

    monkeypatch.setattr("torch.distributed.checkpoint.HuggingFaceStorageWriter", FakeWriter)
    monkeypatch.setattr(
        dcp,
        "save",
        lambda state_dict, storage_writer, process_group=None: (
            saved_states.append(state_dict),
            saved_groups.append(process_group),
        ),
    )
    on_every_rank = MagicMock()
    monkeypatch.setattr(consolidate_mod, "consolidate_safetensors_files_on_every_rank", on_every_rank)
    monkeypatch.setattr(hf_layout_mod, "_restore_single_file_name", lambda path: None)
    monkeypatch.setattr(dist, "barrier", lambda *a, **kw: None)

    parallel_state = MagicMock(dp_mode="ddp")
    for rank in range(3):
        dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=3)
        try:
            save_hf_source_checkpoint(source, {FAKE_A: (module_a, parallel_state)}, str(tmp_path / "export"))
        finally:
            dist.destroy_process_group()

    assert all(not w["enable_consolidation"] for w in writers)
    assert saved_groups == [group, group, group]
    assert [call.kwargs["process_group"] for call in on_every_rank.call_args_list] == [group, group, group]

    slices = []
    for state in saved_states:
        carried = state["decoder.proj.weight"]
        assert isinstance(carried, DTensor)
        assert carried.placements == (Shard(0),)
        assert tuple(carried.shape) == tuple(tensors["decoder.proj.weight"].shape)
        slices.append(carried._local_tensor)
    # 8 rows across 3 ranks -> ceil(8/3)=3 rows on ranks 0 and 1, 2 rows on rank 2.
    assert [s.shape[0] for s in slices] == [3, 3, 2]
    assert torch.equal(torch.cat(slices, dim=0), tensors["decoder.proj.weight"])
