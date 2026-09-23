"""Tests for HF-style OmniModel / OmniConfig checkpoint loading.

The in-tree stand-in is a two-module chain ``fake_module_a → fake_module_b``.
These tests check save layout and load; they do not feed a conversation into
``pre_forward`` / ``post_forward`` / generate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from transformers import PretrainedConfig

from veomni.models.seed_omni.configuration_omni import (
    DEFAULT_GENERATION_GRAPH_FILE,
    DEFAULT_TRAINING_GRAPH_FILE,
    OmniConfig,
    pop_omni_kwargs,
)
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
from veomni.models.seed_omni.modules.fake_model.fake_module_a.modeling import FakeModuleA
from veomni.models.seed_omni.modules.fake_model.fake_module_b.configuration import FakeModuleBConfig
from veomni.models.seed_omni.modules.fake_model.fake_module_b.modeling import FakeModuleB


FAKE_A = "fake_module_a"
FAKE_B = "fake_module_b"
HIDDEN_SIZE = 8


def _chain_edges() -> list[dict]:
    return [{"from": FAKE_A, "to": FAKE_B}, {"from": FAKE_B, "to": "end"}]


def _chain_modules() -> dict:
    return {FAKE_A: {"model_path": FAKE_A}, FAKE_B: {"model_path": FAKE_B}}


def _state_module_ops(module_dir: Path, ops: dict) -> None:
    """Make a module's own ``config.json`` state the kernels it was exported with."""
    config_path = module_dir / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["ops_implementation"] = ops
    config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def _write_module_stub(module_dir: Path, *, model_type: str) -> None:
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": HIDDEN_SIZE}),
        encoding="utf-8",
    )


def _write_omni_checkpoint(root: Path) -> None:
    _write_module_stub(root / FAKE_A, model_type=FAKE_A)
    _write_module_stub(root / FAKE_B, model_type=FAKE_B)

    training_graph = _chain_edges()
    generation_graphs = {
        "infer_gen": {
            "initial": "step",
            "states": {
                "step": {
                    "body": [{"from": FAKE_A, "to": "end"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        },
        "infer_und": {
            "initial": "understand",
            "states": {
                "understand": {
                    "body": [{"from": FAKE_B, "to": "end"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        },
    }

    yaml.safe_dump(
        training_graph,
        (root / DEFAULT_TRAINING_GRAPH_FILE).open("w", encoding="utf-8"),
        sort_keys=False,
    )
    yaml.safe_dump(
        generation_graphs,
        (root / DEFAULT_GENERATION_GRAPH_FILE).open("w", encoding="utf-8"),
        sort_keys=False,
    )

    config = {
        "model_type": "omni",
        "_module_entries": _chain_modules(),
        "infer_type": "infer_gen",
        "generation_kwargs": {"max_new_tokens": 16},
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _minimal_generation_graph(*, module: str = FAKE_A) -> dict:
    return {
        "initial": "run",
        "states": {
            "run": {
                "body": [{"from": module, "to": "end"}],
                "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
            }
        },
    }


def _minimal_generation_graphs(*, module: str = FAKE_A) -> dict:
    return {"infer_gen": _minimal_generation_graph(module=module)}


def _build_omni_model() -> OmniModel:
    config = OmniConfig(
        _module_entries=_chain_modules(),
        training_graphs={"default": _chain_edges()},
        generation_graphs=_minimal_generation_graphs(),
    )
    return OmniModel(
        config,
        {
            FAKE_A: FakeModuleA(FakeModuleAConfig(hidden_size=HIDDEN_SIZE)),
            FAKE_B: FakeModuleB(FakeModuleBConfig(hidden_size=HIDDEN_SIZE)),
        },
    )


def test_omni_config_from_pretrained_hydrates_graph_sidecars(tmp_path):
    _write_omni_checkpoint(tmp_path)

    config = OmniConfig.from_pretrained(tmp_path)

    assert config.training_graph == _chain_edges()
    assert config.infer_types == ["infer_gen", "infer_und"]
    assert config.generation_graph["initial"] == "step"
    assert config.generation_graphs["infer_und"]["initial"] == "understand"
    assert config._module_entries[FAKE_A]["model_path"] == FAKE_A
    assert config.resolve_module_path(tmp_path, FAKE_B) == str(tmp_path / FAKE_B)
    assert isinstance(config._module_configs[FAKE_A], FakeModuleAConfig)
    assert isinstance(config._module_configs[FAKE_B], FakeModuleBConfig)
    assert config._module_configs[FAKE_A].hidden_size == HIDDEN_SIZE


def test_loaded_module_config_applies_slot_overwrites_with_ops_priority(tmp_path):
    """A module config is the file, then the composed model's overwrites.

    ``model_config`` and ``processor_config`` come from the slot. Ops rank
    the file's own lowest, then the root ``ops_implementation``, then the
    slot's ``ops_implementation``.
    """
    _write_omni_checkpoint(tmp_path)
    _state_module_ops(
        tmp_path / FAKE_A,
        {"attn_implementation": "sdpa", "rms_norm_implementation": "torch"},
    )
    root = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    root["ops_implementation"] = {
        "rms_norm_implementation": "eager",
        "attn_implementation": "flash_attention_2",
    }
    root["_module_entries"][FAKE_A] = {
        "model_path": FAKE_A,
        "ops_implementation": {"attn_implementation": "eager"},
        "model_config": {"hidden_size": 16},
        "processor_config": {"packed_preprocess": True},
    }
    (tmp_path / "config.json").write_text(json.dumps(root), encoding="utf-8")

    loaded = OmniConfig.from_pretrained(tmp_path)._module_configs[FAKE_A]

    assert loaded.hidden_size == 16
    assert loaded.processor_config == {"packed_preprocess": True}
    assert loaded.ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "eager",
    }


def test_omni_config_from_pretrained_rejects_unregistered_module_type(tmp_path):
    """Only registered omni modules can be composed; a plain HF type is not a fallback."""
    _write_omni_checkpoint(tmp_path)
    _write_module_stub(tmp_path / FAKE_A, model_type="llama")

    with pytest.raises(KeyError, match="not registered in OMNI_MODEL_REGISTRY"):
        OmniConfig.from_pretrained(tmp_path)


def test_omni_config_infer_type_selects_generation_graph(tmp_path):
    _write_omni_checkpoint(tmp_path)

    config = OmniConfig.from_pretrained(tmp_path)
    assert config.generation_graph["initial"] == "step"

    config.infer_type = "infer_und"
    assert config.generation_graph["initial"] == "understand"

    config.infer_type = "nope"
    with pytest.raises(KeyError, match="Unknown infer_type"):
        _ = config.generation_graph


def test_omni_config_repr_survives_required_init_args(tmp_path):
    """transformers probes `OmniConfig()` for defaults in to_diff_dict/__repr__."""
    _write_omni_checkpoint(tmp_path)
    config = OmniConfig.from_pretrained(tmp_path)
    assert "omni" in repr(config)


def test_omni_config_generation_graph_is_read_only():
    config = OmniConfig(
        _module_entries=_chain_modules(),
        training_graphs={"default": _chain_edges()},
        generation_graphs=_minimal_generation_graphs(),
    )
    with pytest.raises(AttributeError, match="read-only"):
        config.generation_graph = {"initial": "x", "states": {}}


def test_omni_config_without_a_generation_graph_returns_an_empty_map():
    config = OmniConfig(
        _module_entries=_chain_modules(),
        training_graphs={"default": _chain_edges()},
        generation_graphs={},
    )
    assert config.generation_graph == {}


def test_omni_model_from_pretrained_loads_the_fake_chain(tmp_path):
    model = _build_omni_model()
    with torch.no_grad():
        model.get_module(FAKE_A).proj.weight.fill_(0.25)
        model.get_module(FAKE_B).proj.weight.fill_(0.5)
    model.save_pretrained(tmp_path)

    loaded = OmniModel.from_pretrained(tmp_path)

    assert set(loaded.modules_dict) == {FAKE_A, FAKE_B}
    assert isinstance(loaded.get_module(FAKE_A), FakeModuleA)
    assert isinstance(loaded.get_module(FAKE_B), FakeModuleB)
    assert torch.equal(loaded.get_module(FAKE_A).proj.weight, model.get_module(FAKE_A).proj.weight)
    assert torch.equal(loaded.get_module(FAKE_B).proj.weight, model.get_module(FAKE_B).proj.weight)


def test_omni_model_from_pretrained_forwards_dtype_to_modules(tmp_path):
    _build_omni_model().save_pretrained(tmp_path)

    loaded = OmniModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)

    assert loaded.get_module(FAKE_A).proj.weight.dtype == torch.bfloat16
    assert loaded.get_module(FAKE_B).proj.weight.dtype == torch.bfloat16


def test_from_pretrained_takes_each_module_attention_from_the_checkpoint(tmp_path):
    """The persisted kernels have to reach the per-module ``from_pretrained``.

    This entry point has no launcher behind it, so the checkpoint is the only
    place a module's attention can come from — and it is per module, which is
    the whole reason it is not a single load-wide kwarg. Without this the two
    modules both silently take HF's default.
    """
    _build_omni_model().save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["_module_entries"][FAKE_A]["ops_implementation"] = {"attn_implementation": "eager"}
    config["_module_entries"][FAKE_B]["ops_implementation"] = {"attn_implementation": "sdpa"}
    (tmp_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    loaded = OmniModel.from_pretrained(tmp_path)

    assert loaded.get_module(FAKE_A).config._attn_implementation == "eager"
    assert loaded.get_module(FAKE_B).config._attn_implementation == "sdpa"


def test_a_persisted_attention_wins_over_the_callers_load_kwarg(tmp_path):
    """Which of the two ways in to name a kernel takes precedence, per module.

    A module that was exported with its own kernels keeps them; a load-wide
    ``attn_implementation`` only fills in the modules that persisted none. The
    alternative — letting one kwarg flatten every module — is the thing
    per-module ``ops_implementation`` exists to prevent, and it would silently
    undo a checkpoint whose modules were deliberately exported with different
    attention. ``FAKE_B`` persists nothing here, so it shows the other half.
    """
    _build_omni_model().save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["_module_entries"][FAKE_A]["ops_implementation"] = {"attn_implementation": "eager"}
    (tmp_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    loaded = OmniModel.from_pretrained(tmp_path, attn_implementation="sdpa")

    assert loaded.get_module(FAKE_A).config._attn_implementation == "eager"
    assert loaded.get_module(FAKE_B).config._attn_implementation == "sdpa"


def test_kernel_layers_merge_per_field_across_the_model_and_the_module(tmp_path):
    """The composition and the module are different authorities, and both are heard.

    Per-field is the whole point: a model-level default has to reach a module
    that never mentioned that op, without the module's own choices being
    flattened. Whole-block replacement would make a five-layer order pointless
    — the top layer that spoke at all would decide everything.

    ``FAKE_A`` exercises the contested field: its own ``config.json`` says
    ``sdpa``, the composition says ``eager`` for it, and the composition wins
    because it is the one assembling this model out of a module it did not
    necessarily export.
    """
    _build_omni_model().save_pretrained(tmp_path)
    root = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    root["ops_implementation"] = {"rms_norm_implementation": "eager"}  # model-wide default
    root["_module_entries"][FAKE_A]["ops_implementation"] = {"attn_implementation": "eager"}  # the say over A
    (tmp_path / "config.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
    _state_module_ops(tmp_path / FAKE_A, {"attn_implementation": "sdpa"})  # what A was exported with

    loaded = OmniModel.from_pretrained(tmp_path)

    assert loaded.get_module(FAKE_A).config._attn_implementation == "eager"
    assert loaded.config._module_configs[FAKE_A].ops_implementation["rms_norm_implementation"] == "eager"
    # The module that stated nothing still gets the model-level default.
    assert loaded.config._module_configs[FAKE_B].ops_implementation["rms_norm_implementation"] == "eager"


def test_a_module_scoped_kwarg_overrides_what_the_composition_states(tmp_path):
    """``<module>.xxx_implementation`` is the only way past a persisted choice.

    A load-wide kwarg deliberately ranks under what a module persisted (see
    ``test_a_persisted_attention_wins_over_the_callers_load_kwarg``), which
    would leave no way to retarget one module from the call site — the thing a
    debugging session actually needs.
    """
    _build_omni_model().save_pretrained(tmp_path)
    root = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    root["_module_entries"][FAKE_A]["ops_implementation"] = {"attn_implementation": "sdpa"}
    (tmp_path / "config.json").write_text(json.dumps(root, indent=2), encoding="utf-8")

    loaded = OmniModel.from_pretrained(tmp_path, **{f"{FAKE_A}.attn_implementation": "eager"})

    assert loaded.get_module(FAKE_A).config._attn_implementation == "eager"


@pytest.mark.parametrize(
    ("kwarg", "match"),
    [
        ("no_such_module.attn_implementation", "does not declare"),
        (f"{FAKE_A}.not_an_op", "cannot override"),
    ],
)
def test_a_malformed_module_scoped_kwarg_is_rejected(tmp_path, kwarg, match):
    """Neither half of the dotted name may be wrong, quietly.

    HF turns an unrecognized kwarg into a config attribute, so forwarding one
    would ``setattr`` a non-identifier name and load with the caller's override
    silently absent.
    """
    _build_omni_model().save_pretrained(tmp_path)

    with pytest.raises(ValueError, match=match):
        OmniModel.from_pretrained(tmp_path, **{kwarg: "eager"})


def test_omni_kwargs_overwrite_root_ops_and_module_slot_fields():
    """A caller's kwargs overwrite the root kernels and each module slot field.

    ``pop_omni_kwargs`` takes them out before transformers can turn them into
    attributes; ``apply_omni_kwargs`` writes them onto the config. Dict fields
    merge per key. ``model_path`` replaces the slot's path. Anything else
    (``torch_dtype``) stays for HF.
    """
    config = OmniConfig(
        _module_entries={
            FAKE_A: {
                "model_path": FAKE_A,
                "model_config": {"hidden_size": HIDDEN_SIZE, "keep": 1},
                "ops_implementation": {"rms_norm_implementation": "eager"},
            }
        },
        training_graphs={"default": _chain_edges()},
        generation_graphs={"default": _minimal_generation_graph()},
    )
    kwargs = {
        "ops_implementation": {"attn_implementation": "sdpa"},
        f"{FAKE_A}.model_path": "/elsewhere/fake_module_a",
        f"{FAKE_A}.model_config": {"hidden_size": 16},
        f"{FAKE_A}.processor_config": {"packed_preprocess": True},
        f"{FAKE_A}.attn_implementation": "eager",
        "torch_dtype": "float32",
    }

    omni_kwargs = pop_omni_kwargs(kwargs)

    assert kwargs == {"torch_dtype": "float32"}
    assert config.apply_omni_kwargs(omni_kwargs) == {}
    assert config.ops_implementation == {"attn_implementation": "sdpa"}
    entry = config._module_entries[FAKE_A]
    assert entry["model_path"] == "/elsewhere/fake_module_a"
    assert entry["model_config"] == {"hidden_size": 16, "keep": 1}
    assert entry["processor_config"] == {"packed_preprocess": True}
    assert entry["ops_implementation"] == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "eager",
    }


def test_a_kwarg_lands_on_the_layer_it_overrides_and_is_saved_there(tmp_path):
    """A caller's selection becomes part of the model, at its own rank.

    ``apply_omni_kwargs`` merges each selection into the layer it is
    overriding, so a kwarg is a way of *stating* a layer rather than a
    parallel one that exists only while the process lives: the loaded config
    describes the run, a save writes it, and a reload reproduces it without
    the caller repeating themselves. The loaded module config holds the
    resolved kernels, so a save of that config records them.
    """
    _build_omni_model().save_pretrained(tmp_path)
    root = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    root["ops_implementation"] = {"rms_norm_implementation": "eager"}
    root["_module_entries"][FAKE_A]["ops_implementation"] = {"attn_implementation": "sdpa"}
    (tmp_path / "config.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
    _state_module_ops(tmp_path / FAKE_B, {"attn_implementation": "sdpa"})

    loaded = OmniModel.from_pretrained(tmp_path, **{f"{FAKE_A}.attn_implementation": "eager"})
    assert loaded.get_module(FAKE_A).config._attn_implementation == "eager"
    # The slot keeps the caller's say. The loaded module config holds the resolved kernels.
    assert loaded.config._module_entries[FAKE_A]["ops_implementation"] == {"attn_implementation": "eager"}
    assert loaded.config._module_configs[FAKE_A].ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "eager",
    }

    resaved = tmp_path / "resaved"
    loaded.save_pretrained(resaved)

    saved_root = json.loads((resaved / "config.json").read_text(encoding="utf-8"))
    assert saved_root["ops_implementation"] == {"rms_norm_implementation": "eager"}
    assert saved_root["_module_entries"][FAKE_A]["ops_implementation"] == {"attn_implementation": "eager"}
    # The saved module config is the resolved one: B's own attention plus the root default.
    assert json.loads((resaved / FAKE_B / "config.json").read_text(encoding="utf-8"))["ops_implementation"] == {
        "attn_implementation": "sdpa",
        "rms_norm_implementation": "eager",
    }

    reloaded = OmniModel.from_pretrained(resaved)
    assert reloaded.get_module(FAKE_A).config._attn_implementation == "eager"
    assert reloaded.config._module_configs[FAKE_B].ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "sdpa",
    }


def test_a_kernel_block_kwarg_merges_per_field_like_any_other_layer(tmp_path):
    """``ops_implementation={...}`` states choices; it does not replace a layer.

    The block spelling used to be the one kwarg neither kernel path caught:
    it fell through to every module's ``from_pretrained``, where HF turned it
    into a module attribute — the wrong rank, with no error.
    """
    _build_omni_model().save_pretrained(tmp_path)
    root = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    root["ops_implementation"] = {"rms_norm_implementation": "eager"}
    (tmp_path / "config.json").write_text(json.dumps(root, indent=2), encoding="utf-8")

    loaded = OmniModel.from_pretrained(
        tmp_path,
        ops_implementation={"attn_implementation": "sdpa"},
        **{f"{FAKE_B}.ops_implementation": {"attn_implementation": "eager"}},
    )

    assert loaded.config.ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "sdpa",
    }
    assert loaded.get_module(FAKE_A).config._attn_implementation == "sdpa"
    assert loaded.get_module(FAKE_B).config._attn_implementation == "eager"


def test_resolving_the_config_first_loads_the_same_model(tmp_path):
    """A kernel kwarg means the same thing at either entry point.

    ``OmniConfig.from_pretrained`` is where a launcher resolves a checkpoint
    before handing the config to ``OmniModel.from_pretrained(config=...)``, so
    a selection it does not understand is one transformers turns into a config
    attribute — the load then silently runs without it.
    """
    _build_omni_model().save_pretrained(tmp_path)
    kwarg = {f"{FAKE_B}.attn_implementation": "sdpa", "rms_norm_implementation": "eager"}

    config = OmniConfig.from_pretrained(tmp_path, **kwarg)
    assert config.ops_implementation == {"rms_norm_implementation": "eager"}
    assert config._module_entries[FAKE_B]["ops_implementation"] == {"attn_implementation": "sdpa"}
    assert not hasattr(config, f"{FAKE_B}.attn_implementation")

    from_config = OmniModel.from_pretrained(tmp_path, config=config)
    direct = OmniModel.from_pretrained(tmp_path, **kwarg)
    for name in (FAKE_A, FAKE_B):
        assert (
            from_config.config._module_configs[name].ops_implementation
            == direct.config._module_configs[name].ops_implementation
        )
        assert (
            from_config.get_module(name).config._attn_implementation
            == direct.get_module(name).config._attn_implementation
        )


def test_omni_kwargs_overwrite_slot_fields():
    """A caller can overwrite every field a slot stores, not only its kernels.

    ``pop_omni_kwargs`` lifts those keys out before transformers can turn them
    into attributes; ``apply_omni_kwargs`` writes them onto the root ops block
    and the named module's slot. Dict fields merge per key. ``model_path`` and
    ``model_config`` drop a cached module config, because both change what a
    later read of that file returns.
    """
    config = OmniConfig(
        _module_entries={
            FAKE_A: {
                "model_path": FAKE_A,
                "model_config": {"hidden_size": HIDDEN_SIZE, "keep": True},
                "ops_implementation": {"rms_norm_implementation": "eager"},
            }
        },
        training_graphs={},
        generation_graphs={},
    )
    kwargs = {
        "ops_implementation": {"attn_implementation": "sdpa"},
        f"{FAKE_A}.model_path": "/elsewhere/fake_module_a",
        f"{FAKE_A}.model_config": {"hidden_size": 16},
        f"{FAKE_A}.processor_config": {"packed_preprocess": True},
        f"{FAKE_A}.attn_implementation": "eager",
        "torch_dtype": "float32",
    }

    omni_kwargs = pop_omni_kwargs(kwargs)

    assert kwargs == {"torch_dtype": "float32"}
    assert "torch_dtype" not in omni_kwargs
    cached = object()
    config._module_configs[FAKE_A] = cached
    assert config.apply_omni_kwargs(omni_kwargs) == {}
    assert FAKE_A not in config._module_configs

    assert config.ops_implementation == {"attn_implementation": "sdpa"}
    entry = config._module_entries[FAKE_A]
    assert entry["model_path"] == "/elsewhere/fake_module_a"
    assert entry["model_config"] == {"hidden_size": 16, "keep": True}
    assert entry["processor_config"] == {"packed_preprocess": True}
    assert entry["ops_implementation"] == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "eager",
    }

    config._module_configs[FAKE_A] = cached
    config.apply_omni_kwargs({f"{FAKE_A}.processor_config": {"packed_preprocess": False}})
    assert config._module_configs[FAKE_A] is cached
    assert config._module_entries[FAKE_A]["processor_config"] == {"packed_preprocess": False}


def test_a_malformed_kernel_block_kwarg_is_rejected(tmp_path):
    """A block is checked field by field, for the same reason a dotted key is."""
    _build_omni_model().save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="no such kernel fields: not_an_op"):
        OmniModel.from_pretrained(tmp_path, ops_implementation={"not_an_op": "eager"})

    with pytest.raises(ValueError, match="must be a dict"):
        OmniModel.from_pretrained(tmp_path, **{f"{FAKE_A}.ops_implementation": "eager"})


def test_a_slots_kernels_survive_a_config_only_reexport(tmp_path):
    """A slot's kernels must round-trip like ``model_config``.

    Reading each module's own config does not consume what the composition
    stated about it, so a launcher-less load→save still emits the slot's
    ``ops_implementation`` instead of silently dropping it.
    """
    _write_omni_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["_module_entries"][FAKE_A]["model_config"] = {"hidden_size": 16}
    raw["_module_entries"][FAKE_A]["ops_implementation"] = {"attn_implementation": "eager"}
    config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    config = OmniConfig.from_pretrained(tmp_path)
    assert config._module_entries[FAKE_A]["ops_implementation"]["attn_implementation"] == "eager"
    assert config._module_configs[FAKE_A].hidden_size == 16

    exported = tmp_path / "exported"
    config.save_pretrained(exported)
    saved = json.loads((exported / "config.json").read_text(encoding="utf-8"))
    assert saved["_module_entries"][FAKE_A]["ops_implementation"]["attn_implementation"] == "eager"
    assert saved["_module_entries"][FAKE_A]["model_config"]["hidden_size"] == 16
    # Export is self-contained: the location is always the subfolder written.
    assert saved["_module_entries"][FAKE_A]["model_path"] == FAKE_A


def test_omni_model_from_config_builds_unweighted_modules(tmp_path):
    _write_omni_checkpoint(tmp_path)
    config = OmniConfig.from_pretrained(tmp_path)

    model = OmniModel.from_config(config, checkpoint_root=tmp_path)

    assert set(model.modules_dict) == {FAKE_A, FAKE_B}
    assert isinstance(model.get_module(FAKE_A), FakeModuleA)
    assert isinstance(model.get_module(FAKE_B), FakeModuleB)


def test_a_module_configured_outside_the_root_keeps_its_own_path(tmp_path):
    """A module living outside the root keeps resolving to its own path.

    The root has no ``<name>/`` directory here at all, so there is no module
    config to read; the slot stays a bare descriptor and the configured path
    is the only thing weight loading can follow.
    """
    _write_omni_checkpoint(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    _write_module_stub(elsewhere, model_type=FAKE_A)
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text())
    raw["_module_entries"][FAKE_A] = {"model_path": str(elsewhere)}
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    shutil.rmtree(tmp_path / FAKE_A)

    config = OmniConfig.from_pretrained(tmp_path)

    assert config.resolve_module_path(str(tmp_path), FAKE_A) == str(elsewhere)


def test_reading_a_module_config_keeps_the_descriptor_it_was_declared_with(tmp_path):
    """Reading a module's ``config.json`` must not cost its descriptor.

    The slot and the module's own file are two authorities that share a field
    vocabulary, so the module config is kept beside its slot rather than in
    place of it. Collapsing them lost everything only the composition says:
    ``OmniProcessor.from_config`` then built each preprocessor with no
    ``processor_config``, silently disagreeing with the model it serves.
    """
    _write_omni_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text())
    raw["_module_entries"][FAKE_A] = {
        "model_path": FAKE_A,
        "model_config": {"hidden_size": 16},
        "ops_implementation": {"attn_implementation": "eager"},
        "processor_config": {"packed_preprocess": True},
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    config = OmniConfig.from_pretrained(tmp_path)

    assert config._module_entries[FAKE_A]["processor_config"] == {"packed_preprocess": True}
    assert config._module_entries[FAKE_A]["model_config"] == {"hidden_size": 16}
    assert config._module_entries[FAKE_A]["ops_implementation"] == {"attn_implementation": "eager"}
    assert config._module_configs[FAKE_A].hidden_size == 16


def test_an_external_module_path_survives_a_same_named_directory_in_the_root(tmp_path):
    """A configured ``model_path`` wins over ``root/<name>``, present or not.

    Reading the module config keyed on ``root/<name>`` once, so a module
    pointed at another checkpoint was described by the root's same-named
    directory instead — the wrong weights, with no error.
    """
    _write_omni_checkpoint(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    _write_module_stub(elsewhere, model_type=FAKE_A)
    (elsewhere / "config.json").write_text(
        json.dumps({"model_type": FAKE_A, "hidden_size": 32}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text())
    raw["_module_entries"][FAKE_A] = {"model_path": str(elsewhere)}
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    config = OmniConfig.from_pretrained(tmp_path)

    assert config.resolve_module_path(str(tmp_path), FAKE_A) == str(elsewhere)
    assert config._module_configs[FAKE_A].hidden_size == 32


def test_from_pretrained_keeps_a_graph_the_caller_passed(tmp_path):
    """Sidecars are the checkpoint's default, not an override of the caller.

    A launcher runs a checkpoint under the graph its YAML declares; hydrating
    the sidecars over an explicit kwarg ran the exported graph instead.
    """
    _write_omni_checkpoint(tmp_path)
    caller_training_graph = [{"from": FAKE_B, "to": "end"}]
    caller_generation_graphs = {"infer_gen": _minimal_generation_graph(module=FAKE_B)}

    config = OmniConfig.from_pretrained(
        tmp_path,
        training_graphs={"default": caller_training_graph},
        generation_graphs=caller_generation_graphs,
    )

    assert config.training_graph == caller_training_graph
    assert config.generation_graphs == caller_generation_graphs
    assert OmniConfig.from_pretrained(tmp_path).training_graph == _chain_edges()


def test_from_pretrained_keeps_a_scenario_the_caller_selected(tmp_path):
    """``train_type`` / ``infer_type`` kwargs pick the scenario out of the sidecars."""
    _write_omni_checkpoint(tmp_path)

    config = OmniConfig.from_pretrained(tmp_path, infer_type="infer_und")

    assert config.infer_type == "infer_und"
    assert config.generation_graph["initial"] == "understand"


def test_from_config_forwards_load_kwargs_to_unweighted_modules(tmp_path):
    """Building without weights must honour the load options ``__init__`` sees.

    The descriptor branch is the one that reads each module's ``config.json``
    off disk; it used to pass only the per-module ``model_config`` overrides,
    dropping the caller's ``dtype``. It must not swing the other way either: a
    global weight-placement option like ``device_map`` reaches ``__init__``
    through ``_from_config`` and would raise there.
    """
    _write_omni_checkpoint(tmp_path)
    config = OmniConfig.from_pretrained(tmp_path)
    config._module_entries = {name: {"model_path": name} for name in config.module_names}

    model = OmniModel.from_config(config, checkpoint_root=tmp_path, dtype="bfloat16", device_map="auto")

    assert model.get_module(FAKE_A).proj.weight.dtype == torch.bfloat16
    assert model.get_module(FAKE_B).proj.weight.dtype == torch.bfloat16


def test_from_pretrained_loads_modules_from_the_config_origin(tmp_path):
    """Weights come from the directory the config was resolved against.

    Module configs are loaded with :meth:`OmniConfig.checkpoint_root` first.
    Passing another directory to ``OmniModel.from_pretrained`` does not point
    the weights somewhere else.
    """
    origin, target = tmp_path / "origin", tmp_path / "target"
    origin_model = _build_omni_model()
    with torch.no_grad():
        origin_model.get_module(FAKE_A).proj.weight.fill_(1.0)
        origin_model.get_module(FAKE_B).proj.weight.fill_(1.0)
    origin_model.save_pretrained(origin)

    target_model = _build_omni_model()
    with torch.no_grad():
        target_model.get_module(FAKE_A).proj.weight.fill_(2.0)
        target_model.get_module(FAKE_B).proj.weight.fill_(2.0)
    target_model.save_pretrained(target)

    config = OmniConfig.from_pretrained(origin)
    config.name_or_path = str(origin)

    loaded = OmniModel.from_pretrained(target, config=config)

    assert torch.equal(loaded.get_module(FAKE_A).proj.weight, origin_model.get_module(FAKE_A).proj.weight)
    assert torch.equal(loaded.get_module(FAKE_B).proj.weight, origin_model.get_module(FAKE_B).proj.weight)


def test_omni_config_save_pretrained_writes_graph_sidecars(tmp_path):
    model = _build_omni_model()

    model.save_pretrained(tmp_path, save_module_weights=False)

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "training_graphs" not in saved
    assert saved["_module_entries"] == _chain_modules()
    assert "generation_graphs" not in saved
    assert (tmp_path / DEFAULT_GENERATION_GRAPH_FILE).exists()
    assert yaml.safe_load((tmp_path / DEFAULT_TRAINING_GRAPH_FILE).read_text(encoding="utf-8")) == {
        "default": _chain_edges()
    }
    sidecar = yaml.safe_load((tmp_path / DEFAULT_GENERATION_GRAPH_FILE).read_text(encoding="utf-8"))
    assert list(sidecar) == ["infer_gen"]
    assert (tmp_path / "graphs" / "training.mmd").exists()
    assert (tmp_path / "graphs" / "generation_infer_gen.mmd").exists()
    training_mmd = (tmp_path / "graphs" / "training.mmd").read_text(encoding="utf-8")
    assert "flowchart" in training_mmd


def test_inference_only_config_saves_without_a_training_diagram(tmp_path):
    """No ``training_graph`` must cost the save nothing but the training diagram.

    ``from_dict`` defaults ``training_graphs`` to ``{}``, which is what a
    module-only or inference-only checkpoint round-trips as, and
    ``TrainingGraph`` rejects an empty edge list. The diagrams are written last,
    so rendering one unconditionally aborted the save after every real artifact
    — including, via ``OmniModel.save_pretrained``, each module's weights — was
    already on disk.
    """
    config = OmniConfig.from_dict(
        {"_module_entries": _chain_modules(), "generation_graphs": _minimal_generation_graphs()}
    )
    assert config.training_graph == []
    model = OmniModel(
        config,
        {
            FAKE_A: FakeModuleA(FakeModuleAConfig(hidden_size=HIDDEN_SIZE)),
            FAKE_B: FakeModuleB(FakeModuleBConfig(hidden_size=HIDDEN_SIZE)),
        },
    )

    model.save_pretrained(tmp_path, save_module_weights=False)

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "graphs" / "generation_infer_gen.mmd").exists()
    assert not (tmp_path / "graphs" / "training.mmd").exists()
    assert OmniConfig.from_pretrained(tmp_path).training_graph == []


def test_omni_model_save_pretrained_roundtrip_layout(tmp_path):
    model = _build_omni_model()
    save_root = tmp_path / "saved_omni"
    model.save_pretrained(save_root, save_module_weights=False)

    assert (save_root / "config.json").exists()
    assert (save_root / DEFAULT_TRAINING_GRAPH_FILE).exists()
    assert (save_root / DEFAULT_GENERATION_GRAPH_FILE).exists()
    assert (save_root / "graphs" / "training.mmd").exists()
    assert (save_root / "graphs" / "generation_infer_gen.mmd").exists()
    assert (save_root / FAKE_A / "config.json").exists()
    assert (save_root / FAKE_B / "config.json").exists()

    reloaded = OmniConfig.from_pretrained(save_root)
    assert reloaded.training_graph[0]["from"] == FAKE_A
    assert isinstance(reloaded._module_configs[FAKE_A], PretrainedConfig)
    assert isinstance(reloaded._module_configs[FAKE_B], PretrainedConfig)
    assert json.loads((save_root / FAKE_A / "config.json").read_text(encoding="utf-8"))["model_type"] == FAKE_A
    assert json.loads((save_root / FAKE_B / "config.json").read_text(encoding="utf-8"))["model_type"] == FAKE_B


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_dict": {}},
        {"push_to_hub": True},
        {"distributed_checkpoint": True},
    ],
)
def test_omni_model_save_pretrained_rejects_single_model_options(tmp_path, kwargs):
    """``save_pretrained`` takes the base-class signature but cannot honour the
    options that assume one model behind one checkpoint."""
    with pytest.raises(NotImplementedError):
        _build_omni_model().save_pretrained(tmp_path, **kwargs)


def test_save_pretrained_roundtrips_every_generation_scenario(tmp_path):
    """An exported checkpoint stays multi-scenario — it is not locked to the active one."""
    config = OmniConfig(
        _module_entries=_chain_modules(),
        training_graphs={"default": _chain_edges()},
        generation_graphs={
            "infer_gen": _minimal_generation_graph(module=FAKE_A),
            "infer_und": {
                "initial": "understand",
                "states": {
                    "understand": {
                        "body": [{"from": FAKE_B, "to": "end"}],
                        "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                    }
                },
            },
        },
        infer_type="infer_und",
    )
    model = OmniModel(
        config,
        {
            FAKE_A: FakeModuleA(FakeModuleAConfig(hidden_size=HIDDEN_SIZE)),
            FAKE_B: FakeModuleB(FakeModuleBConfig(hidden_size=HIDDEN_SIZE)),
        },
    )

    model.save_pretrained(tmp_path, save_module_weights=False)
    reloaded = OmniConfig.from_pretrained(tmp_path)

    assert reloaded.infer_types == ["infer_gen", "infer_und"]
    assert reloaded.infer_type == "infer_und"
    assert reloaded.generation_graph["initial"] == "understand"
    assert reloaded.generation_graphs["infer_gen"]["initial"] == "run"
    for infer_type in reloaded.infer_types:
        assert (tmp_path / "graphs" / f"generation_{infer_type}.mmd").exists()


def test_omni_model_resolve_generation_kwargs_uses_config_defaults():
    model = _build_omni_model()
    model.config.generation_kwargs = {"max_new_tokens": 64}

    assert model.resolve_generation_kwargs(None) == {"max_new_tokens": 64}
    assert model.resolve_generation_kwargs({"temperature": 0.2}) == {
        "max_new_tokens": 64,
        "temperature": 0.2,
    }
    assert model.resolve_generation_kwargs({"max_new_tokens": 8}) == {"max_new_tokens": 8}


def test_omni_model_post_init_unions_child_no_split_modules():
    """HF ``post_init`` (re-run after children are attached) is the aggregator."""
    model = _build_omni_model()
    assert model._no_split_modules == {"FakeModuleA", "FakeModuleB"}


def test_from_pretrained_rejects_missing_endpoint_method(tmp_path):
    """A sidecar that names a missing method fails when the graph is built at load."""
    _build_omni_model().save_pretrained(tmp_path)
    (tmp_path / DEFAULT_TRAINING_GRAPH_FILE).write_text(
        "default:\n- {from: fake_module_a.encode, to: end}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"FakeModuleA\.encode"):
        OmniModel.from_pretrained(tmp_path)


def test_omni_model_rejects_missing_default_generate():
    """Bare generation endpoints resolve to ``generate``; that method must exist when the FSM is built."""

    class NoGenerate(FakeModuleA):
        generate = None

    config = OmniConfig(
        _module_entries={FAKE_A: {"model_path": FAKE_A}},
        training_graphs={"default": [{"from": FAKE_A, "to": "end"}]},
        generation_graphs=_minimal_generation_graphs(module=FAKE_A),
    )
    with pytest.raises(ValueError, match=r"NoGenerate\.generate"):
        OmniModel(config, {FAKE_A: NoGenerate(FakeModuleAConfig())})
