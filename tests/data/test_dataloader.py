import types
from dataclasses import dataclass
from functools import partial
from typing import Literal

import pytest
from utils import DummyDataset, process_dummy_example

from veomni.data import build_dataloader, build_dataset
from veomni.data.data_loader import DATALOADER_REGISTRY, resolve_dyn_bsz_buffer_size
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy


def _fake_ps(sp_size: int):
    sp_enabled = sp_size > 1
    return types.SimpleNamespace(
        dp_size=1,
        dp_rank=0,
        sp_enabled=sp_enabled,
        sp_size=sp_size,
        sp_rank=0,
    )


def _context_aware_native_dataloader_kwargs():
    return {
        "dataset": [{"input_ids": [1], "attention_mask": [1]}],
        "micro_batch_size": 1,
        "global_batch_size": 1,
        "dataloader_batch_size": 1,
        "max_seq_len": 1024 * 1024,
        "train_steps": 1,
        "dyn_bsz": True,
        "dyn_bsz_buffer_size": 200,
        "dyn_bsz_buffer_policy": "context_aware",
        "dyn_bsz_runtime": "main",
        "dyn_bsz_count_mode": "total",
        "data_modality": "text",
        "num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": None,
    }


def test_vlm_data_arguments_identify_multimodal_policy_scope():
    from veomni.trainer.vlm_trainer import VLMMDataArguments

    assert VLMMDataArguments.data_modality == "multimodal"


def test_base_trainer_forwards_buffer_policy_scope(monkeypatch):
    import veomni.trainer.base as base_mod
    from veomni.arguments.arguments_types import DataloaderConfig
    from veomni.trainer.base import BaseTrainer

    captured_kwargs = {}
    monkeypatch.setattr(base_mod, "build_dataloader", lambda **kwargs: captured_kwargs.update(kwargs) or object())
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.train_dataset = object()
    trainer.collate_fn = None
    trainer.args = types.SimpleNamespace(
        data=types.SimpleNamespace(
            dataloader=DataloaderConfig(
                type="native",
                num_workers=0,
                drop_last=True,
                pin_memory=False,
                prefetch_factor=2,
            ),
            max_seq_len=1024 * 1024,
            dyn_bsz_buffer_size=200,
            dyn_bsz_buffer_policy="context_aware",
            data_modality="multimodal",
        ),
        train=types.SimpleNamespace(
            micro_batch_size=1,
            global_batch_size=1,
            dataloader_batch_size=1,
            dyn_bsz=True,
            dyn_bsz_runtime="worker",
            dyn_bsz_count_mode="total",
            dyn_bsz_physical_overflow_ratio=1.5,
            bsz_warmup_ratio=0.02,
            bsz_warmup_init_mbtoken=200,
            seed=0,
            checkpoint=types.SimpleNamespace(save_steps=1000),
        ),
        train_steps=1,
    )

    trainer._build_dataloader()

    assert captured_kwargs["dyn_bsz_buffer_policy"] == "context_aware"
    assert captured_kwargs["dyn_bsz_runtime"] == "worker"
    assert captured_kwargs["data_modality"] == "multimodal"
    assert captured_kwargs["_veomni_trainer_controls"] is True


@pytest.mark.parametrize(
    ("token_budget", "expected_buffer_size"),
    [
        (512 * 1024, 24),
        (1024 * 1024, 24),
    ],
)
def test_resolve_context_aware_dyn_bsz_buffer_size(token_budget: int, expected_buffer_size: int):
    assert (
        resolve_dyn_bsz_buffer_size(
            buffer_size=200,
            buffer_policy="context_aware",
            max_seq_len=token_budget,
            micro_batch_size=1,
            runtime="main",
            count_mode="total",
            data_modality="text",
        )
        == expected_buffer_size
    )


def test_resolve_fixed_dyn_bsz_buffer_size_preserves_configuration():
    assert (
        resolve_dyn_bsz_buffer_size(
            buffer_size=37,
            buffer_policy="fixed",
            max_seq_len=1024 * 1024,
            micro_batch_size=4,
            runtime="worker",
            count_mode="effective",
            data_modality="multimodal",
        )
        == 37
    )


def test_resolve_context_aware_dyn_bsz_buffer_size_rejects_worker_runtime():
    with pytest.raises(ValueError, match="only supported with dyn_bsz_runtime='main'"):
        resolve_dyn_bsz_buffer_size(
            buffer_size=200,
            buffer_policy="context_aware",
            max_seq_len=1024 * 1024,
            micro_batch_size=1,
            runtime="worker",
            count_mode="total",
            data_modality="text",
        )


def test_resolve_context_aware_dyn_bsz_buffer_size_rejects_unknown_context_length():
    with pytest.raises(ValueError, match="supports max_seq_len values"):
        resolve_dyn_bsz_buffer_size(
            buffer_size=200,
            buffer_policy="context_aware",
            max_seq_len=256 * 1024,
            micro_batch_size=1,
            runtime="main",
            count_mode="total",
            data_modality="text",
        )


def test_resolve_dyn_bsz_buffer_size_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unknown dyn_bsz_buffer_policy"):
        resolve_dyn_bsz_buffer_size(
            buffer_size=200,
            buffer_policy="typo",
            max_seq_len=1024 * 1024,
            micro_batch_size=1,
            runtime="main",
            count_mode="total",
            data_modality="text",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count_mode": "effective"}, "dyn_bsz_count_mode='total'"),
        ({"data_modality": "multimodal"}, "only supported for text data"),
        ({"micro_batch_size": 2}, "micro_batch_size=1"),
    ],
)
def test_resolve_context_aware_dyn_bsz_buffer_size_rejects_unvalidated_configuration(kwargs, message):
    arguments = {
        "buffer_size": 200,
        "buffer_policy": "context_aware",
        "max_seq_len": 1024 * 1024,
        "micro_batch_size": 1,
        "runtime": "main",
        "count_mode": "total",
        "data_modality": "text",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        resolve_dyn_bsz_buffer_size(**arguments)


def test_build_dataloader_resolves_context_aware_buffer_before_registry_dispatch(monkeypatch):
    import veomni.data.data_loader as m_dl

    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: _fake_ps(sp_size=1))
    captured_kwargs = {}

    def external_builder(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    test_loader_name = "external_context_aware_test"
    DATALOADER_REGISTRY[test_loader_name] = external_builder
    try:
        build_dataloader(
            dataloader_type=test_loader_name,
            dyn_bsz=True,
            dyn_bsz_buffer_size=200,
            dyn_bsz_buffer_policy="context_aware",
            max_seq_len=1024 * 1024,
            micro_batch_size=1,
            dyn_bsz_runtime="main",
            dyn_bsz_count_mode="total",
            data_modality="text",
        )
    finally:
        del DATALOADER_REGISTRY[test_loader_name]

    assert captured_kwargs["dyn_bsz_buffer_size"] == 24
    assert captured_kwargs["dyn_bsz_buffer_policy"] == "context_aware"


@pytest.fixture
def context_builder(monkeypatch):
    def register(builder):
        name = "external_context_regression"
        monkeypatch.setitem(DATALOADER_REGISTRY._local_mapping, name, builder)
        return name

    return register


def _external_context_kwargs():
    return dict(
        dyn_bsz_buffer_policy="context_aware",
        max_seq_len=1024 * 1024,
        micro_batch_size=1,
        dyn_bsz_runtime="main",
        dyn_bsz_count_mode="total",
        data_modality="text",
    )


@pytest.mark.parametrize("default_dyn_bsz", [True, False])
def test_external_builder_default_dyn_bsz_is_used(default_dyn_bsz, context_builder):
    def builder(dyn_bsz=default_dyn_bsz, **kwargs):
        return dyn_bsz, kwargs["dyn_bsz_buffer_size"]

    name = context_builder(builder)
    if default_dyn_bsz:
        assert build_dataloader(name, **_external_context_kwargs()) == (True, 24)
    else:
        with pytest.raises(ValueError, match="requires dyn_bsz=True"):
            build_dataloader(name, **_external_context_kwargs())


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (lambda **kwargs: kwargs, "explicit.*dyn_bsz"),
        (lambda dyn_bsz=True, **kwargs: kwargs, "explicit.*dyn_bsz_runtime"),
    ],
)
def test_context_aware_external_builder_requires_known_options(builder, message, context_builder):
    with pytest.raises(ValueError, match=message):
        build_dataloader(
            context_builder(builder), dyn_bsz_buffer_policy="context_aware", max_seq_len=1048576, micro_batch_size=1
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("dyn_bsz_runtime", "worker", "only supported with dyn_bsz_runtime='main'"),
        ("dyn_bsz_count_mode", "effective", "only supported with dyn_bsz_count_mode='total'"),
        ("data_modality", "multimodal", "only supported for text data"),
    ],
)
def test_context_aware_validates_external_defaults(option, value, message, context_builder):
    defaults = dict(dyn_bsz_runtime="main", dyn_bsz_count_mode="total", data_modality="text")
    defaults[option] = value

    def builder(
        dyn_bsz=True,
        dyn_bsz_runtime=defaults["dyn_bsz_runtime"],
        dyn_bsz_count_mode=defaults["dyn_bsz_count_mode"],
        data_modality=defaults["data_modality"],
        **kwargs,
    ):
        raise AssertionError("unsupported settings must be rejected before dispatch")

    with pytest.raises(ValueError, match=message):
        build_dataloader(
            context_builder(builder), dyn_bsz_buffer_policy="context_aware", max_seq_len=1048576, micro_batch_size=1
        )


@pytest.mark.parametrize("error", [ValueError, RuntimeError])
def test_fixed_policy_keeps_builder_when_signature_is_unavailable(monkeypatch, error, context_builder):
    import veomni.data.data_loader as m_dl

    def unavailable(_builder):
        raise error("signature unavailable")

    name = context_builder(lambda **kwargs: kwargs)
    monkeypatch.setattr(m_dl.inspect, "signature", unavailable)
    assert build_dataloader(name, dyn_bsz_buffer_size=37) == {"dyn_bsz_buffer_size": 37}


@pytest.mark.parametrize("accepts_kwargs", [False, True])
def test_fixed_opaque_builder_preserves_legacy_keywords(context_builder, accepts_kwargs):
    class OpaqueLegacyBuilder:
        __signature__ = "invalid"

        def __call__(self, **kwargs):
            if not accepts_kwargs and set(kwargs) != {"dyn_bsz_buffer_size"}:
                raise TypeError("legacy builder received new keywords")
            return kwargs

    assert build_dataloader(
        context_builder(OpaqueLegacyBuilder()),
        _veomni_trainer_controls=True,
        dyn_bsz_buffer_size=37,
        dyn_bsz_buffer_policy="fixed",
        data_modality="text",
    ) == {"dyn_bsz_buffer_size": 37}


def test_fixed_opaque_builder_keeps_public_explicit_modality(context_builder):
    class OpaqueBuilder:
        __signature__ = "invalid"

        def __call__(self, *, data_modality="text", **kwargs):
            return data_modality

    assert build_dataloader(context_builder(OpaqueBuilder()), data_modality="multimodal") == "multimodal"


def test_fixed_trainer_keeps_legacy_optional_modality_default(context_builder):
    def builder(*, data_modality="text", **kwargs):
        return data_modality

    assert (
        build_dataloader(
            context_builder(builder),
            _veomni_trainer_controls=True,
            dyn_bsz_buffer_policy="fixed",
            data_modality="multimodal",
        )
        == "text"
    )


def test_fixed_trainer_keeps_legacy_plugin_policy_default(context_builder):
    def builder(*, dyn_bsz_buffer_policy="adaptive", **kwargs):
        return dyn_bsz_buffer_policy

    assert (
        build_dataloader(
            context_builder(builder),
            _veomni_trainer_controls=True,
            dyn_bsz_buffer_policy="fixed",
            data_modality="multimodal",
        )
        == "adaptive"
    )


def test_fixed_trainer_kwargs_wrapper_keeps_legacy_keyword_contract(context_builder):
    def legacy_loader(*, dyn_bsz_buffer_size):
        return dyn_bsz_buffer_size

    def wrapper(**kwargs):
        return legacy_loader(**kwargs)

    assert (
        build_dataloader(
            context_builder(wrapper),
            _veomni_trainer_controls=True,
            dyn_bsz_buffer_policy="fixed",
            data_modality="multimodal",
            dyn_bsz_buffer_size=37,
        )
        == 37
    )


def test_fixed_trainer_kwargs_named_like_modality_remains_legacy_compatible(context_builder):
    def legacy_loader(*, dyn_bsz_buffer_size):
        return dyn_bsz_buffer_size

    def wrapper(**data_modality):
        return legacy_loader(**data_modality)

    assert (
        build_dataloader(
            context_builder(wrapper),
            _veomni_trainer_controls=True,
            dyn_bsz_buffer_policy="fixed",
            data_modality="multimodal",
            dyn_bsz_buffer_size=37,
        )
        == 37
    )


def test_multimodal_trainer_rejects_context_aware_builder_default(context_builder):
    def builder(
        dyn_bsz_buffer_policy="context_aware",
        data_modality="text",
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_count_mode="total",
        **kwargs,
    ):
        raise AssertionError("multimodal context-aware builder must not run")

    with pytest.raises(ValueError, match="only supported for text data"):
        build_dataloader(
            context_builder(builder),
            _veomni_trainer_controls=True,
            data_modality="multimodal",
            max_seq_len=1048576,
            micro_batch_size=1,
        )


def test_native_dataloader_preserves_legacy_positional_worker_arguments(monkeypatch):
    import veomni.data.data_loader as module

    monkeypatch.setattr(module, "get_parallel_state", lambda: _fake_ps(1))
    loader = module.build_native_dataloader(
        [{"input_ids": [1], "attention_mask": [1]}],
        1,
        1,
        1,
        16,
        1,
        0.0,
        200,
        False,
        "main",
        "total",
        1.5,
        False,
        200,
        0,
        None,
        False,
        False,
        None,
    )
    assert loader.num_workers == 0
    assert loader.drop_last is False
    assert loader.pin_memory is False


@pytest.mark.parametrize("accepts_kwargs", [False, True])
def test_fixed_trainer_preserves_positional_only_builder_defaults(context_builder, accepts_kwargs):
    if accepts_kwargs:

        def builder(data_modality="text", /, **kwargs):
            return data_modality, kwargs
    else:

        def builder(data_modality="text", /):
            return data_modality, {}

    name = context_builder(builder)
    assert build_dataloader(
        name, _veomni_trainer_controls=True, dyn_bsz_buffer_policy="fixed", data_modality="image"
    ) == ("text", {})


@pytest.mark.parametrize("runtime", ["main", "worker"])
def test_context_aware_default_policy_is_validated(runtime, context_builder):
    def builder(
        dyn_bsz_buffer_policy="context_aware",
        dyn_bsz=True,
        dyn_bsz_runtime=runtime,
        dyn_bsz_count_mode="total",
        data_modality="text",
        **kwargs,
    ):
        return kwargs["dyn_bsz_buffer_size"]

    name = context_builder(builder)
    if runtime == "main":
        assert build_dataloader(name, max_seq_len=1048576, micro_batch_size=1) == 24
    else:
        with pytest.raises(ValueError, match="only supported with dyn_bsz_runtime='main'"):
            build_dataloader(name, max_seq_len=1048576, micro_batch_size=1)


def test_context_aware_policy_rejects_uninspectable_builder(context_builder):
    class OpaqueBuilder:
        __signature__ = "invalid"

        def __call__(self, dyn_bsz=False, /, **kwargs):
            raise AssertionError("unknown arguments must be rejected before dispatch")

    with pytest.raises(ValueError, match="requires an inspectable builder signature"):
        build_dataloader(context_builder(OpaqueBuilder()), **_external_context_kwargs())


@pytest.mark.parametrize(
    "option",
    [
        "dyn_bsz_buffer_policy",
        "dyn_bsz_buffer_size",
        "dyn_bsz",
        "dyn_bsz_runtime",
        "dyn_bsz_count_mode",
        "data_modality",
        "max_seq_len",
        "micro_batch_size",
    ],
)
def test_context_aware_policy_rejects_positional_only_overrides(option, context_builder):
    def policy(dyn_bsz_buffer_policy="fixed", /, **kwargs): ...
    def size(dyn_bsz_buffer_size=200, /, **kwargs): ...
    def enabled(dyn_bsz=False, /, **kwargs): ...
    def runtime(dyn_bsz_runtime="worker", /, **kwargs): ...
    def count(dyn_bsz_count_mode="effective", /, **kwargs): ...
    def modality(data_modality="multimodal", /, **kwargs): ...
    def length(max_seq_len=1, /, **kwargs): ...
    def batch(micro_batch_size=2, /, **kwargs): ...

    builders = dict(
        dyn_bsz_buffer_policy=policy,
        dyn_bsz_buffer_size=size,
        dyn_bsz=enabled,
        dyn_bsz_runtime=runtime,
        dyn_bsz_count_mode=count,
        data_modality=modality,
        max_seq_len=length,
        micro_batch_size=batch,
    )
    with pytest.raises(ValueError, match=f"positional-only parameter {option}"):
        build_dataloader(context_builder(builders[option]), dyn_bsz=True, **_external_context_kwargs())


def test_build_dataloader_resolves_context_aware_native_buffer_once(monkeypatch):
    import veomni.data.data_loader as m_dl

    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: _fake_ps(sp_size=1))
    resolve_calls = []
    original_resolver = m_dl.resolve_dyn_bsz_buffer_size

    def recording_resolver(**kwargs):
        resolve_calls.append(kwargs)
        return original_resolver(**kwargs)

    monkeypatch.setattr(m_dl, "resolve_dyn_bsz_buffer_size", recording_resolver)
    build_dataloader(dataloader_type="native", **_context_aware_native_dataloader_kwargs())

    assert len(resolve_calls) == 1
    assert resolve_calls[0]["buffer_size"] == 200


def test_build_dataloader_native_alias_resolves_context_aware_buffer_once(monkeypatch):
    import veomni.data.data_loader as m_dl

    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: _fake_ps(sp_size=1))
    resolve_calls = []
    original_resolver = m_dl.resolve_dyn_bsz_buffer_size

    def recording_resolver(**kwargs):
        resolve_calls.append(kwargs)
        return original_resolver(**kwargs)

    monkeypatch.setattr(m_dl, "resolve_dyn_bsz_buffer_size", recording_resolver)
    alias = "native_context_aware_alias_test"
    DATALOADER_REGISTRY[alias] = m_dl.build_native_dataloader
    try:
        build_dataloader(dataloader_type=alias, **_context_aware_native_dataloader_kwargs())
    finally:
        del DATALOADER_REGISTRY[alias]

    assert len(resolve_calls) == 1
    assert resolve_calls[0]["buffer_size"] == 200


def test_build_dataloader_native_override_resolves_before_external_dispatch():
    captured_kwargs = {}

    def external_builder(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    DATALOADER_REGISTRY["native"] = external_builder
    try:
        build_dataloader(
            dataloader_type="native",
            dyn_bsz=True,
            dyn_bsz_buffer_size=200,
            dyn_bsz_buffer_policy="context_aware",
            max_seq_len=1024 * 1024,
            micro_batch_size=1,
            dyn_bsz_runtime="main",
            dyn_bsz_count_mode="total",
            data_modality="text",
        )
    finally:
        del DATALOADER_REGISTRY["native"]

    assert captured_kwargs["dyn_bsz_buffer_size"] == 24


def test_build_dataloader_rejects_context_aware_when_dyn_bsz_disabled():
    with pytest.raises(ValueError, match="requires dyn_bsz=True"):
        build_dataloader(
            dataloader_type="native",
            dyn_bsz=False,
            dyn_bsz_buffer_policy="context_aware",
            max_seq_len=1024 * 1024,
            micro_batch_size=1,
        )


def test_build_native_dataloader_rejects_context_aware_when_dyn_bsz_disabled():
    from veomni.data.data_loader import build_native_dataloader

    with pytest.raises(ValueError, match="requires dyn_bsz=True"):
        build_native_dataloader(
            dataset=[{"input_ids": [1]}],
            micro_batch_size=1,
            global_batch_size=1,
            dataloader_batch_size=1,
            max_seq_len=1024 * 1024,
            train_steps=1,
            dyn_bsz=False,
            dyn_bsz_buffer_policy="context_aware",
        )


def test_dit_trainer_rejects_context_aware_after_setup_disables_dyn_bsz(monkeypatch):
    import veomni.trainer.dit_trainer as dit_mod
    from veomni.arguments.arguments_types import DataloaderConfig
    from veomni.trainer.dit_trainer import DiTTrainer

    monkeypatch.setattr(
        dit_mod,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, sp_rank=0, dp_size=1),
    )
    trainer = DiTTrainer.__new__(DiTTrainer)
    trainer.base = types.SimpleNamespace(
        train_dataset=object(),
        args=types.SimpleNamespace(
            data=types.SimpleNamespace(
                dataloader=DataloaderConfig(
                    type="native",
                    num_workers=0,
                    drop_last=True,
                    pin_memory=False,
                    prefetch_factor=2,
                ),
                max_seq_len=1024 * 1024,
                dyn_bsz_buffer_size=200,
                dyn_bsz_buffer_policy="context_aware",
                data_modality="diffusion",
                shuffle=False,
            ),
            train=types.SimpleNamespace(
                micro_batch_size=1,
                global_batch_size=1,
                dataloader_batch_size=1,
                dyn_bsz=True,
                dyn_bsz_runtime="main",
                dyn_bsz_count_mode="total",
                dyn_bsz_physical_overflow_ratio=1.5,
                bsz_warmup_ratio=0.02,
                bsz_warmup_init_mbtoken=200,
                seed=0,
                checkpoint=types.SimpleNamespace(save_steps=1000),
            ),
            train_steps=1,
        ),
    )
    # DiTTrainer._setup forces fixed-batch loading before _build_dataloader.
    trainer.base.args.train.dyn_bsz = False

    with pytest.raises(ValueError, match="requires dyn_bsz=True"):
        trainer._build_dataloader()


@pytest.mark.parametrize("missing_argument", ["max_seq_len", "micro_batch_size"])
def test_build_dataloader_context_aware_policy_reports_missing_arguments(missing_argument):
    arguments = {
        "dataloader_type": "native",
        "dyn_bsz": True,
        "dyn_bsz_buffer_policy": "context_aware",
        "max_seq_len": 1024 * 1024,
        "micro_batch_size": 1,
    }
    del arguments[missing_argument]

    with pytest.raises(ValueError, match=rf"missing: {missing_argument}"):
        build_dataloader(**arguments)


@pytest.mark.parametrize("dp_size", [1, 2, 4, 8])
def test_build_dataloader_resolves_context_aware_buffer_for_any_dp_before_registry_dispatch(monkeypatch, dp_size):
    import veomni.data.data_loader as m_dl

    ps = _fake_ps(sp_size=1)
    ps.dp_size = dp_size
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)

    captured_kwargs = {}

    def external_builder(**kwargs):
        captured_kwargs.update(kwargs)

    test_loader_name = "external_context_aware_dp_test"
    DATALOADER_REGISTRY[test_loader_name] = external_builder
    try:
        build_dataloader(
            dataloader_type=test_loader_name,
            dyn_bsz=True,
            dyn_bsz_buffer_size=200,
            dyn_bsz_buffer_policy="context_aware",
            max_seq_len=1024 * 1024,
            micro_batch_size=1,
            dyn_bsz_runtime="main",
            dyn_bsz_count_mode="total",
            data_modality="text",
        )
    finally:
        del DATALOADER_REGISTRY[test_loader_name]

    assert captured_kwargs["dyn_bsz_buffer_size"] == 24


def test_text_batching_minimum_buffer_is_not_a_capacity():
    strategy = TextBatchingStrategy(token_micro_bsz=50, buffer_size=24)
    sample = {
        "input_ids": types.SimpleNamespace(shape=(2,)),
        "attention_mask": types.SimpleNamespace(sum=lambda: 2),
    }

    for _ in range(24):
        strategy.put_item(sample)

    assert not strategy.is_ready_for_micro_batch()

    strategy.put_item(sample)

    assert len(strategy.buffer) == 25
    assert strategy.is_ready_for_micro_batch()
    assert len(strategy.get_micro_batch(step=0)) == 25


def test_build_dataloader_preserves_legacy_fixed_plugin_arguments():
    captured_kwargs = {}

    def legacy_external_builder(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    test_loader_name = "legacy_fixed_test"
    DATALOADER_REGISTRY[test_loader_name] = legacy_external_builder
    try:
        build_dataloader(
            dataloader_type=test_loader_name,
            dyn_bsz=True,
            dyn_bsz_buffer_size=37,
        )
    finally:
        del DATALOADER_REGISTRY[test_loader_name]

    assert captured_kwargs == {"dyn_bsz": True, "dyn_bsz_buffer_size": 37}


def test_trainer_fixed_policy_supports_legacy_explicit_builder(monkeypatch):
    from veomni.trainer.base import BaseTrainer

    @dataclass
    class LoaderConfig:
        type: str = "legacy_explicit"

    received = {}

    def legacy_builder(
        dataset,
        micro_batch_size,
        global_batch_size,
        dataloader_batch_size,
        max_seq_len,
        train_steps,
        bsz_warmup_ratio,
        bsz_warmup_init_mbtoken,
        dyn_bsz,
        dyn_bsz_runtime,
        dyn_bsz_count_mode,
        dyn_bsz_physical_overflow_ratio,
        dyn_bsz_buffer_size,
        seed,
        collate_fn,
        save_steps,
    ):
        received.update(locals())
        return "legacy-loader"

    monkeypatch.setitem(DATALOADER_REGISTRY._local_mapping, "legacy_explicit", legacy_builder)
    trainer = types.SimpleNamespace(
        args=types.SimpleNamespace(
            data=types.SimpleNamespace(
                dataloader=LoaderConfig(),
                max_seq_len=128,
                dyn_bsz_buffer_size=37,
                dyn_bsz_buffer_policy="fixed",
                data_modality="text",
            ),
            train=types.SimpleNamespace(
                micro_batch_size=1,
                global_batch_size=2,
                dataloader_batch_size=1,
                bsz_warmup_ratio=0.0,
                bsz_warmup_init_mbtoken=1,
                dyn_bsz=True,
                dyn_bsz_runtime="main",
                dyn_bsz_count_mode="total",
                dyn_bsz_physical_overflow_ratio=1.5,
                seed=7,
                checkpoint=types.SimpleNamespace(save_steps=10),
            ),
            train_steps=3,
        ),
        train_dataset=[{"input_ids": [1]}],
        collate_fn=None,
    )
    BaseTrainer._build_dataloader(trainer)
    assert trainer.train_dataloader == "legacy-loader"
    assert received["dyn_bsz_buffer_size"] == 37
    assert received["train_steps"] == 3


@pytest.fixture(scope="session")
def dummy_dataset_ci():
    dummy = DummyDataset(size=40, num_shard=1, dataset_name="ci_dyn_bsz_shared")
    yield dummy
    dummy.clean_cache()


@pytest.mark.parametrize("dataset_name", ["iterable", "mapping"])
@pytest.mark.parametrize("dyn_bsz", [True, False])
@pytest.mark.parametrize("sp_size", [1, 2])
@pytest.mark.parametrize("dyn_bsz_runtime", ["main", "worker"])
def test_build_dataloader_dyn_bsz_sp_filling(
    monkeypatch,
    dummy_dataset_ci,
    dataset_name: str,
    dyn_bsz: bool,
    sp_size: int,
    dyn_bsz_runtime: Literal["main", "worker"],
):
    import veomni.data.data_collator as m_col
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=sp_size)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_col, "get_parallel_state", lambda: ps)

    global_batch_size = 8
    micro_batch_size = 2
    max_seq_len = 100

    if dyn_bsz:
        if dyn_bsz_runtime == "main":
            dataloader_batch_size = 1
        else:
            dataloader_batch_size = global_batch_size // micro_batch_size
    else:
        dataloader_batch_size = global_batch_size

    transform = partial(process_dummy_example, max_seq_len=max_seq_len)

    dataset = build_dataset(
        dataset_name=dataset_name,
        train_path=dummy_dataset_ci.save_path,
        transform=transform,
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        dataloader_batch_size=dataloader_batch_size,
        max_seq_len=max_seq_len,
        train_steps=1,
        num_workers=0,
        dyn_bsz=dyn_bsz,
        dyn_bsz_runtime=dyn_bsz_runtime,
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    micro_batches = next(iter(dl))

    if dyn_bsz:
        assert len(micro_batches) == global_batch_size // micro_batch_size
        for micro_batch in micro_batches:
            assert max_seq_len * (micro_batch_size - 1) <= sum(micro_batch["id"]) <= max_seq_len * micro_batch_size
    else:
        assert len(micro_batches) == global_batch_size // micro_batch_size
        for micro_batch in micro_batches:
            assert len(micro_batch["id"]) == micro_batch_size


@pytest.mark.parametrize("dyn_bsz_runtime", ["main", "worker"])
def test_build_dataloader_dyn_bsz_count_mode(
    monkeypatch, dummy_dataset_ci, dyn_bsz_runtime: Literal["main", "worker"]
):
    import veomni.data.data_collator as m_col
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_col, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1 if dyn_bsz_runtime == "main" else 2,
        max_seq_len=16,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime=dyn_bsz_runtime,
        dyn_bsz_count_mode="effective",
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    if dyn_bsz_runtime == "main":
        assert isinstance(dl, DynamicBatchSizeDataLoader)
        assert isinstance(dl.batching_strategy, TextBatchingStrategy)
        assert dl.batching_strategy.buffer._get_length_fn is m_ds.get_length_by_labels_fn
        assert dl.batching_strategy.physical_token_cap == 48
        assert dl.batching_strategy.buffer._get_physical_length_fn is m_ds.get_length_by_attention_mask_fn
    else:
        assert isinstance(dl.dataset, m_ds.DynamicBatchingSizeDataset)
        assert dl.dataset.get_length_fn is m_ds.get_length_by_labels_fn
        assert dl.dataset.physical_token_cap == 48
        assert dl.dataset.get_physical_length_fn is m_ds.get_length_by_attention_mask_fn


def test_build_dataloader_dyn_bsz_physical_overflow_ratio(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1,
        max_seq_len=16,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_count_mode="effective",
        dyn_bsz_physical_overflow_ratio=1.25,
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    assert dl.batching_strategy.physical_token_cap == 40


def test_build_dataloader_suppresses_persistent_workers_with_zero_workers(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1,
        max_seq_len=16,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        persistent_workers=True,
        in_order=False,
        seed=0,
    )

    assert dl._dataloader.persistent_workers is False
    assert dl._dataloader.in_order is False


def test_build_dataloader_forwards_worker_scheduling_kwargs(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    captured = {}

    class FakeDistributedDataloader:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_dl, "DistributedDataloader", FakeDistributedDataloader)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1,
        max_seq_len=16,
        train_steps=1,
        num_workers=2,
        dyn_bsz=False,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
        in_order=False,
        seed=0,
    )

    assert captured["persistent_workers"] is True
    assert captured["in_order"] is False


def test_build_dataloader_rejects_invalid_physical_overflow_ratio(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    with pytest.raises(ValueError, match="dyn_bsz_physical_overflow_ratio must be >= 1.0"):
        build_dataloader(
            "native",
            dataset=dataset,
            micro_batch_size=2,
            global_batch_size=4,
            dataloader_batch_size=1,
            max_seq_len=16,
            train_steps=1,
            num_workers=0,
            dyn_bsz=True,
            dyn_bsz_runtime="main",
            dyn_bsz_count_mode="effective",
            dyn_bsz_physical_overflow_ratio=0.5,
            dyn_bsz_buffer_size=1,
            drop_last=True,
            prefetch_factor=None,
            seed=0,
        )
