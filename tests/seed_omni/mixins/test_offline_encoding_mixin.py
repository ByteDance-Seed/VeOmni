from __future__ import annotations

import inspect
import json
from collections.abc import Collection

import pytest
import torch
from transformers import PretrainedConfig

from veomni.models.seed_omni import OfflineEncodingMixin
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from veomni.models.seed_omni.utils.conversation import ConversationItem


class DummyOfflineConfig(PretrainedConfig):
    model_type = "dummy_offline_config"

    def __init__(self, marker: str = "default", **kwargs: object) -> None:
        self.marker = marker
        super().__init__(**kwargs)


class DummyOfflineModule(OfflineEncodingMixin, TrainingModuleMixin, BaseMixin):
    def __init__(self, cache_mode: str = "full", support_cache: bool = True) -> None:
        self.config = DummyOfflineConfig(support_cache=support_cache)
        self.calls: list[str] = []
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        super().__init__(cache_mode=cache_mode)

    def offline_encode(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"encoded_cache": kwargs["pixel_values"]}

    def online_process(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"latents": kwargs["encoded_cache"]}

    @pre_forward("offline_encode")
    def offline_encode_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **batch: object,
    ) -> dict[str, torch.Tensor]:
        del batch
        self._conversation_carrier = conversation_list
        self.calls.append("offline_encode_pre")
        return {"pixel_values": torch.ones(1)}

    @post_forward("offline_encode")
    def offline_encode_post(self, **outputs: torch.Tensor) -> dict[str, list[list[ConversationItem]] | None]:
        del outputs
        self.calls.append("offline_encode_post")
        return {"conversation_list": self._conversation_carrier}

    @pre_forward("online_process")
    def online_process_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **batch: object,
    ) -> dict[str, torch.Tensor]:
        del batch
        self._conversation_carrier = conversation_list
        self.calls.append("online_process_pre")
        return {"encoded_cache": torch.ones(1)}

    @post_forward("online_process")
    def online_process_post(self, **outputs: torch.Tensor) -> dict[str, list[list[ConversationItem]] | None]:
        del outputs
        self.calls.append("online_process_post")
        return {"conversation_list": self._conversation_carrier}


@pytest.mark.parametrize("cache_mode", ["full", "encode_only", "process_only"])
def test_cache_mode_is_taken_from_the_constructor(cache_mode: str) -> None:
    assert DummyOfflineModule(cache_mode=cache_mode).cache_mode == cache_mode


def test_cache_mode_defaults_to_full() -> None:
    assert DummyOfflineModule().cache_mode == "full"


def test_unknown_cache_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="cache_mode must be one of"):
        DummyOfflineModule(cache_mode="offline_cache")


def test_cached_mode_requires_support_cache() -> None:
    assert DummyOfflineModule(cache_mode="full", support_cache=False).cache_mode == "full"
    with pytest.raises(ValueError, match="requires DummyOfflineConfig.support_cache=True"):
        DummyOfflineModule(cache_mode="encode_only", support_cache=False)


def test_pre_forward_rejects_process_only_for_offline_encode() -> None:
    module = DummyOfflineModule(cache_mode="process_only")

    with pytest.raises(
        ValueError, match="offline_encode requires cache_mode in .* current cache_mode is 'process_only'"
    ):
        module.pre_forward("offline_encode", conversation_list=[])


def test_pre_forward_rejects_encode_only_for_online_process() -> None:
    module = DummyOfflineModule(cache_mode="encode_only")

    with pytest.raises(
        ValueError, match="online_process requires cache_mode in .* current cache_mode is 'encode_only'"
    ):
        module.pre_forward("online_process", conversation_list=[])


def test_default_partial_dcp_hooks_are_noop() -> None:
    module = DummyOfflineModule(cache_mode="process_only")

    assert module.load_partial_dcp_checkpoint("/tmp/load", trainer=object()) is None
    assert module.save_partial_dcp_checkpoint("/tmp/save", trainer=object(), state=object()) is None


def test_default_full_hf_checkpoint_hook_requires_module_implementation() -> None:
    module = DummyOfflineModule(cache_mode="process_only")

    with pytest.raises(NotImplementedError, match="save_full_hf_checkpoint"):
        module.save_full_hf_checkpoint("/tmp/out", source_path="/tmp/source", trainer=object(), state=object())


def test_offline_encoding_mixin_requires_tensor_endpoints() -> None:
    source = inspect.getsource(OfflineEncodingMixin)
    assert "@abstractmethod" in source
    assert "def offline_encode" in source
    assert "def online_process" in source


def test_offline_encoding_mixin_is_not_module_mixin_subclass() -> None:
    assert not issubclass(OfflineEncodingMixin, BaseMixin)


def test_cache_mode_is_set_before_the_model_body_and_kept_off_the_config(tmp_path) -> None:
    """The native body branches on ``cache_mode`` to decide which sub-networks
    to build at all (e.g. a VAE in ``encode_only`` never allocates its decoder),
    so it must be visible inside the body, yet it is a per-run choice and must
    not end up in ``config.json``.
    """
    seen: list[str] = []

    class NativeBody:
        def __init__(self, config: DummyOfflineConfig) -> None:
            seen.append(self.cache_mode)
            self.config = config

    class Module(OfflineEncodingMixin, NativeBody):
        def offline_encode(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
            return {}

        def online_process(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
            return {}

    module = Module(DummyOfflineConfig(support_cache=True), cache_mode="encode_only")

    assert seen == ["encode_only"]  # visible to the native body, not only afterwards
    assert module.cache_mode == "encode_only"
    module.config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["support_cache"] is True
    assert "cache_mode" not in saved


def test_sibling_offline_mixin_wins_mro_over_the_abstract_stubs() -> None:
    """A module's concrete tensor call-sites must sit before this mixin in MRO.

    ``offline_encode`` / ``online_process`` are abstract here; a module supplies
    them from a sibling ``*OfflineMixin`` listed *first* in its bases, so the
    real implementation resolves instead of the stub.
    """

    class SiblingOfflineMixin:
        def offline_encode(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"encoded_cache": kwargs["pixel_values"] * 2}

        def online_process(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"latents": kwargs["encoded_cache"]}

    class Module(SiblingOfflineMixin, OfflineEncodingMixin):
        def __init__(self) -> None:
            self.config = DummyOfflineConfig(support_cache=True)

    assert Module.__mro__.index(SiblingOfflineMixin) < Module.__mro__.index(OfflineEncodingMixin)
    encoded_cache = Module().offline_encode(pixel_values=torch.ones(1))["encoded_cache"]
    assert torch.equal(encoded_cache, torch.full((1,), 2.0))


def test_offline_encoding_mixin_does_not_implement_decorated_hooks() -> None:
    source = inspect.getsource(OfflineEncodingMixin)
    assert "@pre_forward" not in source
    assert "@post_forward" not in source


def test_decorated_hook_slots_can_bind_multiple_contexts() -> None:
    class MultiContextModule(TrainingModuleMixin, BaseMixin):
        @pre_forward("encode", "offline_encode")
        def encode_pre(self, **kwargs: object) -> dict[str, object]:
            return {"seen": kwargs["seen"]}

        @post_forward("encode", "offline_encode")
        def encode_post(self, **outputs: object) -> dict[str, object]:
            return {"done": outputs["done"]}

    module = MultiContextModule()

    assert module.pre_forward("encode", seen=1) == {"seen": 1}
    assert module.pre_forward("offline_encode", seen=2) == {"seen": 2}
    assert module.post_forward("encode", done=3) == {"done": 3}
    assert module.post_forward("offline_encode", done=4) == {"done": 4}


def test_decorated_hook_slots_are_dispatched_by_module_mixin() -> None:
    module = DummyOfflineModule()
    conversation = [[ConversationItem(type="image", value=torch.ones(1), role="assistant")]]

    assert module.pre_forward("offline_encode", conversation_list=conversation) == {"pixel_values": torch.ones(1)}
    assert module.post_forward("offline_encode", encoded_cache=torch.ones(1)) == {"conversation_list": conversation}
    assert module.pre_forward("online_process", conversation_list=conversation) == {"encoded_cache": torch.ones(1)}
    assert module.post_forward("online_process", latents=torch.ones(1)) == {"conversation_list": conversation}
    assert module.calls == [
        "offline_encode_pre",
        "offline_encode_post",
        "online_process_pre",
        "online_process_post",
    ]


def test_cache_mode_is_checked_once_per_offline_method() -> None:
    module = DummyOfflineModule(support_cache=False)
    conversation = [[ConversationItem(type="image", value=torch.ones(1), role="assistant")]]
    calls: list[str] = []

    original = module._check_cache_mode

    def wrapped_check_cache_mode(*, method: str, allowed: Collection[str]) -> None:
        calls.append(method)
        return original(method=method, allowed=allowed)

    module._check_cache_mode = wrapped_check_cache_mode  # type: ignore[method-assign]

    module.pre_forward("offline_encode", conversation_list=conversation)
    module.pre_forward("offline_encode", conversation_list=conversation)
    module.pre_forward("online_process", conversation_list=conversation)
    module.pre_forward("online_process", conversation_list=conversation)

    assert calls == ["offline_encode", "online_process"]
