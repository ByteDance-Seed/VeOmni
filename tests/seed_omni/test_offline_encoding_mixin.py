from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from veomni.models.seed_omni import (
    ENCODED_CACHE_KIND_META_KEY,
    OfflineEncodedCache,
    OfflineEncodingMixin,
)
from veomni.models.seed_omni.mixins import offline_encoding
from veomni.models.seed_omni.mixins.modulemixin import ModuleMixin, post_forward, pre_forward
from veomni.models.seed_omni.utils.conversation import ConversationItem


class DummyEncodedCache(OfflineEncodedCache):
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self.tensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> DummyEncodedCache:
        return cls(tensor)


class DummyOfflineModule(OfflineEncodingMixin, ModuleMixin):
    def __init__(self, cache_mode: str | None = None, freeze: bool | None = None) -> None:
        self.config = SimpleNamespace()
        if cache_mode is not None:
            self.config.cache_mode = cache_mode
        if freeze is not None:
            self.config.freeze = freeze
        self.calls: list[str] = []
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        super().__init__()

    def init_omni_state(self) -> None:
        return None

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


def test_offline_encoded_cache_contract_is_tensor_first() -> None:
    payload = torch.tensor([1.0, 2.0])
    cache = DummyEncodedCache.from_tensor(payload)

    assert torch.equal(cache.to_tensor(), payload)
    assert ENCODED_CACHE_KIND_META_KEY == "encoded_cache_kind"


def test_missing_config_cache_mode_warns_about_full_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(offline_encoding.logger, "warning_rank0", warnings.append)

    assert DummyOfflineModule().cache_mode == "full"

    assert len(warnings) == 1
    assert "config does not define `cache_mode`" in warnings[0]
    assert "falling back to 'full'" in warnings[0]


def test_invalid_cache_mode_has_clear_error() -> None:
    module = DummyOfflineModule(cache_mode="invalid")

    with pytest.raises(ValueError, match="DummyOfflineModule.cache_mode must be one of .* got 'invalid'"):
        module.pre_forward("offline_encode", conversation_list=[])


def test_pre_forward_rejects_disallowed_cache_mode() -> None:
    module = DummyOfflineModule(cache_mode="process_only")

    with pytest.raises(
        ValueError, match="offline_encode requires cache_mode in .* current cache_mode is 'process_only'"
    ):
        module.pre_forward("offline_encode", conversation_list=[])


def test_init_rejects_declared_trainable_cache_module() -> None:
    with pytest.raises(ValueError, match="requires a frozen module"):
        DummyOfflineModule(freeze=False)


def test_default_partial_dcp_hooks_are_noop() -> None:
    module = DummyOfflineModule(cache_mode="process_only")

    assert module.load_partial_dcp_checkpoint("/tmp/load", trainer=object()) is None
    assert module.save_partial_dcp_checkpoint("/tmp/save", trainer=object(), state=object()) is None


def test_default_full_hf_checkpoint_hook_requires_module_implementation() -> None:
    module = DummyOfflineModule(cache_mode="process_only")

    with pytest.raises(NotImplementedError, match="save_full_hf_checkpoint"):
        module.save_full_hf_checkpoint("/tmp/out", source_path="/tmp/source", trainer=object(), state=object())


def test_offline_encoding_mixin_is_not_module_mixin_subclass() -> None:
    assert not issubclass(OfflineEncodingMixin, ModuleMixin)


def test_offline_encoding_mixin_does_not_implement_decorated_hooks() -> None:
    source = inspect.getsource(OfflineEncodingMixin)
    assert "@pre_forward" not in source
    assert "@post_forward" not in source


def test_decorated_hook_slots_can_bind_multiple_contexts() -> None:
    class MultiContextModule(ModuleMixin):
        def init_omni_state(self) -> None:
            return None

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
    module = DummyOfflineModule(cache_mode="full")
    conversation = [[ConversationItem(type="image", value=torch.ones(1), role="assistant")]]

    module.pre_forward("offline_encode", conversation_list=conversation)
    module.config.cache_mode = "invalid"
    module.pre_forward("offline_encode", conversation_list=conversation)

    with pytest.raises(ValueError, match="cache_mode must be one of .* got 'invalid'"):
        module.pre_forward("online_process", conversation_list=conversation)
