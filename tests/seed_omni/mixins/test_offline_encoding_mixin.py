"""``OfflineEncodingMixin``: the two offline-cache endpoints a cacheable module implements."""

from __future__ import annotations

import pytest
import torch

from veomni.models.seed_omni import OfflineEncodingMixin
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from veomni.models.seed_omni.utils.conversation import ConversationItem


class DummyOfflineModule(OfflineEncodingMixin, TrainingModuleMixin, BaseMixin):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        super().__init__()

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


def test_offline_encoding_mixin_requires_tensor_endpoints() -> None:
    assert OfflineEncodingMixin.__abstractmethods__ == {"offline_encode", "online_process"}


def test_offline_encoding_mixin_is_not_module_mixin_subclass() -> None:
    assert not issubclass(OfflineEncodingMixin, BaseMixin)


def test_a_module_missing_an_endpoint_cannot_be_built() -> None:
    class EncodeOnly(OfflineEncodingMixin):
        def offline_encode(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
            return {}

    with pytest.raises(TypeError, match="online_process"):
        EncodeOnly()


def test_offline_encoding_mixin_does_not_implement_decorated_hooks() -> None:
    markers = ("_omni_pre_context", "_omni_post_context")
    assert not [name for name, attr in vars(OfflineEncodingMixin).items() for m in markers if hasattr(attr, m)]


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
