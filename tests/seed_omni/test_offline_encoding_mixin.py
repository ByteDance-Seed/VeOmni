from __future__ import annotations

import inspect
from collections.abc import Collection

import pytest
import torch
from transformers import PretrainedConfig

from veomni.models.seed_omni import OfflineEncodingConfigMixin, OfflineEncodingMixin
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from veomni.models.seed_omni.utils.conversation import ConversationItem


class DummyOfflineConfig(OfflineEncodingConfigMixin, PretrainedConfig):
    model_type = "dummy_offline_config"

    def __init__(self, marker: str = "default", **kwargs: object) -> None:
        self.marker = marker
        super().__init__(**kwargs)


class DummyOfflineModule(OfflineEncodingMixin, TrainingModuleMixin, BaseMixin):
    def __init__(self, support_cache: bool = False, train_type: str = "train") -> None:
        self.config = DummyOfflineConfig(support_cache=support_cache, train_type=train_type)
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


@pytest.mark.parametrize(
    ("support_cache", "train_type", "expected"),
    [
        (False, "offline_cache", "full"),
        (True, "offline_cache", "encode_only"),
        (True, "train_with_cache", "process_only"),
        (True, "train", "full"),
    ],
)
def test_cache_mode_is_derived_from_support_cache_and_train_type(
    support_cache: bool, train_type: str, expected: str
) -> None:
    assert DummyOfflineModule(support_cache=support_cache, train_type=train_type).cache_mode == expected


def test_offline_encoding_config_mixin_consumes_hf_runtime_overrides(tmp_path) -> None:
    DummyOfflineConfig().save_pretrained(tmp_path)

    config, unused = DummyOfflineConfig.from_pretrained(
        tmp_path,
        support_cache=True,
        train_type="train_with_cache",
        return_unused_kwargs=True,
    )

    assert config.support_cache is True
    assert config.train_type == "train_with_cache"
    assert "support_cache" not in unused
    assert "train_type" not in unused


def test_pre_forward_rejects_process_only_for_offline_encode() -> None:
    module = DummyOfflineModule(support_cache=True, train_type="train_with_cache")

    with pytest.raises(
        ValueError, match="offline_encode requires cache_mode in .* current cache_mode is 'process_only'"
    ):
        module.pre_forward("offline_encode", conversation_list=[])


def test_pre_forward_rejects_encode_only_for_online_process() -> None:
    module = DummyOfflineModule(support_cache=True, train_type="offline_cache")

    with pytest.raises(
        ValueError, match="online_process requires cache_mode in .* current cache_mode is 'encode_only'"
    ):
        module.pre_forward("online_process", conversation_list=[])


def test_default_partial_dcp_hooks_are_noop() -> None:
    module = DummyOfflineModule(support_cache=True, train_type="train_with_cache")

    assert module.load_partial_dcp_checkpoint("/tmp/load", trainer=object()) is None
    assert module.save_partial_dcp_checkpoint("/tmp/save", trainer=object(), state=object()) is None


def test_default_full_hf_checkpoint_hook_requires_module_implementation() -> None:
    module = DummyOfflineModule(support_cache=True, train_type="train_with_cache")

    with pytest.raises(NotImplementedError, match="save_full_hf_checkpoint"):
        module.save_full_hf_checkpoint("/tmp/out", source_path="/tmp/source", trainer=object(), state=object())


def test_offline_encoding_mixin_is_not_module_mixin_subclass() -> None:
    assert not issubclass(OfflineEncodingMixin, BaseMixin)


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
