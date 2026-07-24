from types import SimpleNamespace

import pytest
import torch

from veomni.arguments.arguments_types import ChannelLossConfig
from veomni.data.data_collator import MakeMicroBatchCollator
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks.channel_loss_callback import ChannelLossCallback
from veomni.trainer.dit_trainer import DiTTrainer


class _IdentityCollator:
    def __call__(self, features):
        return list(features)


def _canonical_metadata(source_id="stable-id"):
    return {
        "schema_version": 1,
        "coordinate_space": "packed_pre_sp",
        "valid_token_count": 1,
        "segments": [
            {
                "source_id": source_id,
                "segment_index": 0,
                "sample_index": 0,
                "subsegment_index": 0,
                "token_start": 0,
                "token_length": 1,
            }
        ],
    }


def test_fixed_batching_rejects_multiple_transform_parts_instead_of_dropping_them():
    collator = MakeMicroBatchCollator(num_micro_batch=1, internal_data_collator=_IdentityCollator())
    transformed = [[{"input_ids": torch.tensor([1])}, {"input_ids": torch.tensor([2])}]]

    with pytest.raises(ValueError, match="exactly one output per input"):
        collator(transformed)


def test_base_preforward_preserves_generic_model_inputs_without_source_contract():
    trainer = object.__new__(BaseTrainer)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(
        data=SimpleNamespace(enable_multisource=False),
        train=SimpleNamespace(chunk_mbs_config=None, local_rank=0),
    )
    trainer.LOG_SAMPLE = False
    micro_batch = {
        "x": torch.tensor(1),
        "source_id": torch.tensor([7]),
        "dataset_id": torch.tensor([8]),
        "sample_id": torch.tensor([9]),
    }

    result = BaseTrainer.preforward(trainer, micro_batch)

    assert set(result) == {"x", "source_id", "dataset_id", "sample_id"}


def test_base_preforward_strips_legacy_fields_owned_by_multisource_loader():
    trainer = object.__new__(BaseTrainer)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(
        data=SimpleNamespace(enable_multisource=True),
        train=SimpleNamespace(chunk_mbs_config=None, local_rank=0),
    )
    trainer.LOG_SAMPLE = False
    micro_batch = {
        "x": torch.tensor(1),
        "ds_idx": torch.tensor([0]),
        "source_name": ["train/a"],
        "cur_token_num": torch.tensor([1]),
    }

    result = BaseTrainer.preforward(trainer, micro_batch)

    assert set(result) == {"x"}


def test_dit_preforward_strips_canonical_metadata_before_logging_and_model_use(monkeypatch):
    import veomni.trainer.dit_trainer as dit_trainer_module

    trainer = object.__new__(DiTTrainer)
    trainer.base = SimpleNamespace(
        device=torch.device("cpu"),
        LOG_SAMPLE=True,
        args=SimpleNamespace(
            data=SimpleNamespace(enable_multisource=True),
            train=SimpleNamespace(local_rank=0),
        ),
    )
    logged_keys = []
    monkeypatch.setattr(
        dit_trainer_module.helper,
        "print_example",
        lambda *, example, **_kwargs: logged_keys.extend(example.keys()),
    )
    micro_batch = {
        "latents": torch.tensor([1.0]),
        "ds_idx": [0],
        "source_name": ["train/a"],
        "_veomni_packed_source_metadata": _canonical_metadata(),
    }

    result = DiTTrainer.preforward(trainer, micro_batch)

    assert logged_keys == ["latents"]
    assert set(result) == {"latents"}


def test_disabled_channel_callback_preserves_custom_extra_strip_keys():
    cfg = ChannelLossConfig(enable=False, extra_strip_keys=["model_aux"])
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
    callback = ChannelLossCallback(trainer)
    micro_batch = {
        "x": torch.tensor(1),
        "model_aux": torch.tensor(2),
        "source_id": ["stable-id"],
        "_veomni_packed_source_metadata": _canonical_metadata(),
    }

    callback.strip_model_inputs(micro_batch)

    assert set(micro_batch) == {"x", "model_aux"}
