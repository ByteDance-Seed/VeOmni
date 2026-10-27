# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DPO packing provenance must preserve physical segments and source boundaries."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_collator import MainCollator
from veomni.data.data_transform import process_dpo_example
from veomni.data.dpo_collator import DPOProvenanceCollator
from veomni.utils.dpo_provenance import (
    DPO_APPROVED_KEY,
    DPO_FEATURE_KEY,
    DPO_POSTCHECK_KEY,
    DPO_PROOF_KEY,
    DPOInvalidProof,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(1, len(text) + 1))


def test_dpo_config_keeps_existing_positional_argument_order():
    from veomni.trainer.text_dpo_trainer import DPOConfig

    config = DPOConfig(0.1, 0.2)
    assert config.beta == 0.1 and config.label_smoothing == 0.2
    assert config.explicit_segment_provenance is False


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_adds_transform_option_only_when_enabled(monkeypatch, enabled):
    import veomni.trainer.text_dpo_trainer as dpo

    captured = {}
    monkeypatch.setattr(dpo, "build_data_transform", lambda name, **kwargs: captured.update(kwargs))
    trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(
            dpo_config=SimpleNamespace(explicit_segment_provenance=enabled), data=SimpleNamespace(max_seq_len=64)
        )
    )
    trainer.policy_model = SimpleNamespace(tokenizer=Tokenizer(), chat_template=None)
    trainer._build_data_transform()
    if enabled:
        assert captured["record_segment_provenance"] is True
    else:
        assert "record_segment_provenance" not in captured


def _state(sp_enabled=False, rank=0):
    return SimpleNamespace(sp_enabled=sp_enabled, sp_size=2 if sp_enabled else 1, sp_rank=rank)


def _feature(chosen, rejected, source):
    feature = process_dpo_example(
        {"chosen": chosen, "rejected": rejected}, tokenizer=Tokenizer(), record_segment_provenance=True
    )[0]
    feature["ds_idx"] = source
    return feature


@pytest.mark.parametrize("sp_enabled,rank", [(False, 0), (True, 0), (True, 1)])
@pytest.mark.parametrize("pad_to_length", [0, 8])
def test_builtin_segments_are_proven_with_single_token_and_tail(monkeypatch, sp_enabled, rank, pad_to_length):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state(sp_enabled, rank))
    feature = _feature("abc", "d", 5)
    batch = DPOProvenanceCollator(MainCollator(pad_to_length=pad_to_length))([feature])
    proof = batch[DPO_PROOF_KEY]
    assert proof.segment_lengths == (3, 1)
    assert proof.feature_token_counts == (4,)
    assert proof.collator_tail == (4 if pad_to_length else 0)
    assert DPO_FEATURE_KEY not in batch


def test_unknown_or_inconsistent_feature_never_creates_proof(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    feature = _feature("abc", "d", 5)
    legacy = {k: v for k, v in feature.items() if k != DPO_FEATURE_KEY}
    batch = DPOProvenanceCollator(MainCollator())([feature, legacy])
    assert isinstance(batch[DPO_PROOF_KEY], DPOInvalidProof)
    assert DPO_FEATURE_KEY not in batch


@pytest.mark.parametrize("case", ["missing_first", "missing_second", "missing_all", "invalid_id"])
def test_multisource_requires_one_ordered_id_per_feature(monkeypatch, case):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    features = [_feature("abc", "de", 1), _feature("fgh", "ijk", 2)]
    if case in ("missing_first", "missing_all"):
        features[0].pop("ds_idx")
    if case in ("missing_second", "missing_all"):
        features[1].pop("ds_idx")
    if case == "invalid_id":
        features[0]["ds_idx"] = "not-a-number"
    batch = DPOProvenanceCollator(MainCollator(), require_ds_idx=True)(features)
    assert isinstance(batch[DPO_PROOF_KEY], DPOInvalidProof)


@pytest.mark.parametrize("masked_value", [0, 2])
def test_non_all_ones_attention_mask_is_not_approved(monkeypatch, masked_value):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    feature = _feature("abc", "d", 0)
    feature["attention_mask"][0] = masked_value
    batch = DPOProvenanceCollator(MainCollator(), require_ds_idx=True)([feature])
    assert isinstance(batch[DPO_PROOF_KEY], DPOInvalidProof)


@pytest.mark.parametrize("key", ["input_ids", "labels", "position_ids"])
def test_noninteger_token_or_position_tensor_is_not_approved(monkeypatch, key):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    feature = _feature("abc", "d", 0)
    feature[key] = feature[key].float()
    batch = DPOProvenanceCollator(MainCollator())([feature])
    assert isinstance(batch[DPO_PROOF_KEY], DPOInvalidProof)


@pytest.mark.parametrize("key,value", [("input_ids", -1), ("labels", -2), ("position_ids", -1)])
def test_invalid_negative_token_or_position_is_not_approved(monkeypatch, key, value):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    feature = _feature("abc", "d", 0)
    feature[key][-1] = value
    batch = DPOProvenanceCollator(MainCollator())([feature])
    assert isinstance(batch[DPO_PROOF_KEY], DPOInvalidProof)


def test_collator_does_not_mutate_input_labels(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    features = [_feature("abc", "de", 0), _feature("fgh", "ijk", 1)]
    labels_before = [feature["labels"].clone() for feature in features]
    batch = DPOProvenanceCollator(MainCollator())(features)
    for feature, old_labels in zip(features, labels_before):
        torch.testing.assert_close(feature["labels"], old_labels, rtol=0, atol=0)
    assert batch["labels"][0, labels_before[0].numel()] == -100


@pytest.mark.parametrize("key", [DPO_PROOF_KEY, DPO_APPROVED_KEY, DPO_POSTCHECK_KEY])
def test_reserved_provenance_keys_cannot_be_overwritten(monkeypatch, key):
    import veomni.data.data_collator as m
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    feature = _feature("abc", "d", 0)
    feature[key] = torch.tensor([314159])
    with pytest.raises(ValueError, match="reserved"):
        DPOProvenanceCollator(MainCollator())([feature])
    assert feature[key].item() == 314159
    if key in (DPO_APPROVED_KEY, DPO_POSTCHECK_KEY):
        feature.pop(key)
        batch = DPOProvenanceCollator(MainCollator())([feature])
        batch[key] = torch.tensor([314159])
        with pytest.raises(RuntimeError, match="reserved"):
            next_dpo_microbatches(iter([[batch]]), state, state, global_step=1)
        assert batch[key].item() == 314159


def test_all_legacy_features_are_not_approved_by_opt_in_collator(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    feature = _feature("abc", "d", 5)
    feature.pop(DPO_FEATURE_KEY)
    batch = DPOProvenanceCollator(MainCollator())([feature])
    assert isinstance(batch[DPO_PROOF_KEY], DPOInvalidProof)


@pytest.mark.parametrize(
    "changed",
    [
        "labels",
        "source_id",
        "source_range",
        "channel_provenance",
        "cu_k",
        "linear_cu",
        "slices",
        "max_length",
        "mask_flag",
    ],
)
def test_preflight_rejects_changed_batch_or_out_of_range_source(monkeypatch, changed):
    import veomni.data.data_collator as m
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    batch = DPOProvenanceCollator(MainCollator(channel_loss_source_id_keys=("ds_idx",)), require_ds_idx=True)(
        [_feature("abc", "de", 0)]
    )
    if changed == "labels":
        batch["labels"] = torch.cat([batch["labels"], batch["labels"][:, -1:]], dim=-1)
    elif changed == "source_id":
        batch["ds_idx"][0] = 2
    elif changed == "source_range":
        batch = DPOProvenanceCollator(MainCollator(), require_ds_idx=True)([_feature("abc", "de", 42)])
    elif changed == "channel_provenance":
        batch[m.CHANNEL_LOSS_PROVENANCE]["input_mask"][0] = 0
    elif changed == "cu_k":
        batch["cu_seq_lens_k"] = torch.tensor([0, 5])
    elif changed == "linear_cu":
        batch["linear_attn_cu_seq_lens_q"] = torch.tensor([0, 5])
    elif changed == "slices":
        batch["packed_sequence_slices"] = ((0, 5),)
    elif changed == "max_length":
        batch["max_length_q"] = 1
    elif changed == "mask_flag":
        batch["attention_mask_is_all_ones"] = False
    with pytest.raises(RuntimeError, match="metadata failed"):
        next_dpo_microbatches(iter([[batch]]), state, state, global_step=1, source_count=lambda: 1)


@pytest.mark.parametrize("key", ["input_ids", "labels"])
def test_preflight_rejects_token_ids_outside_vocabulary(monkeypatch, key):
    import veomni.data.data_collator as m
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    feature = _feature("abc", "de", 0)
    feature[key][-1] = 100
    batch = DPOProvenanceCollator(MainCollator())([feature])
    with pytest.raises(RuntimeError, match="outside the model vocabulary"):
        next_dpo_microbatches(iter([[batch]]), state, state, global_step=1, vocab_size=lambda: 10)


def test_preflight_rejects_negative_collator_padding(monkeypatch):
    import veomni.data.data_collator as m
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    collator = MainCollator(data_collate_info={"input_ids": (-1, True, -1, 1)}, pad_to_length=8)
    batch = DPOProvenanceCollator(collator)([_feature("abc", "def", 0)])
    with pytest.raises(RuntimeError, match="custom token packing or label padding"):
        next_dpo_microbatches(iter([[batch]]), state, state, global_step=1, vocab_size=lambda: 100)


def test_nonstandard_sp_attention_mask_slicing_is_not_approved(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state(sp_enabled=True))
    collator = MainCollator(data_collate_info={"attention_mask": (-1, True, 1, 1)})
    batch = DPOProvenanceCollator(collator)([_feature("abc", "def", 0)])
    assert isinstance(batch[DPO_PROOF_KEY], DPOInvalidProof)


def test_modified_packing_configuration_is_rejected(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    collator = MainCollator()
    collator.preforward_pipeline[1].collate_infos = dict(collator.collate_infos)
    with pytest.raises(ValueError, match="standard packing configuration"):
        DPOProvenanceCollator(collator)([_feature("abc", "def", 0)])


def test_preflight_rejects_source_name_not_matching_catalog(monkeypatch):
    import veomni.data.data_collator as m
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    feature = _feature("abc", "def", 0)
    feature["source_name"] = "wrong-source"
    batch = DPOProvenanceCollator(MainCollator(), require_ds_idx=True)([feature])
    with pytest.raises(RuntimeError, match="source names disagree"):
        next_dpo_microbatches(
            iter([[batch]]), state, state, global_step=1, source_catalog=lambda: ("expected-source",)
        )


def test_preflight_rejects_consistently_wrong_gdn_metadata(monkeypatch):
    import veomni.data.data_collator as m

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    collator = MainCollator()

    def corrupt_metadata(batch):
        batch["linear_attn_cu_seq_lens_q"] = torch.tensor([0, 2, 6])
        return batch

    collator.preforward_pipeline.append(corrupt_metadata)
    with pytest.raises(ValueError, match="standard collator pipeline"):
        DPOProvenanceCollator(collator)([_feature("abc", "def", 0)])


def test_out_of_range_source_rejects_train_step_before_optimizer(monkeypatch):
    import veomni.data.data_collator as m
    import veomni.trainer.text_dpo_trainer as dpo

    parallel = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: parallel)
    batch = DPOProvenanceCollator(MainCollator(), require_ds_idx=True)([_feature("abc", "de", 42)])
    trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(
            dpo_config=SimpleNamespace(explicit_segment_provenance=True),
            data=SimpleNamespace(enable_multisource=True),
        ),
        state=SimpleNamespace(global_step=0, epoch=0),
    )
    trainer.environ_meter = SimpleNamespace(multisource_tracker=SimpleNamespace(names=["only_source"]))
    trainer.policy_model = SimpleNamespace(
        parallel_state=parallel,
        optimizer=MagicMock(),
        lr_scheduler=MagicMock(),
        model_config=SimpleNamespace(vocab_size=20),
    )
    trainer.reference_model = SimpleNamespace(parallel_state=parallel)
    with pytest.raises(RuntimeError, match="source ID is outside"):
        trainer.train_step(iter([[batch]]))
    trainer.policy_model.optimizer.step.assert_not_called()
    trainer.policy_model.lr_scheduler.step.assert_not_called()


@pytest.mark.parametrize("sp_enabled", [False, True])
def test_classification_collator_never_creates_proof(monkeypatch, sp_enabled):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state(sp_enabled))
    feature = _feature("abc", "def", 5)
    batch = DPOProvenanceCollator(MainCollator(seq_classification=True))([feature])
    assert isinstance(batch[DPO_PROOF_KEY], DPOInvalidProof)


@pytest.mark.parametrize("field", ["labels", "position_ids", "attention_mask"])
def test_inconsistent_input_lengths_never_create_proof(monkeypatch, field):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    feature = _feature("abc", "d", 5)
    feature[field] = torch.cat([feature[field], feature[field][-1:]])
    batch = DPOProvenanceCollator(MainCollator())([feature])
    assert isinstance(batch[DPO_PROOF_KEY], DPOInvalidProof)


@pytest.mark.parametrize("record_provenance", [False, True])
def test_truncated_plaintext_prompt_rejects_lost_completions(monkeypatch, record_provenance):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    with pytest.raises(ValueError, match="no scoreable completion tokens"):
        process_dpo_example(
            {"prompt": "abcde", "chosen": "f", "rejected": "g"},
            tokenizer=Tokenizer(),
            max_seq_len=3,
            record_segment_provenance=record_provenance,
        )
    feature = process_dpo_example(
        {"prompt": "ab", "chosen": "c", "rejected": "d"},
        tokenizer=Tokenizer(),
        max_seq_len=3,
        record_segment_provenance=record_provenance,
    )[0]
    assert feature["labels"].tolist() == [-100, -100, 3, -100, -100, 3]
    if record_provenance:
        batch = DPOProvenanceCollator(MainCollator())([feature])
        assert batch[DPO_PROOF_KEY].segment_lengths == (3, 3)


@pytest.mark.parametrize("record_provenance", [False, True])
def test_plaintext_prompt_completion_boundary_merge_is_scoreable(record_provenance):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers import models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    backend = BackendTokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["Hello world", "Hello there"] * 40,
        trainers.BpeTrainer(
            vocab_size=320,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )

    prompt, chosen, rejected = (backend.encode(text).ids for text in ("Hello ", "Hello world", "Hello there"))
    assert len(prompt) == len(chosen) == len(rejected) == 2
    assert prompt[0] == chosen[0] == rejected[0] and prompt[1] != chosen[1] != rejected[1]

    feature = process_dpo_example(
        {"prompt": "Hello ", "chosen": "world", "rejected": "there"},
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend),
        max_seq_len=2048,
        record_segment_provenance=record_provenance,
    )[0]
    assert feature["input_ids"].tolist() == chosen + rejected
    assert feature["labels"].tolist() == [-100, chosen[1], -100, rejected[1]]


def test_offsets_mask_prompt_only_tokens_when_bpe_resegments_earlier():
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers import models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {
        token: idx
        for idx, token in enumerate(("[UNK]", "a", "k", "i", "g", "d", "v", "ak", "ig", "igg", "ga", "ad", "adv"))
    }
    backend = BackendTokenizer(
        models.BPE(
            vocab=vocab,
            merges=[("a", "d"), ("a", "k"), ("i", "g"), ("g", "a"), ("ig", "g"), ("ad", "v")],
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    feature = process_dpo_example(
        {"prompt": "akigga", "chosen": "dv", "rejected": "d"},
        tokenizer=tokenizer,
        max_seq_len=128,
    )[0]
    assert tokenizer.encode("akigga") == [vocab["ak"], vocab["ig"], vocab["ga"]]
    assert feature["input_ids"].tolist() == [
        vocab["ak"],
        vocab["igg"],
        vocab["adv"],
        vocab["ak"],
        vocab["igg"],
        vocab["ad"],
    ]
    assert feature["labels"].tolist() == [-100, -100, vocab["adv"], -100, -100, vocab["ad"]]


def test_offsets_accept_overlapping_unicode_bytes_and_keep_trailing_eos():
    class UnicodeTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3, 4, 99]

        def __call__(
            self, text, add_special_tokens=True, return_offsets_mapping=False, return_special_tokens_mask=False
        ):
            return {
                "input_ids": [1, 2, 3, 4, 99],
                "offset_mapping": [(0, 1), (1, 2), (1, 2), (1, 2), (0, 0)],
                "special_tokens_mask": [0, 0, 0, 0, 1],
            }

    feature = process_dpo_example(
        {"prompt": "a", "chosen": "你", "rejected": "好"},
        tokenizer=UnicodeTokenizer(),
    )[0]
    assert feature["labels"].tolist() == [-100, 2, 3, 4, 99, -100, 2, 3, 4, 99]


def test_bytelevel_trimmed_zero_width_completion_token_is_supervised():
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers import models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    backend = BackendTokenizer(models.BPE(vocab={"[UNK]": 0, "x": 1, "Ġ": 2, "y": 3}, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    feature = process_dpo_example({"prompt": "x", "chosen": " y", "rejected": " "}, tokenizer=tokenizer)[0]
    assert feature["input_ids"].tolist() == [1, 2, 3, 1, 2]
    assert feature["labels"].tolist() == [-100, 2, 3, -100, 2]
    with pytest.raises(ValueError, match="zero-width token lies at the prompt boundary"):
        process_dpo_example({"prompt": "x ", "chosen": "y", "rejected": "y"}, tokenizer=tokenizer)


def test_eos_alone_does_not_make_a_merged_completion_scoreable():
    class MergedFirstToken:
        def encode(self, text, add_special_tokens=True):
            return [1, 99]

        def __call__(
            self, text, add_special_tokens=True, return_offsets_mapping=False, return_special_tokens_mask=False
        ):
            return {"input_ids": [1, 99], "offset_mapping": [(0, 2), (0, 0)], "special_tokens_mask": [0, 1]}

    with pytest.raises(ValueError, match="no scoreable completion tokens"):
        process_dpo_example({"prompt": "a", "chosen": "b", "rejected": "c"}, tokenizer=MergedFirstToken())


def test_offsetless_tokenizer_rejects_an_unprovable_merged_boundary():
    class SlowBoundaryTokenizer:
        def encode(self, text, add_special_tokens=True):
            return {"Hello ": [1, 2], "Hello world": [1, 3], "Hello there": [1, 4]}[text]

    with pytest.raises(ValueError, match="requires tokenizer offset mapping"):
        process_dpo_example(
            {"prompt": "Hello ", "chosen": "world", "rejected": "there"},
            tokenizer=SlowBoundaryTokenizer(),
        )


def test_slow_tokenizer_with_automatic_trailing_eos_keeps_completion():
    from transformers import PreTrainedTokenizer

    class SlowTokenizer(PreTrainedTokenizer):
        def __init__(self):
            self.vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3, "world": 4, "there": 5}
            self.reverse = {idx: token for token, idx in self.vocab.items()}
            super().__init__(unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]")

        @property
        def vocab_size(self):
            return len(self.vocab)

        def get_vocab(self):
            return self.vocab

        def _tokenize(self, text, **kwargs):
            return text.split()

        def _convert_token_to_id(self, token):
            return self.vocab.get(token, 0)

        def _convert_id_to_token(self, index):
            return self.reverse.get(index, "[UNK]")

        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            return [1, *token_ids_0, 2]

        def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
            if already_has_special_tokens:
                return [int(token in (1, 2)) for token in token_ids_0]
            return [1, *([0] * len(token_ids_0)), 1]

    feature = process_dpo_example(
        {"prompt": "hello ", "chosen": "world", "rejected": "there"}, tokenizer=SlowTokenizer()
    )[0]
    assert feature["labels"].tolist() == [-100, -100, 4, 2, -100, -100, 5, 2]
    literal_eos = process_dpo_example(
        {"prompt": "hello[EOS]", "chosen": "world", "rejected": "there"}, tokenizer=SlowTokenizer()
    )[0]
    assert literal_eos["labels"].tolist() == [-100, -100, -100, 4, 2, -100, -100, -100, 5, 2]
    completion_eos = process_dpo_example(
        {"prompt": "hello", "chosen": "[EOS]world", "rejected": "[EOS]there"}, tokenizer=SlowTokenizer()
    )[0]
    assert completion_eos["labels"].tolist() == [-100, -100, 2, 4, 2, -100, -100, 2, 5, 2]
    truncated_eos = process_dpo_example(
        {"prompt": "hello", "chosen": "[EOS]world", "rejected": "[EOS]there"},
        tokenizer=SlowTokenizer(),
        max_seq_len=3,
    )[0]
    assert truncated_eos["labels"].tolist() == [-100, -100, 2, -100, -100, 2]
    for empty_completion in ("", "   "):
        with pytest.raises(ValueError, match="no scoreable completion"):
            process_dpo_example(
                {"prompt": "hello", "chosen": empty_completion, "rejected": "world"}, tokenizer=SlowTokenizer()
            )


def test_prepadding_is_not_reinterpreted_or_discarded(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    feature = _feature("abc", "def", 5)
    # A producer may include ignored tokens within a physical segment.
    feature["labels"][-2:] = -100
    batch = DPOProvenanceCollator(MainCollator(pad_to_length=8))([feature])
    proof = batch[DPO_PROOF_KEY]
    assert proof.segment_lengths == (3, 3)
    assert proof.collator_tail == 2
    assert batch["labels"].flatten().tolist()[-4:] == [-100, -100, -100, -100]


def test_dpo_single_token_pair_and_collator_tail_preserve_loss_gradients(monkeypatch):
    import veomni.data.data_collator as m
    import veomni.trainer.text_dpo_trainer as dpo
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    monkeypatch.setattr(dpo, "get_parallel_state", lambda: state)
    features = [_feature("abc", "def", 1), _feature("g", "h", 2)]
    batch = DPOProvenanceCollator(MainCollator(pad_to_length=10))(features)
    approved = next_dpo_microbatches(iter([[batch]]), state, state, global_step=1)[0]
    next_dpo_microbatches(
        iter([[approved]]),
        state,
        state,
        global_step=1,
        after_callbacks=True,
        expected_proofs=(approved[DPO_APPROVED_KEY],),
    )
    assert approved[DPO_APPROVED_KEY].segment_lengths == (3, 3, 1, 1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.nn.Parameter(-torch.arange(1, 11, dtype=torch.float32))

        def forward(self, input_ids, **kwargs):
            assert all(key not in kwargs for key in (DPO_PROOF_KEY, DPO_APPROVED_KEY, DPO_POSTCHECK_KEY))
            assert "ds_idx" not in kwargs
            return SimpleNamespace(fused_linear_aux=SimpleNamespace(log_probs=self.values[: input_ids.size(-1)][None]))

    trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(dpo_config=SimpleNamespace(average_log_prob=False, explicit_segment_provenance=True))
    )
    trainer.post_forward = m.PostCollator()
    model = Model()
    chosen, rejected = trainer.concatenated_forward(model, approved)
    loss, _, _ = trainer.dpo_loss(chosen, rejected, torch.zeros_like(chosen), torch.zeros_like(rejected), beta=0.1)
    loss.mean().backward()

    reference = model.values.detach().clone().requires_grad_()
    expected_chosen = torch.stack((reference[0] + reference[1], reference[6] * 0))
    expected_rejected = torch.stack((reference[3] + reference[4], reference[7] * 0))
    expected_loss = -F.logsigmoid(0.1 * (expected_chosen - expected_rejected)).mean()
    expected_loss.backward()
    torch.testing.assert_close(chosen, expected_chosen, rtol=0, atol=0)
    torch.testing.assert_close(rejected, expected_rejected, rtol=0, atol=0)
    torch.testing.assert_close(loss.mean(), expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(model.values.grad, reference.grad, rtol=0, atol=0)


def test_chunk_logprobs_matches_independent_dense_dpo_reference(monkeypatch):
    import veomni.data.data_collator as m
    import veomni.models.loss_utils.chunk_logprobs as chunk
    import veomni.trainer.text_dpo_trainer as dpo
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    monkeypatch.setattr(dpo, "get_parallel_state", lambda: state)
    monkeypatch.setattr(chunk, "get_parallel_state", lambda: state)
    features = [_feature("abc", "def", 1), _feature("g", "h", 2)]
    batch = DPOProvenanceCollator(MainCollator(pad_to_length=10))(features)
    approved = next_dpo_microbatches(iter([[batch]]), state, state, global_step=1)[0]
    next_dpo_microbatches(
        iter([[approved]]),
        state,
        state,
        global_step=1,
        after_callbacks=True,
        expected_proofs=(approved[DPO_APPROVED_KEY],),
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.head = torch.nn.Linear(4, 16, bias=False)

        def forward(self, input_ids, labels, **kwargs):
            log_probs, _ = chunk.chunk_logprobs_function(
                self.embedding(input_ids), self.head.weight, labels, chunk_size=5
            )
            return SimpleNamespace(fused_linear_aux=SimpleNamespace(log_probs=log_probs))

    torch.manual_seed(7)
    model = TinyModel()
    reference = copy.deepcopy(model)
    trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(dpo_config=SimpleNamespace(average_log_prob=False, explicit_segment_provenance=True))
    )
    trainer.post_forward = m.PostCollator()
    chosen, rejected = trainer.concatenated_forward(model, approved)
    loss = -F.logsigmoid(0.1 * (chosen - rejected)).mean()
    loss.backward()

    tokens = approved["input_ids"][0]
    reference_logps = []
    for start, end in ((0, 3), (3, 6), (6, 7), (7, 8)):
        sequence = tokens[start:end]
        if len(sequence) == 1:
            reference_logps.append(reference.head(reference.embedding(sequence)).sum() * 0)
        else:
            logits = reference.head(reference.embedding(sequence[:-1]))
            reference_logps.append(F.log_softmax(logits, dim=-1).gather(-1, sequence[1:, None]).sum())
    ref_chosen = torch.stack((reference_logps[0], reference_logps[2]))
    ref_rejected = torch.stack((reference_logps[1], reference_logps[3]))
    ref_loss = -F.logsigmoid(0.1 * (ref_chosen - ref_rejected)).mean()
    ref_loss.backward()
    torch.testing.assert_close(chosen, ref_chosen, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(rejected, ref_rejected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-7)
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=1e-5, atol=5e-8)


def test_direct_forward_requires_preflight_before_model_execution(monkeypatch):
    import veomni.data.data_collator as m
    import veomni.trainer.text_dpo_trainer as dpo

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    batch = DPOProvenanceCollator(MainCollator())([_feature("abc", "d", 1)])
    trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(dpo_config=SimpleNamespace(average_log_prob=False, explicit_segment_provenance=True))
    )
    called = []
    with pytest.raises(RuntimeError, match="approved before and after callbacks"):
        trainer.concatenated_forward(lambda **kwargs: called.append(True), batch)
    assert not called


@pytest.mark.parametrize("mutated", [False, True])
def test_post_callback_preflight_allows_metadata_removal_but_rejects_tensor_mutation(monkeypatch, mutated):
    import veomni.data.data_collator as m
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    batch = DPOProvenanceCollator(MainCollator(channel_loss_source_id_keys=("ds_idx",)), require_ds_idx=True)(
        [_feature("abc", "d", 0)]
    )
    next_dpo_microbatches(iter([[batch]]), state, state, global_step=1, source_count=lambda: 1, vocab_size=lambda: 16)
    batch.pop("ds_idx")
    batch.pop(m.CHANNEL_LOSS_PROVENANCE)
    if mutated:
        batch["labels"][0, 1] = 8
        with pytest.raises(RuntimeError, match="changed during a callback"):
            next_dpo_microbatches(
                iter([[batch]]),
                state,
                state,
                global_step=1,
                after_callbacks=True,
                expected_proofs=(batch[DPO_APPROVED_KEY],),
            )
    else:
        next_dpo_microbatches(
            iter([[batch]]),
            state,
            state,
            global_step=1,
            after_callbacks=True,
            expected_proofs=(batch[DPO_APPROVED_KEY],),
        )
        assert batch[DPO_POSTCHECK_KEY] == batch[DPO_APPROVED_KEY]


@pytest.mark.parametrize("operation", ["add", "change"])
def test_post_callback_preflight_rejects_metadata_addition_or_mutation(monkeypatch, operation):
    import veomni.data.data_collator as m
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    batch = DPOProvenanceCollator(MainCollator())([_feature("abc", "d", 0)])
    next_dpo_microbatches(iter([[batch]]), state, state, global_step=1)
    if operation == "add":
        batch["cur_token_num"] = 12345
    else:
        batch["ds_idx"] = torch.tensor([42])
    with pytest.raises(RuntimeError, match="metadata was added or changed"):
        next_dpo_microbatches(
            iter([[batch]]),
            state,
            state,
            global_step=1,
            after_callbacks=True,
            expected_proofs=(batch[DPO_APPROVED_KEY],),
        )


def test_sp_input_evidence_covers_additional_feature_metadata(monkeypatch):
    import veomni.data.data_collator as m

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    collator = DPOProvenanceCollator(MainCollator())
    first, second = _feature("abc", "d", 0), _feature("abc", "d", 0)
    first["cur_token_num"], second["cur_token_num"] = 1, 2
    assert collator([first])[DPO_PROOF_KEY].input_digest != collator([second])[DPO_PROOF_KEY].input_digest
    first["unknown_model_argument"] = 1
    assert isinstance(collator([first])[DPO_PROOF_KEY], DPOInvalidProof)


@pytest.mark.parametrize("key", ["input_ids", "labels", "cu_seq_lens_q"])
def test_configured_source_keys_cannot_hide_model_inputs(monkeypatch, key):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _state())
    collator = DPOProvenanceCollator(MainCollator(channel_loss_source_id_keys=(key,)))
    with pytest.raises(ValueError, match="source metadata keys must not overlap model inputs"):
        collator([_feature("abc", "d", 0)])


def test_post_callback_preflight_rejects_removed_or_reordered_microbatch(monkeypatch):
    import veomni.data.data_collator as m
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    collator = DPOProvenanceCollator(MainCollator())
    first, second = collator([_feature("abc", "d", 0)]), collator([_feature("ab", "de", 1)])
    batches = next_dpo_microbatches(iter([[first, second]]), state, state, global_step=1)
    expected = tuple(batch[DPO_APPROVED_KEY] for batch in batches)
    for altered in ([first], [second, first]):
        with pytest.raises(RuntimeError, match="microbatch count or order changed"):
            next_dpo_microbatches(
                iter([altered]), state, state, global_step=1, after_callbacks=True, expected_proofs=expected
            )


def test_step_begin_callback_failure_does_not_update_optimizer(monkeypatch):
    import veomni.data.data_collator as m
    import veomni.trainer.text_dpo_trainer as dpo

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    batch = DPOProvenanceCollator(MainCollator())([_feature("abc", "d", 0)])
    trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(
            dpo_config=SimpleNamespace(explicit_segment_provenance=True),
            data=SimpleNamespace(enable_multisource=False),
        ),
        state=SimpleNamespace(global_step=0, epoch=0),
        _reset_async_activation_offload_if_enabled=MagicMock(),
        on_step_begin=MagicMock(side_effect=ValueError("injected callback failure")),
    )
    trainer.policy_model = SimpleNamespace(
        parallel_state=state,
        model_config=SimpleNamespace(vocab_size=16),
        optimizer=MagicMock(),
        lr_scheduler=MagicMock(),
    )
    trainer.reference_model = SimpleNamespace(parallel_state=state)
    with pytest.raises(ValueError, match="injected callback failure"):
        trainer.train_step(iter([[batch]]))
    trainer.policy_model.optimizer.step.assert_not_called()
    trainer.policy_model.lr_scheduler.step.assert_not_called()


def test_shared_meter_ignores_proof_when_opt_in_is_disabled():
    from veomni.utils import helper
    from veomni.utils.dpo_provenance import DPOPackedProof

    proof = DPOPackedProof((3, 1), (4,), 0, (None,), "a" * 64, "b" * 64, "c" * 64, "d" * 64)
    ordinary = {"cu_seq_lens_q": torch.tensor([0, 3, 4])}
    with_proof = {**ordinary, DPO_APPROVED_KEY: proof}
    assert helper._compute_seqlens(with_proof) == helper._compute_seqlens(ordinary)
    assert helper._compute_seqlens(with_proof, dpo_provenance_enabled=True) == [3, 1]


def test_world_packet_is_bounded_independently_of_microbatch_count(monkeypatch):
    import pickle

    import veomni.data.data_collator as m
    import veomni.trainer.dpo_provenance as protocol

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    batch = DPOProvenanceCollator(MainCollator())([_feature("abc", "d", 1)])
    original = protocol._validate_packets
    packets = []

    def capture(value):
        packets.append(pickle.dumps(value))
        original(value)

    monkeypatch.setattr(protocol, "_validate_packets", capture)
    for count in (1, 64):
        protocol.next_dpo_microbatches(iter([[dict(batch) for _ in range(count)]]), state, state, global_step=1)
    assert len(packets[1]) <= len(packets[0]) + 8
    assert len(packets[1]) < 1024


@pytest.mark.parametrize("fault", ["none", "wrong_members", "duplicate_slot", "wrong_size", "different_input"])
def test_group_descriptors_validate_complete_ordered_membership(fault):
    from veomni.trainer.dpo_provenance import _group_digest, _validate_packets

    packets = []
    for rank in range(4):
        group = (0, 1) if rank < 2 else (2, 3)
        packets.append(
            {
                "version": 1,
                "rank": rank,
                "step": 1,
                "epoch": 1,
                "phase": "before_callbacks",
                "source_count": None,
                "source_catalog_digest": None,
                "vocab_size": 16,
                "status": "ok",
                "error": "",
                "batch_count": 1,
                "proof_digest": "a" * 64,
                "group_digest": _group_digest(group),
                "group_size": 2,
                "group_slot": rank % 2,
            }
        )
    if fault == "wrong_members":
        packets[1]["group_digest"] = _group_digest((0, 2))
    elif fault == "duplicate_slot":
        packets[1]["group_slot"] = 0
    elif fault == "wrong_size":
        packets[1]["group_size"] = 3
    elif fault == "different_input":
        packets[1]["proof_digest"] = "b" * 64
    if fault == "none":
        _validate_packets(packets)
    else:
        with pytest.raises(RuntimeError):
            _validate_packets(packets)


def test_large_sequence_parallel_group_uses_bounded_descriptors():
    import pickle

    from veomni.trainer.dpo_provenance import _group_digest, _validate_packets

    size = 512
    group_hash = _group_digest(tuple(range(size)))
    packets = [
        {
            "version": 1,
            "rank": rank,
            "step": 1,
            "epoch": 1,
            "phase": "before_callbacks",
            "source_count": None,
            "source_catalog_digest": None,
            "vocab_size": 16,
            "status": "ok",
            "error": "",
            "batch_count": 32,
            "proof_digest": "a" * 64,
            "group_digest": group_hash,
            "group_size": size,
            "group_slot": rank,
        }
        for rank in range(size)
    ]
    _validate_packets(packets)
    assert max(len(pickle.dumps(packet)) for packet in packets) < 1024


def test_dpo_multisource_counts_features_not_attention_segments(monkeypatch):
    import veomni.data.data_collator as m
    from veomni.trainer.dpo_provenance import next_dpo_microbatches
    from veomni.utils import helper

    state = _state()
    monkeypatch.setattr(m, "get_parallel_state", lambda: state)
    batch = DPOProvenanceCollator(MainCollator())([_feature("abc", "de", 0), _feature("fghi", "jk", 1)])
    approved = next_dpo_microbatches(iter([[batch]]), state, state, global_step=1)[0]

    meter = object.__new__(helper.EnvironMeter)
    meter.dpo_provenance_enabled = True
    meter.config = SimpleNamespace(condition_model_type=None)
    meter.batch_seqlens, meter.batch_source_seqlens, meter.batch_ds_idx, meter.images_seqlens = [], [], [], []
    meter.enable_multisource = True
    meter.add(approved)
    assert meter.batch_seqlens == [3, 2, 4, 2]
    assert meter.batch_source_seqlens == [5, 6]
    assert meter.batch_ds_idx == [0, 1]

    class Loader:
        def update_consumed_tokens(self, value):
            self.update = value

    loader = Loader()
    tracker = object.__new__(helper.MultiSourceInfoTracker)
    tracker.dataloader = loader
    tracker.parallel_state = SimpleNamespace(dp_size=1, dp_group=None, tp_enabled=False, tp_rank=0)
    tracker.accumulate_counter, tracker.batch_idx = {}, 0
    tracker.names, tracker.boundary_type = ["a", "b"], "token"
    monkeypatch.setattr(helper.dist, "all_gather_object", lambda dest, obj, group=None: dest.__setitem__(0, obj))
    tracker.step(meter.batch_ds_idx, meter.batch_source_seqlens)
    assert tracker.accumulate_counter[0].num_tokens == 5
    assert tracker.accumulate_counter[1].num_tokens == 6
    assert loader.update == (0, 11)
