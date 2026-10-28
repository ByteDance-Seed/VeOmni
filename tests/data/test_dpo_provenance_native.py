# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Native four-NPU gate for DPO provenance and HCCL object collectives."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous, torchrun
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type
from veomni.utils.dpo_provenance import DPO_APPROVED_KEY, DPO_PROOF_KEY, DPOInvalidProof


class _Tokenizer:
    dpo_unpadded_output_contract = True
    dpo_unpadded_output_contract_id = "test-tokenizer-v1"

    def encode(self, text, add_special_tokens=True):
        return [ord(char) for char in text]


def _native_feature(text):
    from veomni.data.data_transform import process_dpo_example

    return process_dpo_example(
        {"prompt": "p", "chosen": text, "rejected": "d"},
        tokenizer=_Tokenizer(),
        record_segment_provenance=True,
    )[0]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_native_provenance_fixture_has_shared_masked_context():
    sample = _native_feature("abc")
    assert sample["input_ids"].tolist() == [ord(c) for c in "pabcpd"]
    assert sample["labels"].tolist() == [-100, ord("a"), ord("b"), ord("c"), -100, ord("d")]
    assert sample["position_ids"].tolist() == [0, 1, 2, 3, 0, 1]


def _native_provenance_hccl(require_hccl=False):
    import veomni.data.data_collator as data_collator
    from veomni.data.dpo_collator import DPOProvenanceCollator
    from veomni.trainer.dpo_provenance import next_dpo_microbatches

    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    if require_hccl:
        assert get_device_type() == "npu" and dist.get_backend() == "hccl", "four NPU HCCL ranks are required"
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    group_id = rank // 2
    state = SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=rank % 2, sp_group=groups[group_id])
    data_collator.get_parallel_state = lambda: state

    def feature(text):
        out = _native_feature(text)
        out["ds_idx"] = group_id
        return out

    def raise_collate_error(message):
        raise ValueError(message)
        yield

    collator = DPOProvenanceCollator(data_collator.MainCollator(pad_to_length=8))
    for scenario in (
        "consistent",
        "different_input",
        "different_extra_metadata",
        "collator_wrong_sp_rank",
        "preexisting_mask_padding",
        "missing",
        "all_missing",
        "invalid",
        "one_eof",
        "microbatch_count",
        "empty_batch",
        "mutated_labels",
        "mutated_attention",
        "different_global_metadata",
        "source_range",
        "vocab_range",
        "vocab_mismatch",
        "catalog_mismatch",
        "callback_mutation",
        "callback_metadata_added",
        "callback_metadata_changed",
        "callback_drop",
    ):
        chosen = "xyz" if scenario == "different_input" and rank == 1 else "abc"
        sample = feature(chosen)
        if scenario == "different_extra_metadata":
            sample["cur_token_num"] = 1 if rank != 1 else 2
        if scenario == "preexisting_mask_padding" and rank == 1:
            sample["attention_mask"][-1] = 0
        if scenario == "source_range" and rank == 1:
            sample["ds_idx"] = 42
        if scenario == "vocab_range" and rank == 1:
            sample["labels"][-1] = 2000
        active_collator = collator
        if scenario == "collator_wrong_sp_rank" and rank == 1:
            data_collator.get_parallel_state = lambda: SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=0)
            active_collator = DPOProvenanceCollator(data_collator.MainCollator(pad_to_length=8))
            data_collator.get_parallel_state = lambda: state
        if scenario == "different_global_metadata" and rank == 1:
            altered = data_collator.MainCollator(pad_to_length=8)

            def alter_gdn_boundaries(value):
                value["linear_attn_cu_seq_lens_q"] = value["linear_attn_cu_seq_lens_q"].clone()
                value["linear_attn_cu_seq_lens_q"][1] = 2
                return value

            altered.preforward_pipeline.append(alter_gdn_boundaries)
            active_collator = DPOProvenanceCollator(altered)
        collate_error = None
        try:
            batch = active_collator([sample])
        except ValueError as error:
            collate_error = str(error)
            batch = {}
        if (scenario == "missing" and rank == 1) or scenario == "all_missing":
            batch.pop(DPO_PROOF_KEY)
        if scenario == "invalid" and rank == 1:
            batch[DPO_PROOF_KEY] = DPOInvalidProof("injected invalid provenance")
        if scenario == "mutated_labels" and rank == 1:
            batch["labels"][0, 0] += 1
        if scenario == "mutated_attention" and rank == 1:
            batch["max_length_q"] += 1
        batches = [batch, dict(batch)] if scenario == "microbatch_count" and rank >= 2 else [batch]
        if scenario == "callback_drop":
            batches = [batch, dict(batch)]
        if scenario == "empty_batch" and rank == 1:
            batches = []
        iterator = iter([]) if scenario == "one_eof" and rank == 1 else iter([batches])
        if collate_error is not None:
            iterator = raise_collate_error(collate_error)
        try:
            expected_vocab_size = 999 if scenario == "vocab_mismatch" and rank == 1 else 1000
            expected_catalog = ("a", "different") if scenario == "catalog_mismatch" and rank == 1 else ("a", "b")
            approved = next_dpo_microbatches(
                iterator,
                state,
                state,
                global_step=1,
                epoch=1,
                source_count=lambda: 2,
                source_catalog=lambda value=expected_catalog: value,
                vocab_size=lambda value=expected_vocab_size: value,
            )
            expected_proofs = tuple(batch[DPO_APPROVED_KEY] for batch in approved)
            if scenario == "callback_drop":
                approved.pop()
            if scenario == "callback_mutation" and rank == 1:
                approved[0]["labels"][0, 0] += 1
            if scenario == "callback_metadata_added" and rank == 1:
                approved[0]["cur_token_num"] = 12345
            if scenario == "callback_metadata_changed" and rank == 1:
                approved[0]["ds_idx"] = torch.tensor([99])
            next_dpo_microbatches(
                iter([approved]),
                state,
                state,
                global_step=1,
                epoch=1,
                after_callbacks=True,
                expected_proofs=expected_proofs,
            )
            result = "approved" if DPO_APPROVED_KEY in approved[0] else "legacy"
        except RuntimeError:
            result = "error"
        statuses = [None] * world
        dist.all_gather_object(statuses, result)
        assert statuses == ["approved" if scenario == "consistent" else "error"] * world, (scenario, statuses)


def _gloo_provenance_worker(rank, world_size, endpoint):
    dist.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        assert dist.get_backend() == "gloo"
        _native_provenance_hccl()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_dpo_provenance_gloo():
    spawn_with_rendezvous(_gloo_provenance_worker, args=(4,), nprocs=4)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=4))
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires four Ascend NPUs and HCCL")
def test_dpo_provenance_native_hccl():
    torchrun(_native_provenance_hccl, world_size=4, require_hccl=True)
