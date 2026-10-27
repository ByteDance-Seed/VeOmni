"""Integer-only QSA layout reuse must preserve packed samples and owner order."""

import pytest
import torch

from veomni.models.transformers.qwen4_exp.qsa_metadata import cached_host_metadata, prepare_qsa_metadata


@pytest.mark.parametrize("segments", [((0, 0, 16),), ((0, 0, 3), (0, 3, 16)), ((0, 0, 16), (1, 0, 16))])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_cached_layout_matches_reference(segments, world):
    """All ranks retain the exact uncoalesced owner-major metadata contract."""
    local = 16 // world
    for rank in range(world):
        metadata = cached_host_metadata(segments, 4, local, rank, world)
        reference = [(i, b, x) for i, (b, start, end) in enumerate(segments) for x in range(start, end - 3, 4)]
        owned = tuple(x for x in reference if x[2] // local == rank)
        gathered = tuple(x for owner in range(world) for x in reference if x[2] // local == owner)
        sections = (
            tuple(x[1] for x in owned),
            tuple(x[2] for x in owned),
            tuple(x[0] for x in gathered),
            tuple(x[2] for x in gathered),
        )
        assert metadata.counts == tuple(sum(x[2] // local == owner for x in reference) for owner in range(world))
        assert metadata.section_sizes == tuple(map(len, sections))
        assert metadata.flat_metadata == tuple(value for section in sections for value in section)


def test_cache_is_bounded_and_packing_sensitive():
    """Changing boundaries or ranks must not reuse a previous sample's layout."""
    cached_host_metadata.cache_clear()
    for end in range(5, 16):
        result = cached_host_metadata(((0, 0, end),), 4, 16, 0, 1)
        assert all(isinstance(value, int) for value in result.flat_metadata)
        assert all(isinstance(value, int) for value in result.counts)
    assert cached_host_metadata.cache_info().currsize == 2
    first = cached_host_metadata(((0, 0, 16),), 4, 8, 0, 2)
    assert first is cached_host_metadata(((0, 0, 16),), 4, 8, 0, 2)
    assert first != cached_host_metadata(((0, 0, 3), (0, 3, 16)), 4, 8, 0, 2)
    assert first != cached_host_metadata(((0, 0, 16),), 4, 8, 1, 2)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_prepared_query_segments_match_global_reference(world, packed):
    """Rank-local maps must exactly equal the previous full-batch maps' slices."""
    lengths = (3, 13, 7, 9)
    bounds = torch.tensor([0, 3, 16, 23, 32]) if packed else None
    segments = ((0, 0, 3), (0, 3, 16), (1, 0, 7), (1, 7, 16)) if packed else ((0, 0, 16), (1, 0, 16))
    ids = torch.empty(2, 16, dtype=torch.long)
    starts = torch.empty_like(ids)
    for segment, (batch, start, end) in enumerate(segments):
        ids[batch, start:end] = segment
        starts[batch, start:end] = start
    local = 16 // world
    for rank in range(world):
        metadata = prepare_qsa_metadata(2, local, 4, rank, world, torch.device("cpu"), bounds, lengths)
        assert metadata.layout == (2, local, 4, rank, world)
        assert torch.equal(metadata.local_segment_ids, ids[:, rank * local : (rank + 1) * local])
        assert torch.equal(metadata.local_segment_starts, starts[:, rank * local : (rank + 1) * local])
        assert isinstance(metadata.counts, tuple)
        assert metadata.local_segment_ids.numel() == 2 * local


@pytest.mark.parametrize("bounds", [[1, 16], [0, 15], [0, 8, 8, 16], [0, 9, 16]])
def test_prepared_metadata_rejects_invalid_packing(bounds):
    with pytest.raises(ValueError):
        prepare_qsa_metadata(2, 8, 4, 0, 1, torch.device("cpu"), torch.tensor(bounds))


def test_host_counts_never_need_tensor_to_host_copy(monkeypatch):
    from veomni.distributed.context_parallel import dsa_cp

    gathered = torch.tensor([11, 12, 21, -9, -9, -9]).reshape(1, 6, 1)
    monkeypatch.setattr(dsa_cp, "gather_outputs", lambda *args, **kwargs: gathered)

    def unexpected_tolist(self):
        raise AssertionError("host counts must not call Tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", unexpected_tolist)
    actual = dsa_cp.all_gather_compressed_rows(torch.tensor([11, 12]).reshape(1, 2, 1), (2, 1, 0), object())
    assert torch.equal(actual, torch.tensor([11, 12, 21]).reshape(1, 3, 1))


@pytest.mark.parametrize("checkpointing", [False, True])
def test_model_reuses_metadata_without_changing_outputs_or_gradients(monkeypatch, checkpointing):
    """Real eager decoder layers reuse one preparation, also during recomputation."""
    import copy
    from importlib import import_module

    from transformers import AutoConfig

    from tests.tools.training_utils import make_eager_ops_config

    module = import_module("veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu")
    config = AutoConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json").text_config
    config.ple_layer_ids = []
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config._set_ops_implementation(make_eager_ops_config())
    candidate = module.Qwen4ExpTextModel(config)
    reference = copy.deepcopy(candidate)
    if checkpointing:
        for model in (candidate, reference):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for layer in reference.layers:

        def drop_prepared(_module, args, kwargs):
            kwargs.pop("qsa_metadata", None)
            return args, kwargs

        layer.self_attn.indexer.register_forward_pre_hook(drop_prepared, with_kwargs=True)

    preparations = []

    def prepare(*args, **kwargs):
        result = prepare_qsa_metadata(*args, **kwargs)
        preparations.append(result)
        return result

    monkeypatch.setattr(module, "prepare_qsa_metadata", prepare)
    ids = torch.arange(16)[None] + 3

    def run(model, lengths):
        bounds = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
        output = model(input_ids=ids, cu_seq_lens_q=bounds, packed_seq_lens=lengths, use_cache=False).last_hidden_state
        output.square().mean().backward()
        return output

    actual = run(candidate, (3, 13))
    assert len(preparations) == 1
    first_metadata = preparations[0]
    expected = run(reference, (3, 13))
    assert len(preparations) > 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for (name, actual_param), (reference_name, expected_param) in zip(
        candidate.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        if actual_param.grad is None or expected_param.grad is None:
            assert actual_param.grad is expected_param.grad is None, name
        else:
            torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=0, atol=0, msg=name)
    preparations.clear()
    run(candidate, (7, 9))
    assert len(preparations) == 1
    assert preparations[0] is not first_metadata
    assert not torch.equal(preparations[0].local_segment_ids, first_metadata.local_segment_ids)
