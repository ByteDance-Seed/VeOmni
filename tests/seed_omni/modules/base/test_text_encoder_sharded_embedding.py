"""SeedOmni text encoders on a ``ShardedEmbedding`` table.

The base ``TextEncoder`` builds ``embed_tokens`` as a ``ShardedEmbedding`` and decodes
a tied head through ``embed_tokens.project``; nothing above the embedding reads its
weight. Qwen3's image mode trains only the vision special-token rows, so its row
mask has to survive what parallelization does to that weight: the ``emb`` plan
replaces it with this rank's vocab rows, and FSDP2 replaces it again with a sharded
parameter. The FSDP2 runs wrap the table by hand on CPU gloo ranks, as
``tests/distributed/test_emb_parallel.py`` does, and compare with a dense reference.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from veomni.distributed.emb_parallel import sharded_embedding
from veomni.models.seed_omni.modules.base.text_encoder.configuration import TextEncoderConfig
from veomni.models.seed_omni.modules.base.text_encoder.modeling import TextEncoder
from veomni.models.seed_omni.modules.qwen3.text_encoder.accelerated import Qwen3TextEncoderAccelerated
from veomni.models.seed_omni.modules.qwen3.text_encoder.accelerated import accelerated as qwen3_accelerated
from veomni.models.seed_omni.modules.qwen3.text_encoder.configuration import Qwen3TextEncoderConfig


_VOCAB, _HIDDEN = 8, 4
# One vision row in emb rank 0's half of the table, two in rank 1's.
_VISION_IDS = {"<|vision_start|>": 1, "<|vision_end|>": 6, "<|image_pad|>": 7}
_RANK_IDS = [[0, 1, 1, 5], [6, 2, 7], [3, 4, 7, 1], []]  # the last rank holds no tokens


def _table() -> torch.Tensor:
    return torch.randn(_VOCAB, _HIDDEN, generator=torch.Generator().manual_seed(0))


def _upstream_grad(rank: int, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(100 + rank))


def _install_state(state, monkeypatch=None) -> None:
    """Point ``ShardedEmbedding`` and the Qwen3 row mask at ``state`` (``None``: never initialized)."""
    patches = [
        (sharded_embedding, "is_parallel_state_initialized", lambda: state is not None),
        (sharded_embedding, "get_parallel_state", lambda: state),
        (qwen3_accelerated, "get_parallel_state", lambda: state),
    ]
    for module, name, value in patches:
        if monkeypatch is None:
            setattr(module, name, value)
        else:
            monkeypatch.setattr(module, name, value)


def _image_mode_text_encoder() -> Qwen3TextEncoderAccelerated:
    config = Qwen3TextEncoderConfig(
        vocab_size=_VOCAB, hidden_size=_HIDDEN, tie_word_embeddings=True, enable_image=True
    )
    text_encoder = Qwen3TextEncoderAccelerated(config)
    # ``freeze_model`` only resolves the vision ids; skip the tokenizer setter's chat template.
    text_encoder._tokenizer = SimpleNamespace(convert_tokens_to_ids=_VISION_IDS.__getitem__)
    return text_encoder


def _encode_then_project(text_encoder, ids: list[int]) -> torch.Tensor:
    """Lookup and tied head as separate calls, like the graph's encode and decode nodes."""
    embeds = text_encoder.encode(input_ids=torch.tensor(ids, dtype=torch.long))["inputs_embeds"]
    return text_encoder._project(torch.tanh(embeds))


def _dense_masked_grad() -> torch.Tensor:
    """Every rank's table gradient on the unsplit table, with the frozen rows zeroed."""
    table = _table().requires_grad_(True)
    for rank, ids in enumerate(_RANK_IDS):
        logits = F.linear(torch.tanh(F.embedding(torch.tensor(ids, dtype=torch.long), table)), table)
        (logits * _upstream_grad(rank, *logits.shape)).sum().backward()
    keep = torch.zeros(_VOCAB, dtype=torch.bool)
    keep[list(_VISION_IDS.values())] = True
    return table.grad * keep.unsqueeze(1)


_LR, _WD = 0.1, 0.5


def _vision_rows() -> torch.Tensor:
    keep = torch.zeros(_VOCAB, dtype=torch.bool)
    keep[list(_VISION_IDS.values())] = True
    return keep


def _dense_adamw_step(table: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """One AdamW step on a dense copy, decaying every row."""
    param = nn.Parameter(table.clone())
    param.grad = grad.clone()
    torch.optim.AdamW([param], lr=_LR, weight_decay=_WD, foreach=False).step()
    return param.detach()


def _fsdp_rank_main(rank: int, rendezvous: str, out_dir: str, layout: str) -> None:
    """``emb``: emb=2 x emb_fsdp=2. ``dp``: emb off, the table a plain FSDP2 unit over all ranks."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard

    world = len(_RANK_IDS)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        if layout == "emb":
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("emb_fsdp", "emb"))
            emb_rank = mesh["emb"].get_local_rank()
            _install_state(
                SimpleNamespace(
                    extra_parallel_sizes={"emb": 2},
                    extra_parallel_enabled=lambda name: True,
                    extra_parallel_group=lambda name: mesh["emb"].get_group(),
                    extra_parallel_rank=lambda name: emb_rank,
                )
            )
            rows = _VOCAB // 2
            chunk = slice(emb_rank * rows, (emb_rank + 1) * rows)
            # FSDP2's default divide factor: gloo has no PREMUL_SUM, which a custom factor needs.
            divisor = mesh["emb_fsdp"].size()
            shard_kwargs = {"mesh": mesh["emb_fsdp"], "shard_placement_fn": lambda param: Shard(1)}
        else:
            _install_state(None)
            chunk = slice(None)
            divisor = world
            shard_kwargs = {"mesh": init_device_mesh("cpu", (world,))}

        text_encoder = _image_mode_text_encoder()
        text_encoder.freeze_model()
        # The order ModuleRuntime builds in: freeze, then the plan slices the rows, then FSDP2.
        text_encoder.embed_tokens.weight = nn.Parameter(_table()[chunk].clone())
        fully_shard(text_encoder.embed_tokens, reshard_after_forward=True, **shard_kwargs)

        logits = _encode_then_project(text_encoder, _RANK_IDS[rank])
        (logits * _upstream_grad(rank, *logits.shape)).sum().backward()

        grad = text_encoder.embed_tokens.weight.grad.full_tensor()
        expected = _dense_masked_grad()[chunk] / divisor

        optimizer = torch.optim.AdamW(text_encoder.parameters(), lr=_LR, weight_decay=_WD, foreach=False)
        text_encoder.configure_optimizer(optimizer)
        optimizer.step()
        stepped = text_encoder.embed_tokens.weight.detach().full_tensor()
        vision = _vision_rows()[chunk]
        reference = _dense_adamw_step(_table()[chunk], expected)
        result = {
            "grad_matches_dense_masked": torch.allclose(grad, expected, atol=1e-5),
            "frozen_rows_are_zero": bool((grad[expected.abs().sum(1) == 0] == 0).all()),
            "frozen_rows_unchanged": torch.equal(stepped[~vision], _table()[chunk][~vision]),
            "vision_rows_match_adamw": torch.allclose(stepped[vision], reference[vision], atol=1e-6),
        }
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(result, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layout", ["emb", "dp"])
def test_qwen3_image_mode_trains_only_vision_rows_through_fsdp(tmp_path, layout):
    world = len(_RANK_IDS)
    mp.spawn(_fsdp_rank_main, args=(str(tmp_path / "rendezvous"), str(tmp_path), layout), nprocs=world, join=True)
    for rank in range(world):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert all(result.values()), (rank, result)


def test_qwen3_image_mode_masks_the_weight_that_replaced_the_frozen_one(monkeypatch):
    """No FSDP: the weight loader still swaps the parameter after ``freeze_model``."""
    _install_state(None, monkeypatch)
    text_encoder = _image_mode_text_encoder()
    text_encoder.freeze_model()
    text_encoder.embed_tokens.weight = nn.Parameter(_table())
    for rank, ids in enumerate(_RANK_IDS):
        logits = _encode_then_project(text_encoder, ids)
        (logits * _upstream_grad(rank, *logits.shape)).sum().backward()
    torch.testing.assert_close(text_encoder.embed_tokens.weight.grad, _dense_masked_grad())


def test_qwen3_image_mode_weight_decay_reaches_only_vision_rows(monkeypatch):
    """The table shares a group with another parameter, and a resume reloads the group's decay."""
    _install_state(None, monkeypatch)
    text_encoder = _image_mode_text_encoder()
    text_encoder.freeze_model()
    text_encoder.embed_tokens.weight = nn.Parameter(_table())
    other = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([text_encoder.embed_tokens.weight, other], lr=_LR, weight_decay=_WD, foreach=False)
    text_encoder.configure_optimizer(optimizer)
    saved = optimizer.state_dict()
    for group in saved["param_groups"]:
        group["weight_decay"] = _WD
    optimizer.load_state_dict(saved)

    vision = _vision_rows()
    for _ in range(2):
        optimizer.zero_grad()
        logits = _encode_then_project(text_encoder, _RANK_IDS[0])
        (logits * _upstream_grad(0, *logits.shape)).sum().backward()
        other.grad = torch.zeros_like(other)
        optimizer.step()

    weight = text_encoder.embed_tokens.weight.detach()
    assert torch.equal(weight[~vision], _table()[~vision])
    # Every vision row is either looked up or read by the tied head, so each one moved.
    assert not torch.equal(weight[vision], _table()[vision])
    torch.testing.assert_close(other.detach(), torch.full((2,), (1 - _LR * _WD) ** 2))


def test_qwen3_image_mode_vision_rows_follow_adamw_with_weight_decay(monkeypatch):
    _install_state(None, monkeypatch)
    text_encoder = _image_mode_text_encoder()
    text_encoder.freeze_model()
    text_encoder.embed_tokens.weight = nn.Parameter(_table())
    optimizer = torch.optim.AdamW(text_encoder.parameters(), lr=_LR, weight_decay=_WD, foreach=False)
    text_encoder.configure_optimizer(optimizer)
    logits = _encode_then_project(text_encoder, _RANK_IDS[1])
    (logits * _upstream_grad(1, *logits.shape)).sum().backward()
    reference = _dense_adamw_step(_table(), text_encoder.embed_tokens.weight.grad)
    optimizer.step()
    vision = _vision_rows()
    torch.testing.assert_close(text_encoder.embed_tokens.weight.detach()[vision], reference[vision])


def test_text_only_qwen3_trains_every_row(monkeypatch):
    _install_state(None, monkeypatch)
    text_encoder = _image_mode_text_encoder()
    text_encoder.config.enable_image = False
    text_encoder.freeze_model()
    text_encoder.embed_tokens.weight = nn.Parameter(_table())
    logits = _encode_then_project(text_encoder, _RANK_IDS[0])
    logits.sum().backward()
    assert not text_encoder.embed_tokens._forward_pre_hooks
    assert text_encoder.embed_tokens.weight.grad[[0, 5]].abs().sum() > 0


@pytest.mark.parametrize("pad_token_id", [None, 3])
def test_pad_token_id_becomes_the_embedding_padding_idx(pad_token_id):
    config = TextEncoderConfig(vocab_size=_VOCAB, hidden_size=_HIDDEN, pad_token_id=pad_token_id)
    assert TextEncoder(config).embed_tokens.padding_idx == pad_token_id
