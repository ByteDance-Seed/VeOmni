from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from veomni.models.seed_omni.conversation import ConversationItem, iter_desired_items
from veomni.models.seed_omni.generation_graph import GenerationGraph
from veomni.models.seed_omni.modules.bagel.carrier_updates import (
    insert_before,
    materialize_carrier_updates,
    replace_fields,
    replace_value,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import pack_training_conversation
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_FLOW_QUERY,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_VAE_CONTEXT,
)
from veomni.models.seed_omni.modules.bagel.text_encoder.processing import apply_image_marker


_BAGEL_CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs/seed_omni/Bagel/bagel_7b_mot"
_HIDDEN_SIZE = 4
_Z_CHANNELS = 2
_IMAGE_MARKER_SOURCES = {BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT, BAGEL_FLOW_QUERY}


def _item(
    value: object,
    *,
    type_: str,
    role: str = "user",
    source: str | None = None,
    meta: dict[str, Any] | None = None,
) -> ConversationItem:
    return ConversationItem(type=type_, value=value, role=role, source=source, meta={} if meta is None else dict(meta))


def _raw_prompt_sample() -> tuple[list[ConversationItem], ConversationItem, ConversationItem]:
    raw_image = _item(torch.arange(12, dtype=torch.float32).reshape(3, 2, 2), type_="image")
    text = _item("edit this image", type_="text")
    return [raw_image, text], raw_image, text


def _vae_context_latent() -> torch.Tensor:
    return torch.arange(8, dtype=torch.float32).reshape(_Z_CHANNELS, 2, 2)


def _siglip_image_embeds() -> torch.Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(3, _HIDDEN_SIZE)


def _flow_context_embeds() -> torch.Tensor:
    return torch.arange(8, dtype=torch.float32).reshape(2, _HIDDEN_SIZE) + 20


def _text_embeds() -> torch.Tensor:
    return torch.arange(8, dtype=torch.float32).reshape(2, _HIDDEN_SIZE) + 40


def _marker_embeds() -> torch.Tensor:
    return torch.tensor([[[101.0, 102.0, 103.0, 104.0], [201.0, 202.0, 203.0, 204.0]]])


def _prompt_updates(
    image_item: ConversationItem,
    *,
    latent_value: torch.Tensor | None = None,
    image_embeds: torch.Tensor | None = None,
) -> tuple[ConversationItem, list[Any]]:
    latent_item = _item(
        _vae_context_latent() if latent_value is None else latent_value,
        type_="output",
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    return latent_item, [
        insert_before(image_item, latent_item),
        replace_fields(
            image_item,
            value=_siglip_image_embeds() if image_embeds is None else image_embeds,
            source=BAGEL_SIGLIP_CONTEXT,
        ),
    ]


def _materialized_prompt_sequential() -> tuple[list[ConversationItem], ConversationItem, ConversationItem]:
    sample, image_item, text_item = _raw_prompt_sample()
    latent_item, _ = _prompt_updates(image_item)
    materialize_carrier_updates([sample], [insert_before(image_item, latent_item)])
    materialize_carrier_updates(
        [sample],
        [replace_fields(image_item, value=_siglip_image_embeds(), source=BAGEL_SIGLIP_CONTEXT)],
    )
    return sample, latent_item, text_item


def _materialized_prompt_from_same_snapshot() -> tuple[list[ConversationItem], ConversationItem, ConversationItem]:
    sample, image_item, text_item = _raw_prompt_sample()
    latent_item, updates = _prompt_updates(image_item)
    materialize_carrier_updates([sample], updates)
    return sample, latent_item, text_item


def _assert_carrier_equivalent(
    left: list[ConversationItem],
    right: list[ConversationItem],
) -> None:
    assert [(item.type, item.role, item.meta) for item in left] == [
        (item.type, item.role, item.meta) for item in right
    ]
    for left_item, right_item in zip(left, right, strict=True):
        if torch.is_tensor(left_item.value) or torch.is_tensor(right_item.value):
            assert torch.equal(left_item.value, right_item.value)
        else:
            assert left_item.value == right_item.value


def test_bagel_edit_prompt_graph_exposes_vae_then_siglip_prompt_producers() -> None:
    graph_config = yaml.safe_load((_BAGEL_CONFIG_DIR / "graph_infer_edit.yaml").read_text())["generation_graph"]
    prompt_body = graph_config["states"]["prompt_encode"]["body"]

    assert {"from": "bagel_text_encoder", "to": "bagel_qwen2_mot"} in prompt_body
    assert {"from": "bagel_vae.encode_context", "to": "bagel_siglip_navit"} in prompt_body
    assert {"from": "bagel_siglip_navit", "to": "bagel_flow_connector.embed_context_latents"} in prompt_body
    assert {
        "from": "bagel_flow_connector.embed_context_latents",
        "to": "bagel_text_encoder.encode_image_markers",
    } in (prompt_body)
    assert {
        "from": "bagel_text_encoder.encode_image_markers",
        "to": "bagel_qwen2_mot",
    } in (prompt_body)

    forbidden_edges = {
        ("bagel_siglip_navit", "bagel_vae.encode_context"),
    }
    assert not forbidden_edges.intersection({(edge["from"], edge["to"]) for edge in prompt_body})

    graph = GenerationGraph(graph_config)
    sequence = graph.state_node_sequence("prompt_encode")
    assert "bagel_vae.encode_context" in sequence
    assert "bagel_siglip_navit.generate" in sequence


def test_same_snapshot_prompt_updates_match_current_sequential_materialization() -> None:
    sequential, _, _ = _materialized_prompt_sequential()
    parallel_compatible, latent_item, _ = _materialized_prompt_from_same_snapshot()

    _assert_carrier_equivalent(parallel_compatible, sequential)
    assert parallel_compatible[0] is latent_item
    assert parallel_compatible[1].type == "image"
    assert torch.equal(parallel_compatible[1].value, _siglip_image_embeds())


def test_same_snapshot_replace_targets_original_image_not_inserted_latent() -> None:
    sample, image_item, _ = _raw_prompt_sample()
    text_item = sample[1]
    latent_item, updates = _prompt_updates(image_item)

    materialize_carrier_updates([sample], updates)

    assert [id(item) for item in sample] == [id(latent_item), id(image_item), id(text_item)]
    assert torch.equal(latent_item.value, _vae_context_latent())
    assert torch.equal(image_item.value, _siglip_image_embeds())


def test_repeated_prompt_update_materialization_does_not_duplicate_context_latent() -> None:
    sample, image_item, _ = _raw_prompt_sample()
    text_item = sample[1]
    latent_item, updates = _prompt_updates(image_item)

    materialize_carrier_updates([sample], updates)
    materialize_carrier_updates([sample], updates)

    assert sample.count(latent_item) == 1
    assert [id(item) for item in sample] == [id(latent_item), id(image_item), id(text_item)]
    assert torch.equal(image_item.value, _siglip_image_embeds())


def test_image_marker_wrapping_is_idempotent() -> None:
    image_item = _item(_siglip_image_embeds(), type_="image", source=BAGEL_SIGLIP_CONTEXT)
    marker_embeds = _marker_embeds().squeeze(0)

    apply_image_marker(image_item, marker_embeds, device=torch.device("cpu"), dtype=torch.float32)
    wrapped_once = image_item.value.clone()
    apply_image_marker(image_item, marker_embeds, device=torch.device("cpu"), dtype=torch.float32)

    assert torch.equal(image_item.value, wrapped_once)
    assert image_item.value.shape == (5, _HIDDEN_SIZE)
    assert torch.equal(image_item.value[:1], marker_embeds[:1])
    assert torch.equal(image_item.value[-1:], marker_embeds[1:])


def test_downstream_prompt_consumers_see_sequentially_equivalent_carrier() -> None:
    sample, latent_item, text_item = _materialized_prompt_from_same_snapshot()

    flow_items = list(iter_desired_items([sample], types=["output"], sources=[BAGEL_VAE_CONTEXT]))
    assert flow_items == [latent_item]
    materialize_carrier_updates([sample], [replace_value(latent_item, _flow_context_embeds())])

    image_items = [
        item for item in sample if item.type in {"image", "output"} and item.source in _IMAGE_MARKER_SOURCES
    ]
    assert image_items == [latent_item, sample[1]]
    for image_item in image_items:
        apply_image_marker(image_item, _marker_embeds().squeeze(0), device=torch.device("cpu"), dtype=torch.float32)
    text_item.value = _text_embeds()

    packed = pack_training_conversation(
        [sample],
        device=torch.device("cpu"),
        dtype=torch.float32,
        hidden_size=_HIDDEN_SIZE,
    )

    assert packed is not None
    assert [span.item for span in packed.spans] == [latent_item, sample[1], text_item]
    assert packed.sample_lens == [11]
    assert torch.equal(packed.packed_gen_token_indexes, torch.arange(0, 4))
    assert torch.equal(packed.packed_und_token_indexes, torch.arange(4, 11))
