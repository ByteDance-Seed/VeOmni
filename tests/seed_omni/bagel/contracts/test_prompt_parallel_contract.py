from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modulemixin import BagelQwen2MoTModuleMixin
from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import preprocess_mot_inputs
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_VAE_CONTEXT,
)
from veomni.models.seed_omni.modules.bagel.text_encoder.processing import apply_image_marker
from veomni.models.seed_omni.utils.conversation import ConversationItem, iter_desired_items


_BAGEL_CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs/seed_omni/Bagel/bagel_7b_mot"
_HIDDEN_SIZE = 4
_Z_CHANNELS = 2


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


def _marker_text(value: torch.Tensor, *, source: str) -> ConversationItem:
    return _item(
        value,
        type_="text",
        role="user",
        source=source,
        meta={"labels": torch.full((1,), -100)},
    )


def _materialized_prompt() -> tuple[list[ConversationItem], ConversationItem, ConversationItem]:
    sample, image_item, text_item = _raw_prompt_sample()
    latent_item = _item(
        _vae_context_latent(),
        type_="image",
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    markers = _marker_embeds().squeeze(0)
    start_vae = _marker_text(markers[:1], source=BAGEL_VAE_CONTEXT)
    end_vae = _marker_text(markers[1:], source=BAGEL_VAE_CONTEXT)
    start_siglip = _marker_text(markers[:1], source=BAGEL_SIGLIP_CONTEXT)
    end_siglip = _marker_text(markers[1:], source=BAGEL_SIGLIP_CONTEXT)
    image_item.value = _siglip_image_embeds()
    image_item.source = BAGEL_SIGLIP_CONTEXT
    sample[:] = [start_vae, latent_item, end_vae, start_siglip, image_item, end_siglip, text_item]
    return sample, latent_item, text_item


def test_bagel_edit_prompt_graph_exposes_independent_prompt_producers() -> None:
    graph_config = yaml.safe_load((_BAGEL_CONFIG_DIR / "graph_infer_edit.yaml").read_text())["generation_graph"]
    prompt_body = graph_config["states"]["prompt_encode"]["body"]

    assert {"from": "bagel_text_encoder", "to": "bagel_qwen2_mot"} in prompt_body
    assert {"from": "bagel_siglip_navit", "to": "bagel_qwen2_mot"} in prompt_body
    assert {"from": "bagel_vae.encode_context", "to": "bagel_flow_connector.embed_context_latents"} in prompt_body
    assert {"from": "bagel_flow_connector.embed_context_latents", "to": "bagel_qwen2_mot"} in prompt_body
    assert all(edge["to"] != "bagel_text_encoder.encode_image_markers" for edge in prompt_body)

    forbidden_edges = {
        ("bagel_vae.encode_context", "bagel_siglip_navit"),
        ("bagel_siglip_navit", "bagel_vae.encode_context"),
        ("bagel_siglip_navit", "bagel_flow_connector.embed_context_latents"),
    }
    assert not forbidden_edges.intersection({(edge["from"], edge["to"]) for edge in prompt_body})

    graph = GenerationGraph(graph_config)
    sequence = graph.state_node_sequence("prompt_encode")
    assert "bagel_vae.encode_context" in sequence
    assert "bagel_siglip_navit.generate" in sequence


def test_materialized_edit_prompt_has_vae_context_before_siglip_context() -> None:
    sample, latent_item, text_item = _materialized_prompt()
    image_item = sample[4]

    assert [
        index for index, item in enumerate(sample) if item is latent_item or item is image_item or item is text_item
    ] == [
        1,
        4,
        6,
    ]
    assert latent_item.source == BAGEL_VAE_CONTEXT
    assert image_item.source == BAGEL_SIGLIP_CONTEXT
    assert torch.equal(latent_item.value, _vae_context_latent())
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
    sample, latent_item, text_item = _materialized_prompt()

    flow_items = list(iter_desired_items([sample], types=["image"], sources=[BAGEL_VAE_CONTEXT]))
    assert flow_items == [latent_item]
    latent_item.value = _flow_context_embeds()

    text_item.value = _text_embeds()

    packed = preprocess_mot_inputs(
        [sample],
        device=torch.device("cpu"),
        dtype=torch.float32,
        hidden_size=_HIDDEN_SIZE,
    )

    assert packed is not None
    assert [span.item for span in packed.spans] == [latent_item, sample[4], text_item]
    assert [span.length for span in packed.spans] == [4, 5, 2]
    assert packed.spans[0].items == (sample[0], latent_item, sample[2])
    assert packed.spans[1].items == (sample[3], sample[4], sample[5])
    assert packed.sample_lens == [11]
    assert torch.equal(packed.packed_sequence[:1], _marker_embeds().squeeze(0)[:1])
    assert torch.equal(packed.packed_sequence[1:3], _flow_context_embeds())
    assert torch.equal(packed.packed_sequence[3:4], _marker_embeds().squeeze(0)[1:])
    assert torch.equal(packed.packed_gen_token_indexes, torch.tensor([1, 2]))
    assert torch.equal(packed.packed_und_token_indexes, torch.tensor([0, 3, 4, 5, 6, 7, 8, 9, 10]))


def test_mot_forward_post_scatters_virtual_marker_triplet_hidden_states() -> None:
    sample, latent_item, text_item = _materialized_prompt()
    siglip_item = sample[4]
    latent_item.value = _flow_context_embeds()
    text_item.value = _text_embeds()

    packed = preprocess_mot_inputs(
        [sample],
        device=torch.device("cpu"),
        dtype=torch.float32,
        hidden_size=_HIDDEN_SIZE,
    )
    assert packed is not None

    model = SimpleNamespace(
        device=torch.device("cpu"),
        _conversation_carrier=[sample],
        _packed_training=packed,
    )
    hidden_states = torch.arange(11 * _HIDDEN_SIZE, dtype=torch.float32).reshape(11, _HIDDEN_SIZE)

    out = BagelQwen2MoTModuleMixin.forward_post(model, hidden_states)

    assert out["conversation_list"] == [sample]
    assert torch.equal(sample[0].value, hidden_states[0:1])
    assert torch.equal(latent_item.value, hidden_states[1:3])
    assert torch.equal(sample[2].value, hidden_states[3:4])
    assert torch.equal(sample[3].value, hidden_states[4:5])
    assert torch.equal(siglip_item.value, hidden_states[5:8])
    assert torch.equal(sample[5].value, hidden_states[8:9])
    assert torch.equal(text_item.value, hidden_states[9:11])
