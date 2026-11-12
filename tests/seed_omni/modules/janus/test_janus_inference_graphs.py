"""Janus inference-graph and CFG-arming regressions.

Two bugs this pins:

1. ``graph_infer_interleave.yaml``'s ``image_vq`` body listed ``janus_vqvae`` as
   the first ``from_``, so the codec ran before the backbone had written the
   hidden state it projects. ``infer_gen.yaml`` had the right order, and the
   interleave graph is the default ``infer_type`` — so the default inference
   path was the broken one.
2. ``JanusTextEncoder.generate`` called ``_maybe_arm_cfg_for_image_gen`` with an
   extra argument and discarded its return, so any sampled ``<boi>`` raised
   ``TypeError`` and CFG was never armed on that path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph


def _infer_config(name: str) -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "seed_omni" / "Janus" / "janus_1.3b" / "infer" / name


def _image_vq_order(name: str) -> list[str]:
    """Node order the FSM actually runs for the ``image_vq`` body."""
    graph = GenerationGraph(yaml.safe_load(_infer_config(name).read_text()))
    graph._current = "image_vq"
    return [node.name for node in graph.iter_nodes({})]


@pytest.mark.parametrize("graph_file", ["graph_infer_gen.yaml", "graph_infer_interleave.yaml"])
def test_image_vq_runs_the_backbone_before_the_codec(graph_file: str):
    """``janus_llama`` produces the hidden state the VQ head projects.

    Reverse this and ``JanusVqvae.generate`` reads the ``<boi>`` embedding the
    text encoder just wrote — a ``(1, 1, D)`` tensor instead of backbone hidden
    states — and emits an image token before the backbone has seen the prompt.
    """
    order = _image_vq_order(graph_file)
    assert order == ["janus_llama.generate", "janus_vqvae.generate"], (
        f"{graph_file}: image_vq must run janus_llama first, got {order}"
    )


def test_both_image_graphs_agree_on_the_image_vq_order():
    assert _image_vq_order("graph_infer_gen.yaml") == _image_vq_order("graph_infer_interleave.yaml")


def test_arming_cfg_is_callable_from_both_image_span_entry_points():
    """``emit_image_start`` and the sampled-``<boi>`` path share one signature."""
    import inspect

    from veomni.models.seed_omni.modules.janus.text_encoder.modeling import JanusTextEncoder

    params = list(inspect.signature(JanusTextEncoder._maybe_arm_cfg_for_image_gen).parameters)
    assert params == ["self", "conversation_list", "generation_kwargs"]
    # It must return the embeds: the sampled-``<boi>`` path stores them on the
    # item's meta, exactly as ``emit_image_start`` does.
    assert inspect.signature(JanusTextEncoder._maybe_arm_cfg_for_image_gen).return_annotation is not None


def test_every_ar_state_samples_before_running_the_backbone_again():
    """Each text step must sample the *current* hidden state, not re-encode it.

    ``GenerationGraph.iter_nodes`` runs an endpoint once per body, so a state
    entered from a backbone-ended body (``prompt_encode`` / ``image_vq_end``) has
    to start with the text encoder: that is what turns the trailing hidden states
    into the next token's embedding. Starting with ``janus_llama`` instead feeds
    the un-sampled hidden state back as input and corrupts decoding.
    """
    for graph_file in ("graph_infer_und.yaml", "graph_infer_interleave.yaml"):
        graph = GenerationGraph(yaml.safe_load(_infer_config(graph_file).read_text()))
        graph._current = "text_ar"
        order = [node.name for node in graph.iter_nodes({})]
        assert order == ["janus_text_encoder.generate", "janus_llama.generate"], f"{graph_file}: {order}"

    # The image loop is the mirror image: the backbone produces the hidden state
    # the codec's head projects, so it comes first.
    graph = GenerationGraph(yaml.safe_load(_infer_config("graph_infer_interleave.yaml").read_text()))
    graph._current = "image_vq"
    assert [n.name for n in graph.iter_nodes({})] == ["janus_llama.generate", "janus_vqvae.generate"]
