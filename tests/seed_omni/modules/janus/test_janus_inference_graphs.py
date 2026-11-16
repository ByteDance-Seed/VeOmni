"""Janus inference-graph node ordering.

Two properties the FSM bodies must keep:

* ``image_vq`` runs the backbone before the codec. ``JanusVqvae.generate`` projects
  a hidden state the backbone produces; reversed, it would read the ``<boi>``
  embedding the text encoder just wrote instead.
* An AR text state starts from the text encoder. ``GenerationGraph.iter_nodes``
  runs an endpoint once per body, so a state entered from a backbone-ended body
  (``prompt_encode``, or ``image_vq_end`` in a graph that had one) has to
  re-encode the trailing token rather than feed the un-sampled hidden state back.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph


def _infer_config(name: str) -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "seed_omni" / "Janus" / "janus" / "infer" / name


def _node_order(graph_file: str, state: str) -> list[str]:
    """The order the FSM actually runs a state's body in."""
    graph = GenerationGraph(yaml.safe_load(_infer_config(graph_file).read_text()))
    graph._current = state
    return [node.name for node in graph.iter_nodes({})]


def test_image_vq_runs_the_backbone_before_the_codec():
    assert _node_order("graph_infer_gen.yaml", "image_vq") == [
        "janus_llama.generate",
        "janus_vqvae.generate",
    ]


def test_an_ar_text_state_starts_from_the_text_encoder():
    """``infer_und`` decodes token by token, so its AR state re-encodes first."""
    assert _node_order("graph_infer_und.yaml", "text_ar") == [
        "janus_text_encoder.generate",
        "janus_llama.generate",
    ]
