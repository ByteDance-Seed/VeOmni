"""End-to-end print-driven flow tests for the SeedOmni V2 graph runtime.

Scope
-----
These tests exercise the *graph behaviour* — TrainingGraph topo order,
GenerationGraph FSM step / transition semantics, ``_loss`` aggregation,
edge routing including the virtual ``end`` keyword — without any real
ML modules.  Every module is a :mod:`print_modules` stand-in that
records its calls into a shared log; assertions then read the log to
verify the framework executes things in the expected order.

Two scenarios are covered, using a single Janus-style OmniConfig:

* **Training**: a 4-module DAG (text-embed, vision, vqvae, AR backbone)
  with two leaf nodes ``tok_decode`` and ``vae_decode`` flowing to
  ``end``.  Verifies topo order and that ``_loss`` from each loss-emitting
  module sums into the total scalar.

* **Inference**: a 3-state FSM (``text_ar`` → ``image_vq`` → ``text_ar``
  → ``done``) driven by a deterministic token script in
  :class:`PrintTextEmbed`.  Verifies FSM step-execution rules (``from``
  / ``to`` first-encounter dispatch, edge ctx routing) and that
  transitions fire on the right conditions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from veomni.models.seed_omni import OmniConfig, OmniModel

from .print_modules import (
    PrintARBackbone,
    PrintTextEmbed,
    PrintVisionEncoder,
    PrintVQVAE,
)


# ── Special token ids used in the FSM transitions ────────────────────────────
TOK_GEN_IMG = 100578  # text_ar → image_vq
TOK_EOS = 2  # text_ar → done


# ── Config builder ───────────────────────────────────────────────────────────


def _config_dict() -> Dict[str, Any]:
    """Janus-style schema used by both training and inference scenarios.

    Notes
    -----
    * ``training_graph.edges`` lists every edge the trainer should walk —
      including two ``to: end`` sinks (``tok_dec_sink``, ``vae_dec_sink``)
      that pin the leaf nodes into the active set without routing
      anywhere.
    * ``generation_graph.states.<name>.body`` lists ONLY edges (the
      design forbids node names in body).
    * Token-length ``fixed: 3`` for ``image_vq`` keeps the test trace
      compact while still exercising the steps_complete transition.
    """
    return {
        "modules": {
            "text_embed": {"micro_batch_size": 4},
            "vision": {"micro_batch_size": 4},
            "vqvae": {"micro_batch_size": 4},
            "ar": {"micro_batch_size": 2},
        },
        "nodes": {
            "tok_encode": {"module": "text_embed.encode"},
            "tok_decode": {"module": "text_embed.decode"},
            "vis_encode": {"module": "vision"},
            "vae_encode": {"module": "vqvae.encode"},
            "vae_decode": {"module": "vqvae.decode"},
            "run_ar": {"module": "ar"},
        },
        "edges": {
            # ── training-only edges ──
            "vis_to_ar": {
                "from": "vis_encode",
                "output": "image_embeds",
                "to": "run_ar",
                "as": "und_image_embeds",
            },
            "vae_enc_to_ar": {
                "from": "vae_encode",
                "output": "gen_embeds",
                "to": "run_ar",
                "as": "gen_image_embeds",
            },
            "tok_enc_to_ar": {
                "from": "tok_encode",
                "output": "inputs_embeds",
                "to": "run_ar",
                "as": "inputs_embeds",
            },
            "ar_to_tok_dec": {
                "from": "run_ar",
                "output": "hidden_states",
                "to": "tok_decode",
                "as": "hidden_states",
            },
            "ar_to_vae_dec": {
                "from": "run_ar",
                "output": "hidden_states",
                "to": "vae_decode",
                "as": "hidden_states",
            },
            "vae_token_to_dec": {
                "from": "vae_encode",
                "output": "vq_token_ids",
                "to": "vae_decode",
                "as": "gt_token_ids",
            },
            "tok_dec_sink": {"from": "tok_decode", "to": "end"},
            "vae_dec_sink": {"from": "vae_decode", "to": "end"},
            # ── inference-only feedback ──
            "vae_dec_to_ar": {
                "from": "vae_decode",
                "output": "embed",
                "to": "run_ar",
                "as": "inputs_embeds",
            },
        },
        "training_graph": {
            "edges": [
                "vis_to_ar",
                "vae_enc_to_ar",
                "tok_enc_to_ar",
                "ar_to_tok_dec",
                "ar_to_vae_dec",
                "vae_token_to_dec",
                "tok_dec_sink",
                "vae_dec_sink",
            ],
        },
        "generation_graph": {
            "initial": "text_ar",
            "done_state": "done",
            "states": {
                "text_ar": {
                    "body": ["tok_enc_to_ar", "ar_to_tok_dec"],
                    "token_length": {"type": "variable"},
                    "transitions": [
                        {
                            "condition": {"type": "token_match", "token_id": TOK_GEN_IMG},
                            "next_state": "image_vq",
                        },
                        {
                            "condition": {"type": "token_match", "token_id": TOK_EOS},
                            "next_state": "done",
                        },
                    ],
                },
                "image_vq": {
                    "body": ["ar_to_vae_dec", "vae_dec_to_ar"],
                    "token_length": {"type": "fixed", "value": 3},
                    "transitions": [
                        {
                            "condition": {"type": "steps_complete"},
                            "next_state": "text_ar",
                        },
                    ],
                },
                "done": {
                    "body": [],
                    "token_length": {"type": "fixed", "value": 0},
                    "transitions": [],
                },
            },
        },
    }


def _build_model(token_script):
    log: List[str] = []
    cfg = OmniConfig.from_dict(_config_dict())
    modules = {
        "text_embed": PrintTextEmbed("text_embed", log, token_script=token_script),
        "vision": PrintVisionEncoder("vision", log),
        "vqvae": PrintVQVAE("vqvae", log),
        "ar": PrintARBackbone("ar", log),
    }
    return OmniModel(cfg, modules), log


# ── Training (DAG) flow ──────────────────────────────────────────────────────


def test_training_graph_topology_and_active_nodes():
    model, _ = _build_model(token_script=[])
    g = model.training_graph

    order = g.execution_order
    assert set(order) == {"vis_encode", "vae_encode", "tok_encode", "run_ar", "tok_decode", "vae_decode"}

    # Source nodes must precede run_ar; tok_decode / vae_decode must follow.
    for src in ("vis_encode", "vae_encode", "tok_encode"):
        assert order.index(src) < order.index("run_ar"), order
    for sink in ("tok_decode", "vae_decode"):
        assert order.index(sink) > order.index("run_ar"), order

    assert set(g.sources) == {"vis_encode", "vae_encode", "tok_encode"}
    assert set(g.sinks) == {"tok_decode", "vae_decode"}


def test_training_forward_calls_each_node_once_in_order():
    model, log = _build_model(token_script=[])

    trace: List[str] = []
    model(
        trace=trace,
        input_ids=10,
        pixel_values="<pix>",
        labels="<labels>",
    )

    # Each node executed exactly once, in topo order.
    expected_nodes = list(model.training_graph.execution_order)
    assert trace == [f"forward:{n}" for n in expected_nodes], trace

    # vqvae appears under TWO node names — one underlying instance, two calls.
    vqvae_calls = [evt for evt in log if evt.startswith("vqvae.")]
    assert len(vqvae_calls) == 2
    assert any("vqvae.encode(" in evt for evt in vqvae_calls)
    assert any("vqvae.decode(" in evt for evt in vqvae_calls)

    # Edge routing carried strings unchanged into run_ar.
    ar_kwargs = next(evt for evt in log if evt.startswith("ar.forward("))
    for k in ("inputs_embeds", "und_image_embeds", "gen_image_embeds"):
        assert k in ar_kwargs


def test_training_forward_aggregates_single_loss_per_module():
    model, _ = _build_model(token_script=[])

    out = model(input_ids=10, pixel_values="<pix>", labels="<labels>")

    # Each loss-emitting node contributes one scalar; they are summed.
    losses = out["losses"]
    assert set(losses.keys()) == {"run_ar", "tok_decode", "vae_decode"}
    assert all(t.ndim == 0 for t in losses.values())

    expected = sum(t.item() for t in losses.values())
    assert isinstance(out["loss"], torch.Tensor)
    assert out["loss"].ndim == 0
    assert abs(out["loss"].item() - expected) < 1e-6


# ── Inference (FSM) flow ─────────────────────────────────────────────────────


def test_fsm_transitions_text_ar_to_image_vq_and_back():
    """Token script: 10, 11, <gen_img>, 20, 21, </s>.

    Expected behaviour:
        text_ar  iter 1: tok=10            → no transition
        text_ar  iter 2: tok=11            → no transition
        text_ar  iter 3: tok=<gen_img>     → transition to image_vq
        image_vq iter 1..3                  → steps_complete → text_ar
        text_ar  iter 4: tok=20            → no transition
        text_ar  iter 5: tok=21            → no transition
        text_ar  iter 6: tok=</s>          → transition to done → stop
    """
    model, _ = _build_model(token_script=[10, 11, TOK_GEN_IMG, 20, 21, TOK_EOS])

    trace: List[str] = []
    final_ctx = model.generate(
        request={"max_new_tokens": 50},
        context={"input_ids": "<bos>", "attention_mask": "<mask>"},
        max_new_tokens=50,
        trace=trace,
    )

    text_ar_step = ["text_ar:tok_encode", "text_ar:run_ar", "text_ar:tok_decode"]
    image_vq_step = ["image_vq:run_ar", "image_vq:vae_decode"]
    expected = (
        # Three text_ar iterations until <gen_img>.
        text_ar_step * 3
        + ["transition: text_ar -> image_vq [token_match(100578)]"]
        # Three image_vq iterations until steps_complete.
        + image_vq_step * 3
        + ["transition: image_vq -> text_ar [steps_complete]"]
        # Three more text_ar iterations: 20, 21, </s>.
        + text_ar_step * 3
        + ["transition: text_ar -> done [token_match(2)]"]
    )
    assert trace == expected, "trace mismatch:\n" + "\n".join(trace)

    # FSM correctly stopped at the configured done state.
    assert model.generation_graph.is_done()
    # Final ctx carries the last sampled token.
    assert final_ctx["last_token_id"] == TOK_EOS


def test_fsm_image_vq_routes_embed_back_into_inputs_embeds():
    """In ``image_vq``, edge ``vae_dec_to_ar`` writes ctx['embed'] → ctx['inputs_embeds'].

    Verifying: after one iteration in image_vq, ``inputs_embeds`` in ctx
    equals the stand-in produced by :meth:`PrintVQVAE.decode`.
    """
    model, _ = _build_model(token_script=[TOK_GEN_IMG])

    trace: List[str] = []
    # Run only enough steps to reach image_vq for one iteration.
    final_ctx = model.generate(
        request={"max_new_tokens": 50},
        context={"input_ids": "<bos>"},
        max_new_tokens=2,  # iter 1: text_ar emits gen_img → transition; iter 2: image_vq runs once
        trace=trace,
    )

    # The vae_dec_to_ar edge fires AFTER vae_decode runs, copying
    # ctx['embed'] into ctx['inputs_embeds'].
    assert final_ctx["embed"] == "<vq_decode_embed>"
    assert final_ctx["inputs_embeds"] == "<vq_decode_embed>"


def test_fsm_unwound_state_node_sequence_is_derived():
    """text_ar's body is two edges; node sequence derives unique endpoints."""
    model, _ = _build_model(token_script=[])
    g = model.generation_graph

    assert g.state_node_sequence("text_ar") == ["tok_encode", "run_ar", "tok_decode"]
    assert g.state_node_sequence("image_vq") == ["run_ar", "vae_decode"]
    assert g.state_node_sequence("done") == []


# ── Visualization smoke tests ────────────────────────────────────────────────


def test_training_graph_mermaid_renders_end_sink():
    model, _ = _build_model(token_script=[])
    txt = model.training_graph.to_mermaid(title="print-flow training")

    assert txt.startswith("---\ntitle: print-flow training\n---\nflowchart TD")
    # Active nodes labelled with module.method.
    assert "tok_decode<br/><i>text_embed.decode</i>" in txt
    assert "vae_encode<br/><i>vqvae.encode</i>" in txt
    # `end` sink is rendered when at least one edge points to it.
    assert "end_sink" in txt
    assert "tok_decode --> end_sink" in txt
    assert "vae_decode --> end_sink" in txt


def test_generation_graph_mermaid_lists_node_sequence():
    model, _ = _build_model(token_script=[])
    txt = model.generation_graph.to_mermaid(title="print-flow inference FSM")

    assert "stateDiagram-v2" in txt
    assert "text_ar" in txt and "image_vq" in txt and "done" in txt
    # Derived node sequences embedded in state labels.
    assert "tok_encode → run_ar → tok_decode" in txt
    assert "run_ar → vae_decode" in txt
    # Transitions labelled with conditions.
    assert "text_ar --> image_vq : token_match(100578)" in txt
    assert "image_vq --> text_ar : steps_complete" in txt
