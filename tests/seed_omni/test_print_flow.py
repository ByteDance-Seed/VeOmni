"""End-to-end print-driven flow tests for the SeedOmni V2 graph runtime.

Scope
-----
These tests exercise the *graph behaviour* — TrainingGraph topo order,
GenerationGraph FSM step / transition / topo-execution / permissive
edge routing semantics, ``_loss`` aggregation, edge routing including
the virtual ``end`` keyword — without any real ML modules.  Every
module is a :mod:`print_modules` stand-in that records its calls into
a shared log; assertions then read the log to verify the framework
executes things in the expected order.

Coverage
--------
* **Training**: a 4-module DAG (text-embed, vision, vqvae, AR backbone)
  with two leaf nodes ``tok_decode`` and ``vae_decode`` flowing to
  ``end``.  Verifies topo order and that ``_loss`` from each loss-emitting
  module sums into the total scalar.

* **Inference (interleave)**: 5-state FSM (``text_ar`` →
  ``image_vq_start`` → ``image_vq`` → ``image_vq_end`` → ``text_ar`` →
  ``done``) covering boundary-token emission via dedicated
  ``emit_image_start`` / ``emit_image_end`` nodes — the framework has
  no special-cased boundary logic.

* **Inference (T2I-only)**: 5-state FSM that starts with
  ``prompt_to_image`` and terminates after the image span — no
  subsequent text generation.

* **Inference (understanding)**: tests the **multi-source** initial
  state (``prompt_to_text`` with both vision_to_ar and tok_enc_to_ar
  feeding ``run_ar``) — verifies the topological body-execution rule
  (run_ar fires only after BOTH incoming edges have been processed),
  and the **permissive routing** behaviour when no image is present
  (siglip returns ``{}``, edge silently skips, run_ar still executes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from veomni.models.seed_omni import OmniConfig, OmniModel

from .print_modules import (
    PrintARBackbone,
    PrintTextEmbed,
    PrintVisionEncoder,
    PrintVQVAE,
)


# ── Special token ids used in the FSM transitions ────────────────────────────
TOK_BOI = 100016  # <begin_of_image> — text_ar → image_vq_start
TOK_EOI = 100593  # <end_of_image>   — emitted by emit_image_end
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
            # Inference-only Janus boundary-token emitters (model-owned).
            "emit_image_start": {"module": "text_embed.emit_image_start"},
            "emit_image_end": {"module": "text_embed.emit_image_end"},
        },
        "edges": {
            # ── training (and shared with inference)
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
            # ── inference-only edges
            "vae_dec_to_ar": {
                "from": "vae_decode",
                "output": "embed",
                "to": "run_ar",
                "as": "inputs_embeds",
            },
            "emit_start_to_ar": {
                "from": "emit_image_start",
                "output": "inputs_embeds",
                "to": "run_ar",
                "as": "inputs_embeds",
            },
            "emit_end_to_ar": {
                "from": "emit_image_end",
                "output": "inputs_embeds",
                "to": "run_ar",
                "as": "inputs_embeds",
            },
            "emit_start_sink": {"from": "emit_image_start", "to": "end"},
            "emit_end_sink": {"from": "emit_image_end", "to": "end"},
            # Body terminal for `run_ar` in bridge / prompt-priming states
            # where the LLM updates its KV cache without any same-body
            # consumer (image_vq_start, image_vq_end, prompt_to_image).
            "ar_run_sink": {"from": "run_ar", "to": "end"},
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
    }


# ── Inference YAML builders (mirrors configs/seed_omni/janus_1.3b/infer_*.yaml) ──


def _interleave_generation_graph(image_steps: int = 3) -> Dict[str, Any]:
    """Interleave T2T+T2I FSM (matches infer_interleave.yaml shape)."""
    return {
        "initial": "text_ar",
        "done_state": "done",
        "states": {
            "text_ar": {
                "body": ["tok_enc_to_ar", "ar_to_tok_dec", "tok_dec_sink"],
                "token_length": {"type": "variable"},
                "transitions": [
                    {"condition": {"type": "token_match", "token_id": TOK_BOI}, "next_state": "image_vq_start"},
                    {"condition": {"type": "token_match", "token_id": TOK_EOS}, "next_state": "done"},
                ],
            },
            "image_vq_start": {
                "body": ["emit_start_to_ar", "emit_start_sink", "ar_run_sink"],
                "token_length": {"type": "fixed", "value": 1},
                "transitions": [{"condition": {"type": "steps_complete"}, "next_state": "image_vq"}],
            },
            "image_vq": {
                "body": ["ar_to_vae_dec", "vae_dec_to_ar"],
                "token_length": {"type": "fixed", "value": image_steps},
                "transitions": [{"condition": {"type": "steps_complete"}, "next_state": "image_vq_end"}],
            },
            "image_vq_end": {
                "body": ["emit_end_to_ar", "emit_end_sink", "ar_run_sink"],
                "token_length": {"type": "fixed", "value": 1},
                "transitions": [{"condition": {"type": "steps_complete"}, "next_state": "text_ar"}],
            },
            "done": {"body": [], "token_length": {"type": "fixed", "value": 0}, "transitions": []},
        },
    }


def _t2i_generation_graph(image_steps: int = 3) -> Dict[str, Any]:
    """T2I-only FSM (matches infer_t2i.yaml shape)."""
    return {
        "initial": "prompt_to_image",
        "done_state": "done",
        "states": {
            "prompt_to_image": {
                "body": ["tok_enc_to_ar", "ar_run_sink"],  # seed KV cache; no decoding
                "token_length": {"type": "fixed", "value": 1},
                "transitions": [{"condition": {"type": "steps_complete"}, "next_state": "image_vq_start"}],
            },
            "image_vq_start": {
                "body": ["emit_start_to_ar", "emit_start_sink", "ar_run_sink"],
                "token_length": {"type": "fixed", "value": 1},
                "transitions": [{"condition": {"type": "steps_complete"}, "next_state": "image_vq"}],
            },
            "image_vq": {
                "body": ["ar_to_vae_dec", "vae_dec_to_ar"],
                "token_length": {"type": "fixed", "value": image_steps},
                "transitions": [{"condition": {"type": "steps_complete"}, "next_state": "image_vq_end"}],
            },
            "image_vq_end": {
                "body": ["emit_end_to_ar", "emit_end_sink", "ar_run_sink"],
                "token_length": {"type": "fixed", "value": 1},
                "transitions": [{"condition": {"type": "steps_complete"}, "next_state": "done"}],
            },
            "done": {"body": [], "token_length": {"type": "fixed", "value": 0}, "transitions": []},
        },
    }


def _understanding_generation_graph() -> Dict[str, Any]:
    """I2T / VQA FSM (matches infer_understanding.yaml shape).

    The initial ``prompt_to_text`` state has TWO incoming routing edges
    into ``run_ar`` (vision_to_ar + tok_enc_to_ar) — exercises the
    topological body-execution rule.
    """
    return {
        "initial": "prompt_to_text",
        "done_state": "done",
        "states": {
            "prompt_to_text": {
                "body": ["vis_to_ar", "tok_enc_to_ar", "ar_to_tok_dec", "tok_dec_sink"],
                "token_length": {"type": "fixed", "value": 1},
                "transitions": [
                    {"condition": {"type": "token_match", "token_id": TOK_EOS}, "next_state": "done"},
                    {"condition": {"type": "steps_complete"}, "next_state": "text_ar"},
                ],
            },
            "text_ar": {
                "body": ["tok_enc_to_ar", "ar_to_tok_dec", "tok_dec_sink"],
                "token_length": {"type": "variable"},
                "transitions": [
                    {"condition": {"type": "token_match", "token_id": TOK_EOS}, "next_state": "done"},
                ],
            },
            "done": {"body": [], "token_length": {"type": "fixed", "value": 0}, "transitions": []},
        },
    }


def _build_model(token_script, generation_graph: Optional[Dict[str, Any]] = None):
    log: List[str] = []
    cfg_dict = _config_dict()
    if generation_graph is not None:
        cfg_dict["generation_graph"] = generation_graph
    cfg = OmniConfig.from_dict(cfg_dict)
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


# ── Inference (FSM) flow — interleave T2T + T2I ──────────────────────────────


def test_fsm_interleave_text_to_image_to_text():
    """Token script: 10, 11, <boi>, 20, 21, </s>.

    Expected behaviour (interleave FSM with bridge states):

        text_ar         iter 1: tok=10         → no transition
        text_ar         iter 2: tok=11         → no transition
        text_ar         iter 3: tok=<boi>      → image_vq_start
        image_vq_start  iter 1                  → steps_complete → image_vq
        image_vq        iter 1..3               → steps_complete → image_vq_end
        image_vq_end    iter 1                  → steps_complete → text_ar
        text_ar         iter 4: tok=20         → no transition
        text_ar         iter 5: tok=21         → no transition
        text_ar         iter 6: tok=</s>       → done
    """
    model, _ = _build_model(
        token_script=[10, 11, TOK_BOI, 20, 21, TOK_EOS],
        generation_graph=_interleave_generation_graph(image_steps=3),
    )

    trace: List[str] = []
    final_ctx = model.generate(
        request={"max_new_tokens": 50},
        context={"input_ids": "<bos>", "attention_mask": "<mask>"},
        max_new_tokens=50,
        trace=trace,
    )

    text_ar_step = ["text_ar:tok_encode", "text_ar:run_ar", "text_ar:tok_decode"]
    # Bridge state body order: [emit_*, emit_sink, ar_run_sink].  The emitter
    # runs on the first edge; ar_run_sink fires run_ar last.
    vq_start_step = ["image_vq_start:emit_image_start", "image_vq_start:run_ar"]
    image_vq_step = ["image_vq:run_ar", "image_vq:vae_decode"]
    vq_end_step = ["image_vq_end:emit_image_end", "image_vq_end:run_ar"]

    expected = (
        text_ar_step * 3
        + [f"transition: text_ar -> image_vq_start [token_match({TOK_BOI})]"]
        + vq_start_step * 1
        + ["transition: image_vq_start -> image_vq [steps_complete]"]
        + image_vq_step * 3
        + ["transition: image_vq -> image_vq_end [steps_complete]"]
        + vq_end_step * 1
        + ["transition: image_vq_end -> text_ar [steps_complete]"]
        + text_ar_step * 3
        + [f"transition: text_ar -> done [token_match({TOK_EOS})]"]
    )
    assert trace == expected, "trace mismatch:\n" + "\n".join(trace)

    assert model.generation_graph.is_done()
    assert final_ctx["last_token_id"] == TOK_EOS


def test_fsm_image_vq_routes_embed_back_into_inputs_embeds():
    """In ``image_vq``, edge ``vae_dec_to_ar`` writes ctx['embed'] → ctx['inputs_embeds']."""
    model, _ = _build_model(
        token_script=[TOK_BOI],
        generation_graph=_interleave_generation_graph(image_steps=3),
    )

    final_ctx = model.generate(
        request={"max_new_tokens": 50},
        context={"input_ids": "<bos>"},
        # iter 1: text_ar emits boi → transition; iter 2: image_vq_start; iter 3: image_vq once.
        max_new_tokens=3,
    )

    # The vae_dec_to_ar edge fires AFTER vae_decode runs in image_vq.
    assert final_ctx["embed"] == "<vq_decode_embed>"
    assert final_ctx["inputs_embeds"] == "<vq_decode_embed>"


def test_fsm_emit_image_start_runs_inside_bridge_body():
    """The boundary-token emitter is just another node — no FSM magic."""
    model, log = _build_model(
        token_script=[TOK_BOI],
        generation_graph=_interleave_generation_graph(image_steps=1),
    )
    final_ctx = model.generate(
        request={"max_new_tokens": 50},
        context={"input_ids": "<bos>"},
        max_new_tokens=10,
    )
    # The emit_image_start was actually invoked (proves model-side ownership).
    assert any(evt.startswith("text_embed.emit_boi(") for evt in log)
    assert any(evt.startswith("text_embed.emit_eoi(") for evt in log)
    # The boundary token id flowed into ctx, not magically appended by FSM.
    assert final_ctx["last_token_id"] in (TOK_BOI, TOK_EOI, TOK_EOS, 2)


# ── Inference (FSM) flow — T2I-only ──────────────────────────────────────────


def test_fsm_t2i_only_starts_with_prompt_state_and_ends_after_image():
    """T2I-only FSM: prompt_to_image → bridges → image_vq → done."""
    model, _ = _build_model(
        token_script=[],  # never decode text
        generation_graph=_t2i_generation_graph(image_steps=2),
    )

    trace: List[str] = []
    model.generate(
        request={"max_new_tokens": 50},
        context={"input_ids": "<bos>"},
        max_new_tokens=20,
        trace=trace,
    )

    expected = (
        # prompt_to_image: tok_encode → run_ar (ar_run_sink triggers run_ar; no decode).
        ["prompt_to_image:tok_encode", "prompt_to_image:run_ar"]
        + ["transition: prompt_to_image -> image_vq_start [steps_complete]"]
        + ["image_vq_start:emit_image_start", "image_vq_start:run_ar"]
        + ["transition: image_vq_start -> image_vq [steps_complete]"]
        + ["image_vq:run_ar", "image_vq:vae_decode"] * 2
        + ["transition: image_vq -> image_vq_end [steps_complete]"]
        + ["image_vq_end:emit_image_end", "image_vq_end:run_ar"]
        + ["transition: image_vq_end -> done [steps_complete]"]
    )
    assert trace == expected, "trace mismatch:\n" + "\n".join(trace)
    assert model.generation_graph.is_done()


# ── Inference (FSM) flow — understanding (multi-source + permissive) ─────────


def test_fsm_understanding_multi_source_runs_ar_after_both_inputs_route():
    """``prompt_to_text`` body: vis_to_ar → tok_enc_to_ar → ar_to_tok_dec → sink.

    Topological execution rule: ``run_ar`` has fan-in 2 (vis_to_ar +
    tok_enc_to_ar).  It must execute exactly ONCE, **after** the second
    routing edge has been processed — at which point both
    ``und_image_embeds`` and ``inputs_embeds`` are present in ctx.
    """
    model, log = _build_model(
        token_script=[42, TOK_EOS],
        generation_graph=_understanding_generation_graph(),
    )

    trace: List[str] = []
    model.generate(
        request={"max_new_tokens": 10},
        context={"input_ids": "<bos>", "pixel_values": "<pix>"},
        max_new_tokens=10,
        trace=trace,
    )

    # Each visited node should appear once per FSM iteration.  The order
    # within `prompt_to_text` is: vis_encode (when vis_to_ar starts),
    # tok_encode (when tok_enc_to_ar starts), then run_ar (after BOTH
    # incoming edges have been processed), then tok_decode.
    assert trace[:4] == [
        "prompt_to_text:vis_encode",
        "prompt_to_text:tok_encode",
        "prompt_to_text:run_ar",
        "prompt_to_text:tok_decode",
    ], trace[:4]

    # run_ar received BOTH inputs_embeds and und_image_embeds in this iteration.
    ar_call = next(evt for evt in log if "ar.generate_step(" in evt)
    assert "inputs_embeds" in ar_call
    assert "und_image_embeds" in ar_call


def test_fsm_understanding_text_only_prompt_uses_permissive_routing():
    """Text-only initial context (no ``pixel_values``).

    siglip returns ``{}``; the ``vis_to_ar`` edge has no ``image_embeds``
    in ctx so routing silently skips.  ``run_ar`` still executes (its
    fan-in counter still decrements regardless of whether the route
    succeeded), and ``und_image_embeds`` simply isn't passed as a kwarg.
    """
    model, log = _build_model(
        token_script=[7, TOK_EOS],
        generation_graph=_understanding_generation_graph(),
    )

    model.generate(
        request={"max_new_tokens": 10},
        context={"input_ids": "<bos>"},  # no pixel_values
        max_new_tokens=10,
    )

    # vis_encode was invoked once (the FSM passes whatever's in ctx as
    # kwargs; with no `pixel_values` field, the call lands on the
    # ``return {}`` fast path).
    vis_calls = [evt for evt in log if evt.startswith("vision.forward(")]
    assert len(vis_calls) == 1
    assert "pixel_values" not in vis_calls[0]

    # vis_to_ar therefore had nothing to route — `image_embeds` was never
    # in ctx — and run_ar ran without `und_image_embeds`.
    ar_calls = [evt for evt in log if "ar.generate_step(" in evt]
    assert any("inputs_embeds" in c and "und_image_embeds" not in c for c in ar_calls), ar_calls


# ── State node-sequence accessor (uses the topological order over body) ──────


def test_fsm_state_node_sequence_derived():
    """text_ar's body lists three edges; node sequence dedups endpoints."""
    model, _ = _build_model(
        token_script=[],
        generation_graph=_interleave_generation_graph(),
    )
    g = model.generation_graph

    assert g.state_node_sequence("text_ar") == ["tok_encode", "run_ar", "tok_decode"]
    assert g.state_node_sequence("image_vq") == ["run_ar", "vae_decode"]
    # Bridge bodies: emit_image_*, run_ar (the run_ar comes from ar_run_sink).
    assert g.state_node_sequence("image_vq_start") == ["emit_image_start", "run_ar"]
    assert g.state_node_sequence("image_vq_end") == ["emit_image_end", "run_ar"]
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
    model, _ = _build_model(
        token_script=[],
        generation_graph=_interleave_generation_graph(),
    )
    txt = model.generation_graph.to_mermaid(title="print-flow inference FSM")

    assert "stateDiagram-v2" in txt
    for s in ("text_ar", "image_vq_start", "image_vq", "image_vq_end", "done"):
        assert s in txt
    # Transitions labelled with conditions.
    assert f"text_ar --> image_vq_start : token_match({TOK_BOI})" in txt
    assert "image_vq --> image_vq_end : steps_complete" in txt
