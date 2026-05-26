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

from typing import Any

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


def _config_dict() -> dict[str, Any]:
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
            "text_encoder": {"micro_batch_size": 4},
            "vision": {"micro_batch_size": 4},
            "vqvae": {"micro_batch_size": 4},
            "ar": {"micro_batch_size": 2},
        },
        "nodes": {
            "tok_encode": {"module": "text_encoder.encode"},
            "tok_decode": {"module": "text_encoder.decode"},
            "vis_encode": {"module": "vision"},
            "vae_encode": {"module": "vqvae.encode"},
            "vae_decode": {"module": "vqvae.decode"},
            "run_ar": {"module": "ar"},
            # Inference-only Janus boundary-token emitters (model-owned).
            "emit_image_start": {"module": "text_encoder.emit_image_start"},
            "emit_image_end": {"module": "text_encoder.emit_image_end"},
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


def _interleave_generation_graph() -> dict[str, Any]:
    """Interleave T2T+T2I FSM (matches infer_interleave.yaml shape).

    ``image_vq`` runs ``token_length: variable`` and exits when
    ``vae_decode`` writes ``ctx['image_complete'] = True`` — the
    PrintVQVAE module fakes this signal after ``image_steps`` consecutive
    decode calls (the test fixture wires that via ``_build_model``).
    """
    return {
        "initial": "text_ar",
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
                "token_length": {"type": "variable"},
                "transitions": [
                    {"condition": {"type": "ctx_flag", "key": "image_complete"}, "next_state": "image_vq_end"},
                ],
            },
            "image_vq_end": {
                "body": ["emit_end_to_ar", "emit_end_sink", "ar_run_sink"],
                "token_length": {"type": "fixed", "value": 1},
                "transitions": [{"condition": {"type": "steps_complete"}, "next_state": "text_ar"}],
            },
        },
    }


def _t2i_generation_graph() -> dict[str, Any]:
    """T2I-only FSM (matches infer_t2i.yaml shape).

    ``image_vq`` is variable-length and listens for the
    ``image_complete`` signal — same contract as `_interleave_generation_graph`.
    """
    return {
        "initial": "prompt_to_image",
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
                "token_length": {"type": "variable"},
                "transitions": [
                    {"condition": {"type": "ctx_flag", "key": "image_complete"}, "next_state": "image_vq_end"},
                ],
            },
            "image_vq_end": {
                "body": ["emit_end_to_ar", "emit_end_sink", "ar_run_sink"],
                "token_length": {"type": "fixed", "value": 1},
                "transitions": [{"condition": {"type": "steps_complete"}, "next_state": "done"}],
            },
        },
    }


def _understanding_generation_graph() -> dict[str, Any]:
    """I2T / VQA FSM (matches infer_understanding.yaml shape).

    The initial ``prompt_to_text`` state has TWO incoming routing edges
    into ``run_ar`` (vision_to_ar + tok_enc_to_ar) — exercises the
    topological body-execution rule.
    """
    return {
        "initial": "prompt_to_text",
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
        },
    }


def _build_model(
    token_script,
    generation_graph: dict[str, Any] | None = None,
    image_steps: int | None = None,
):
    """Construct an OmniModel with the print-only modules.

    ``image_steps`` configures PrintVQVAE to emit ``image_complete=True``
    after that many inference ``decode()`` calls — mirrors the real Janus
    VQ decoder's "image complete" signal that drives the
    ``image_vq → image_vq_end`` transition via ``ctx_flag``.
    """
    log: list[str] = []
    cfg_dict = _config_dict()
    if generation_graph is not None:
        cfg_dict["generation_graph"] = generation_graph
    cfg = OmniConfig.from_dict(cfg_dict)
    modules = {
        "text_encoder": PrintTextEmbed("text_encoder", log, token_script=token_script),
        "vision": PrintVisionEncoder("vision", log),
        "vqvae": PrintVQVAE("vqvae", log, image_steps=image_steps),
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

    trace: list[str] = []
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
        generation_graph=_interleave_generation_graph(),
        image_steps=3,  # PrintVQVAE emits `image_complete` on the 3rd decode
    )

    trace: list[str] = []
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
        + ["transition: image_vq -> image_vq_end [ctx_flag(image_complete)]"]
        + vq_end_step * 1
        + ["transition: image_vq_end -> text_ar [steps_complete]"]
        + text_ar_step * 3
        + [f"transition: text_ar -> done [token_match({TOK_EOS})]"]
    )
    assert trace == expected, "trace mismatch:\n" + "\n".join(trace)

    assert model.generation_graph.is_done()
    assert final_ctx["last_token_id"] == TOK_EOS
    # The `image_complete` flag was popped on transition — never leaks.
    assert "image_complete" not in final_ctx


def test_fsm_image_vq_routes_embed_back_into_inputs_embeds():
    """In ``image_vq``, edge ``vae_dec_to_ar`` writes ctx['embed'] → ctx['inputs_embeds']."""
    model, _ = _build_model(
        token_script=[TOK_BOI],
        generation_graph=_interleave_generation_graph(),
        image_steps=3,
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
        generation_graph=_interleave_generation_graph(),
        image_steps=1,
    )
    final_ctx = model.generate(
        request={"max_new_tokens": 50},
        context={"input_ids": "<bos>"},
        max_new_tokens=10,
    )
    # The emit_image_start was actually invoked (proves model-side ownership).
    assert any(evt.startswith("text_encoder.emit_boi(") for evt in log)
    assert any(evt.startswith("text_encoder.emit_eoi(") for evt in log)
    # The boundary token id flowed into ctx, not magically appended by FSM.
    assert final_ctx["last_token_id"] in (TOK_BOI, TOK_EOI, TOK_EOS, 2)


# ── Inference (FSM) flow — T2I-only ──────────────────────────────────────────


def test_fsm_t2i_only_starts_with_prompt_state_and_ends_after_image():
    """T2I-only FSM: prompt_to_image → bridges → image_vq → done."""
    model, _ = _build_model(
        token_script=[],  # never decode text
        generation_graph=_t2i_generation_graph(),
        image_steps=2,
    )

    trace: list[str] = []
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
        + ["transition: image_vq -> image_vq_end [ctx_flag(image_complete)]"]
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

    trace: list[str] = []
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


# ── ctx_flag transition (module-driven completion signal) ───────────────────


def test_fsm_ctx_flag_transition_fires_on_module_signal():
    """`image_vq` loops until vae_decode writes `ctx['image_complete']=True`.

    The PrintVQVAE mock counts inference `decode()` calls and emits the
    flag on call N — proving the FSM exits the variable-length state
    when (and only when) the module says it's done.
    """
    n_patches = 5
    model, _ = _build_model(
        token_script=[TOK_BOI],  # first text_ar step emits boi → enters image_vq
        generation_graph=_interleave_generation_graph(),
        image_steps=n_patches,
    )

    trace: list[str] = []
    model.generate(
        request={"max_new_tokens": 50},
        context={"input_ids": "<bos>"},
        max_new_tokens=50,
        trace=trace,
    )

    # vae_decode ran exactly `n_patches` times inside image_vq — i.e. the FSM
    # didn't exit early and didn't run an extra step after the signal.
    image_vq_decodes = [e for e in trace if e == "image_vq:vae_decode"]
    assert len(image_vq_decodes) == n_patches

    # The transition fired with the ctx_flag condition (not steps_complete).
    assert "transition: image_vq -> image_vq_end [ctx_flag(image_complete)]" in trace


def test_fsm_ctx_flag_cleared_after_transition():
    """Once `ctx_flag` fires, the framework pops the key — no stale reuse."""
    model, _ = _build_model(
        token_script=[TOK_BOI, TOK_EOS],  # → image, → done
        generation_graph=_interleave_generation_graph(),
        image_steps=2,
    )

    final_ctx = model.generate(
        request={"max_new_tokens": 50},
        context={"input_ids": "<bos>"},
        max_new_tokens=50,
    )
    # If the framework didn't pop, image_complete would still be True in the
    # final ctx — and worse, would have re-fired the flag check in subsequent
    # states had any of them listened for it.
    assert "image_complete" not in final_ctx


def test_fsm_ctx_flag_does_not_fire_until_module_writes_it():
    """A state with a `ctx_flag` transition stays in-state while the flag is absent."""
    # image_steps=None → PrintVQVAE never emits image_complete → image_vq
    # would loop forever.  Cap it via max_new_tokens and verify the state
    # never transitioned.
    model, _ = _build_model(
        token_script=[TOK_BOI],
        generation_graph=_interleave_generation_graph(),
        image_steps=None,
    )

    trace: list[str] = []
    model.generate(
        request={"max_new_tokens": 6},
        context={"input_ids": "<bos>"},
        max_new_tokens=6,  # 1 text_ar step + 1 image_vq_start + N image_vq steps
        trace=trace,
    )

    # The transition out of image_vq must NEVER appear — the flag was never set.
    assert "transition: image_vq -> image_vq_end [ctx_flag(image_complete)]" not in trace
    # And the FSM is still alive in image_vq (didn't reach `done`).
    assert not model.generation_graph.is_done()


def test_fsm_ctx_flag_condition_requires_key():
    """`ctx_flag` without a non-empty `key` is rejected at FSM build time."""
    import pytest

    fsm = _interleave_generation_graph()
    fsm["states"]["image_vq"]["transitions"] = [
        {"condition": {"type": "ctx_flag"}, "next_state": "image_vq_end"},
    ]
    with pytest.raises(ValueError, match=r"ctx_flag.*requires.*key"):
        _build_model(token_script=[], generation_graph=fsm)


def test_fsm_unknown_condition_type_rejected():
    """Typos / unsupported condition kinds fail loud at FSM build time."""
    import pytest

    fsm = _interleave_generation_graph()
    fsm["states"]["image_vq"]["transitions"] = [
        {"condition": {"type": "ctx_flagg", "key": "image_complete"}, "next_state": "image_vq_end"},
    ]
    with pytest.raises(ValueError, match="Unknown FSM condition type"):
        _build_model(token_script=[], generation_graph=fsm)


# ── Built-in `done` state + finalize hook ────────────────────────────────────


def test_fsm_done_state_is_framework_injected():
    """`done` is added by GenerationGraph even when the YAML never mentions it."""
    fsm = _interleave_generation_graph()
    # Sanity-check the fixture: the YAML does NOT carry a `done:` block.
    assert "done" not in fsm["states"]
    assert "done_state" not in fsm

    model, _ = _build_model(token_script=[], generation_graph=fsm)
    g = model.generation_graph
    # Yet the FSM still recognises `done` as a terminal state.
    assert g.is_done() is False
    assert g.state_node_sequence("done") == []
    # Transitions targeting `done` validate against the auto-injected state.
    transitions = [t for state in fsm["states"].values() for t in state["transitions"] if t["next_state"] == "done"]
    assert transitions, "fixture should have at least one transition to `done` to be meaningful"


def test_fsm_user_declared_done_state_rejected():
    """Authoring a `done:` block must raise — that's a stale-YAML signal."""
    import pytest

    fsm = _interleave_generation_graph()
    fsm["states"]["done"] = {"body": [], "token_length": {"type": "fixed", "value": 0}, "transitions": []}
    with pytest.raises(ValueError, match="reserved and auto-injected"):
        _build_model(token_script=[], generation_graph=fsm)


def test_fsm_done_state_config_knob_rejected():
    """`generation_graph.done_state` is gone — must raise to surface stale YAML."""
    import pytest

    fsm = _interleave_generation_graph()
    fsm["done_state"] = "done"
    with pytest.raises(ValueError, match="no longer configurable"):
        _build_model(token_script=[], generation_graph=fsm)


def test_finalize_hook_fires_on_done_and_collects_outputs():
    """`OmniModule.finalize` is called once when the FSM enters `done`.

    The print modules don't override finalize → default no-op → empty
    `finalize` dict in the trace and no `ctx['finalize']` injected.  We
    dynamically attach a custom finalize on one module and verify the
    framework collects its output.
    """
    model, _ = _build_model(token_script=[TOK_EOS], generation_graph=_interleave_generation_graph())

    # Default behaviour: no finalize outputs.
    trace_default: list[str] = []
    ctx_default = model.generate(request={}, context={"input_ids": "<bos>"}, max_new_tokens=5, trace=trace_default)
    assert "finalize" not in ctx_default
    assert not any(e.startswith("finalize:") for e in trace_default)

    # Inject a custom finalize on `text_encoder`.
    text_encoder = model.modules_dict["text_encoder"]
    text_encoder.finalize = lambda *, ctx, request: {"decoded": "hello world", "n_tokens": 1}

    trace_custom: list[str] = []
    ctx_custom = model.generate(
        request={"prompt": "hi"},
        context={"input_ids": "<bos>"},
        max_new_tokens=5,
        trace=trace_custom,
    )
    assert ctx_custom["finalize"] == {"text_encoder": {"decoded": "hello world", "n_tokens": 1}}
    assert "finalize:text_encoder" in trace_custom


def test_finalize_hook_rejects_non_dict_return():
    """Modules that mis-implement finalize get a clear error, not silent corruption."""
    import pytest

    model, _ = _build_model(token_script=[TOK_EOS], generation_graph=_interleave_generation_graph())
    text_encoder = model.modules_dict["text_encoder"]
    text_encoder.finalize = lambda *, ctx, request: "not a dict"

    with pytest.raises(TypeError, match="must return a dict"):
        model.generate(request={}, context={"input_ids": "<bos>"}, max_new_tokens=5)


def test_finalize_hook_receives_ctx_and_request():
    """The hook can read final ctx + the original request — accumulation is the module's job."""
    captured: list[dict[str, Any]] = []

    model, _ = _build_model(token_script=[TOK_EOS], generation_graph=_interleave_generation_graph())
    text_encoder = model.modules_dict["text_encoder"]

    def _capture_finalize(*, ctx: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        captured.append({"ctx_keys": sorted(ctx), "request": dict(request)})
        return {}

    text_encoder.finalize = _capture_finalize

    request = {"prompt": "describe", "max_new_tokens": 5}
    model.generate(request=request, context={"input_ids": "<bos>"}, max_new_tokens=5)

    assert len(captured) == 1
    assert captured[0]["request"] == request
    assert "input_ids" in captured[0]["ctx_keys"]


# ── Visualization smoke tests ────────────────────────────────────────────────


def test_training_graph_mermaid_renders_end_sink():
    model, _ = _build_model(token_script=[])
    txt = model.training_graph.to_mermaid(title="print-flow training")

    # New layout: LR flowchart with ELK renderer + per-rank subgraphs, no
    # `losses` collector (each module emits its own scalar _loss).
    assert txt.startswith("---\ntitle: print-flow training\n---\n")
    assert "flowchart LR" in txt
    # Active nodes labelled with module.method.
    assert "tok_decode<br/><i>text_encoder.decode</i>" in txt
    assert "vae_encode<br/><i>vqvae.encode</i>" in txt
    # `end` sink is rendered when at least one edge points to it.
    assert "end_sink" in txt
    assert "tok_decode --> end_sink" in txt
    assert "vae_decode --> end_sink" in txt
    assert "losses" not in txt


def test_generation_graph_mermaid_renders_body_subgraphs_and_loops():
    model, _ = _build_model(
        token_script=[],
        generation_graph=_interleave_generation_graph(),
    )
    txt = model.generation_graph.to_mermaid(title="print-flow inference FSM")

    # New layout: flowchart LR + ELK + body subgraphs.  No stateDiagram-v2.
    assert "stateDiagram-v2" not in txt
    assert "flowchart LR" in txt
    assert "%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%" in txt

    # Every non-done state is rendered as `subgraph state_<name> [<name>]`.
    for s in ("text_ar", "image_vq_start", "image_vq", "image_vq_end"):
        assert f"subgraph state_{s} [{s}]" in txt
    # `done` has no body — it is NOT drawn as a subgraph.  A small terminal
    # circle absorbs every transition that targets it.
    assert "subgraph state_done" not in txt
    assert 'fsm_done(("⏹"))' in txt

    # Inside each body, nodes are namespaced as `<state>__<node>` so the same
    # node can appear in multiple states without ID collisions.  Body nodes
    # carry the same `name<br/><i>module.method</i>` label as the training graph.
    assert 'text_ar__run_ar["run_ar<br/><i>ar.forward</i>"]' in txt
    assert "text_ar__tok_encode -->" in txt and "text_ar__run_ar" in txt

    # State transitions: thick `==>` arrows + quoted condition labels (so the
    # styling is visually distinct from intra-body `output → as` data edges).
    assert f'state_text_ar ==>|"token_match({TOK_BOI})"| state_image_vq_start' in txt
    # `image_vq` runs until vae_decode signals completion via `ctx_flag`.
    assert 'state_image_vq ==>|"ctx_flag(image_complete)"| state_image_vq_end' in txt
    # Transitions whose target is the `done` state route to the terminal node.
    assert f'state_text_ar ==>|"token_match({TOK_EOS})"| fsm_done' in txt

    # Self-loop iteration counts: `fixed: N` → label `×N`; `variable` → unlabelled.
    # `image_vq` is variable (loops until ctx_flag fires), `text_ar` is variable
    # (loops until token_match fires) → both unlabelled.
    assert "state_image_vq -.-> state_image_vq" in txt
    assert "state_text_ar -.-> state_text_ar" in txt
    # Bridge state has fixed=1 → labelled `×1`.
    assert 'state_image_vq_start -.->|"×1"| state_image_vq_start' in txt


# ── OmniModel topology contract ──────────────────────────────────────────────
#
# These tests pin down OmniModel's structural contract that downstream
# wiring (notably build_parallelize_model's per-sub-module weights_path
# dispatch in D2.2) depends on:
#   * sub-modules are direct children — not nested under `modules_dict.<name>`;
#   * named_parameters fqns flatten to `<name>.<rest>`;
#   * the legacy `model.modules_dict[name]` view still works for back-compat.


def test_omni_model_named_children_yields_submodules_directly():
    """``model.named_children()`` enumerates sub-modules in declared order
    with no `modules_dict.` middle layer — this is what
    `build_parallelize_model(weights_path: Mapping[str, str])` will key off
    in D2.2 to dispatch per-sub-module weight loading.
    """
    model, _ = _build_model(token_script=[])

    names = [name for name, _ in model.named_children()]

    # Order matches OmniConfig.module_names declaration order.
    assert names == ["text_encoder", "vision", "vqvae", "ar"]

    # No nn.ModuleDict middle attribute leaks into the children iteration.
    assert "modules_dict" not in names

    # Each named child IS the actual OmniModule instance — no wrapping.
    for name, child in model.named_children():
        assert child is getattr(model, name)


def test_omni_model_named_parameters_fqn_lacks_modules_dict_prefix():
    """Parameter fqns shape as `<sub_module>.<rest>` so checkpoint shard
    names map 1:1 to sub-module names without prefix-stripping logic.
    """
    model, _ = _build_model(token_script=[])

    # Print modules carry one trainable buffer-as-parameter (see
    # PrintTextEmbed).  We just need at least one fqn to assert the shape.
    fqns = [n for n, _ in model.named_parameters()]
    if fqns:
        for fqn in fqns:
            head = fqn.split(".", 1)[0]
            assert head in {"text_encoder", "vision", "vqvae", "ar"}, (
                f"fqn {fqn!r} starts with {head!r} — should be a top-level "
                f"sub-module name, not 'modules_dict' or any wrapper."
            )


def test_omni_model_modules_dict_is_back_compat_view():
    """`model.modules_dict[name]` still works for callers that haven't been
    migrated yet (e.g. test fixtures that mutate a sub-module).  The view
    returns the same instance as `getattr(model, name)`.
    """
    model, _ = _build_model(token_script=[])

    view = model.modules_dict
    # Plain dict view — not an nn.ModuleDict that would re-register children.
    assert isinstance(view, dict)
    assert set(view) == {"text_encoder", "vision", "vqvae", "ar"}
    for name, mod in view.items():
        assert mod is getattr(model, name)


def test_omni_model_rejects_module_name_colliding_with_framework_attr():
    """Sub-module names that collide with OmniModel's own attributes are
    rejected loudly — preventing `add_module('config', ...)` from silently
    overwriting ``self.config`` (the kind of bug that surfaces only when
    something downstream reads it).

    Renames ``text_encoder → config`` everywhere in the print-flow YAML
    fixture (modules, every node's ``module`` reference, every edge's
    ``from`` / ``to`` reference) to construct a config that *would*
    succeed without the guard, then asserts the guard fires.
    """
    import pytest

    def _rename(s: str, old: str, new: str) -> str:
        # `text_encoder` may appear as either the bare module name (`module: text_encoder`)
        # or as the prefix in `module: text_encoder.encode` etc.  Replace conservatively
        # at word boundaries so we don't touch `text_ar` etc.
        if s == old:
            return new
        if s.startswith(old + "."):
            return new + s[len(old) :]
        return s

    cfg_dict = _config_dict()
    cfg_dict["modules"] = {("config" if k == "text_encoder" else k): v for k, v in cfg_dict["modules"].items()}
    cfg_dict["nodes"] = {
        n_name: {**n_def, "module": _rename(n_def["module"], "text_encoder", "config")}
        for n_name, n_def in cfg_dict["nodes"].items()
    }
    cfg = OmniConfig.from_dict(cfg_dict)

    log: list[str] = []
    modules = {
        "config": PrintTextEmbed("config", log, token_script=[]),
        "vision": PrintVisionEncoder("vision", log),
        "vqvae": PrintVQVAE("vqvae", log),
        "ar": PrintARBackbone("ar", log),
    }
    with pytest.raises(ValueError, match="collide with framework attribute"):
        OmniModel(cfg, modules)
