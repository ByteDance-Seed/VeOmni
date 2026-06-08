"""
Visualize OmniModel graphs from a VeOmni launcher YAML.

Contract
--------
Pass a single launcher YAML (e.g. ``configs/seed_omni/janus_1.3b/veomni_janus.yaml``)
as the positional ``config_file`` argument.  Its ``model:`` section is consumed
by :class:`OmniInferModelArguments` (slim: only the four graph-pointing fields),
specifically:

* ``model.omni_train_yaml_path`` — master training vocabulary
* ``model.omni_infer_yaml_path`` — dict of scenario → inference YAML

and writes **four** diagrams to ``graphs/<yaml_basename>/`` (folder name
matches the launcher YAML stem, e.g. ``veomni_janus.yaml`` → ``graphs/veomni_janus/``):

1. ``training.{html|mmd}`` — training DAG from ``training_graph``
2. ``<infer_key>.{html|mmd}`` — one inference FSM per entry in ``omni_infer_yaml_path``

Usage
-----
  # Default: raw Mermaid (.mmd) → graphs/veomni_janus/
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b/veomni_janus.yaml

  # Browser-renderable HTML instead
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b/veomni_janus.yaml \\
      --visualize.format html

  # Inspect a single training YAML directly (override the graph pointers on CLI)
  python scripts/visualize_omni_graph.py \\
      --model.omni_train_yaml_path configs/seed_omni/janus_1.3b/train.yaml \\
      --model.omni_infer_yaml_path '{}'
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Literal

import yaml

from veomni.arguments import parse_args
from veomni.arguments.arguments_types import DataArguments
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.generation_graph import GenerationGraph
from veomni.models.seed_omni.training_graph import TrainingGraph
from veomni.trainer.omni_inferencer import OmniInferModelArguments
from veomni.trainer.omni_trainer import OmniModelArguments, VeOmniOmniArguments


OutputFormat = Literal["html", "mmd"]

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; color: #222; }}
    h1   {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
    .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .meta code {{ background: #f3f3f7; padding: 0 0.3em; border-radius: 3px; }}
    .mermaid {{ background: #fafafa; padding: 1rem; border-radius: 6px; }}
  </style>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
  </script>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">{meta}</div>
  <pre class="mermaid">
{body}
  </pre>
</body>
</html>
"""

_MASTER_KEYS = ("nodes", "edges", "training_graph")


# ── Argument dataclasses ─────────────────────────────────────────────────────


@dataclass
class VisualizeArguments:
    """``visualize.*`` — per-invocation knobs for the graph visualizer."""

    format: Literal["html", "mmd"] = field(
        default="mmd",
        metadata={"help": "Output format: raw Mermaid (.mmd, default) or browser HTML (.html)."},
    )


@dataclass
class Arguments:
    """Root config for ``visualize_omni_graph`` — consumed by :func:`parse_args`."""

    model: OmniInferModelArguments = field(default_factory=OmniInferModelArguments)
    visualize: VisualizeArguments = field(default_factory=VisualizeArguments)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _yaml_stem(yaml_path: str) -> str:
    return os.path.splitext(os.path.basename(yaml_path))[0]


def _safe_load_yaml(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        sys.exit(f"failed to parse YAML `{path}`: {e}")
    except OSError as e:
        sys.exit(f"failed to read YAML `{path}`: {e}")
    return data if isinstance(data, dict) else {}


def _validate_train_yaml(train_yaml: str) -> None:
    data = _safe_load_yaml(train_yaml)
    if not any(k in data for k in _MASTER_KEYS):
        sys.exit(
            f"`{train_yaml}` must declare at least one of `nodes` / `edges` / `training_graph` (master training YAML)."
        )


def _validate_infer_yaml(infer_yaml: str) -> None:
    data = _safe_load_yaml(infer_yaml)
    if "generation_graph" not in data:
        sys.exit(f"`{infer_yaml}` has no `generation_graph` section (inference YAML).")


def _write_diagram(
    path: str,
    *,
    fmt: OutputFormat,
    title: str,
    body: str,
    meta: str,
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if fmt == "html":
        content = _HTML_TEMPLATE.format(title=title, meta=meta, body=body)
    else:
        content = body + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _render_training(cfg: OmniConfig, *, title: str) -> tuple[str, str]:
    graph = TrainingGraph(
        nodes=cfg.nodes,
        edges=cfg.edges,
        training_edges=cfg.training_edges,
    )
    body = graph.to_mermaid(title=title)
    meta = (
        f"<div>execution_order: <code>{', '.join(graph.execution_order)}</code></div>"
        f"<div>sources: <code>{', '.join(graph.sources)}</code></div>"
        f"<div>sinks: <code>{', '.join(graph.sinks)}</code></div>"
    )
    return body, meta


def _render_fsm(cfg: OmniConfig, *, title: str) -> tuple[str, str]:
    fsm = GenerationGraph(fsm_config=cfg.generation_graph, nodes=cfg.nodes, edges=cfg.edges)
    body = fsm.to_mermaid(title=title)
    meta = (
        f"<div>fsm_initial: <code>{fsm.initial_state}</code></div>"
        f"<div>fsm_states: <code>{', '.join(fsm.state_names)}</code></div>"
    )
    return body, meta


def _output_dir(launcher_yaml: str | None, fallback_train_yaml: str) -> str:
    """Pick the output directory for the diagrams.

    Prefer ``graphs/<launcher_yaml_stem>/`` (matches the V1 convention so
    existing docs / references keep working); fall back to ``graphs/<train_yaml_stem>/``
    when the visualizer was invoked without a launcher YAML (e.g. with
    explicit ``--model.omni_train_yaml_path`` overrides).
    """
    stem = _yaml_stem(launcher_yaml) if launcher_yaml else _yaml_stem(fallback_train_yaml)
    return os.path.join("graphs", stem)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args, launcher_yaml = parse_args(Arguments, return_config_path=True)
    fmt: OutputFormat = args.visualize.format

    train_yaml = args.model.omni_train_yaml_path or ""
    infer_map: dict[str, str] = dict(args.model.omni_infer_yaml_path or {})

    if not train_yaml:
        sys.exit(
            "`model.omni_train_yaml_path` is missing (set in launcher YAML or pass via --model.omni_train_yaml_path)."
        )
    if not os.path.isfile(train_yaml):
        sys.exit(f"training YAML not found: {train_yaml}")
    _validate_train_yaml(train_yaml)
    for infer_key, infer_path in infer_map.items():
        if not os.path.isfile(infer_path):
            sys.exit(f"inference YAML not found for {infer_key!r}: {infer_path}")
        _validate_infer_yaml(infer_path)

    out_dir = _output_dir(launcher_yaml, train_yaml)
    launcher_label = _yaml_stem(launcher_yaml) if launcher_yaml else _yaml_stem(train_yaml)
    ext = ".html" if fmt == "html" else ".mmd"
    model_root = args.model.model_path or ""
    model_args = OmniModelArguments(
        model_path=model_root,
        config_path=model_root or ".",
        tokenizer_path=getattr(args.model, "tokenizer_path", None),
        omni_train_yaml_path=train_yaml,
    )
    global_args = VeOmniOmniArguments(
        model=model_args,
        data=DataArguments(train_path=""),
    )._to_base_args()

    # 1. Training graph (train YAML only). Use the same loader path as
    # trainer/inferencer so visualisation stays in sync with runtime config merge.
    cfg_train = OmniConfig._init(
        global_args=global_args,
        model_path=global_args.model.model_path,
        train_yaml_path=train_yaml,
    )
    train_title = f"{launcher_label} — training"
    train_body, train_meta = _render_training(cfg_train, title=train_title)
    train_path = os.path.join(out_dir, "training" + ext)
    _write_diagram(train_path, fmt=fmt, title=train_title, body=train_body, meta=train_meta)
    print(f"wrote {train_path}", file=sys.stderr)

    # 2. One FSM per inference scenario.
    for infer_key, infer_path in sorted(infer_map.items()):
        cfg = OmniConfig._init(
            global_args=global_args,
            model_path=global_args.model.model_path,
            train_yaml_path=train_yaml,
            infer_yaml_path=infer_path,
        )
        fsm_title = f"{launcher_label} — {infer_key}"
        fsm_body, fsm_meta = _render_fsm(cfg, title=fsm_title)
        fsm_path = os.path.join(out_dir, infer_key + ext)
        _write_diagram(fsm_path, fmt=fmt, title=fsm_title, body=fsm_body, meta=fsm_meta)
        print(f"wrote {fsm_path}", file=sys.stderr)

    print(f"\nDone — {1 + len(infer_map)} {fmt} diagrams under {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
