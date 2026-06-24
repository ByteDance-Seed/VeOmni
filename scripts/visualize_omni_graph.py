"""
Visualize OmniModel graphs from a VeOmni omni launcher YAML.

Contract
--------
Pass a single launcher ``base.yaml`` (e.g.
``configs/seed_omni/Janus/janus_1.3b/base.yaml``) as the positional
``config_file`` argument.  Its graph-pointing fields are consumed:

* ``model.modules`` / ``model.train_graph`` — training vocabulary + DAG
* ``infer.infer_graph`` — dict of scenario → generation-graph YAML

and writes diagrams to ``graphs/<model_dir>_<stem>/`` (the launcher YAML's
parent directory name + stem, e.g.
``configs/seed_omni/Janus/janus_1.3b/base.yaml`` → ``graphs/janus_1.3b_base/``;
the parent prefix disambiguates the per-model ``base.yaml`` launchers):

1. ``training.{html|mmd}`` — training DAG from ``training_graph``
2. ``<infer_key>.{html|mmd}`` — one inference FSM per entry in ``infer.infer_graph``

Usage
-----
  # Default: raw Mermaid (.mmd) → graphs/janus_1.3b_base/
  python scripts/visualize_omni_graph.py configs/seed_omni/Janus/janus_1.3b/base.yaml

  # Browser-renderable HTML instead
  python scripts/visualize_omni_graph.py configs/seed_omni/Janus/janus_1.3b/base.yaml \\
      --visualize.format html
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Literal

from veomni.arguments import OmniArguments, parse_omni_args
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph
from veomni.models.seed_omni.graphs.training_graph import TrainingGraph


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


@dataclass
class VisualizeArguments:
    """``visualize.*`` — per-invocation knobs for the graph visualizer."""

    format: Literal["html", "mmd"] = field(
        default="mmd",
        metadata={"help": "Output format: raw Mermaid (.mmd, default) or browser HTML (.html)."},
    )


@dataclass
class Arguments(OmniArguments):
    """Root config for ``visualize_omni_graph`` — extends the omni launcher schema."""

    visualize: VisualizeArguments = field(default_factory=VisualizeArguments)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _yaml_stem(yaml_path: str) -> str:
    return os.path.splitext(os.path.basename(yaml_path))[0]


def _yaml_label(yaml_path: str) -> str:
    """``<parent_dir>_<stem>`` — disambiguates per-model launchers.

    Every model's launcher is named ``base.yaml``, so the stem alone collides
    (all land in ``graphs/base/``).  Prefix the parent directory name (the
    model dir, e.g. ``janus_1.3b``) → ``janus_1.3b_base``.
    """
    stem = _yaml_stem(yaml_path)
    parent = os.path.basename(os.path.dirname(os.path.abspath(yaml_path)))
    return f"{parent}_{stem}" if parent else stem


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
    graph = TrainingGraph(edges=cfg.training_graph)
    body = graph.to_mermaid(title=title)
    meta = (
        f"<div>execution_order: <code>{', '.join(graph.execution_order)}</code></div>"
        f"<div>sources: <code>{', '.join(graph.sources)}</code></div>"
        f"<div>sinks: <code>{', '.join(graph.sinks)}</code></div>"
    )
    return body, meta


def _render_fsm(cfg: OmniConfig, *, title: str) -> tuple[str, str]:
    fsm = GenerationGraph(fsm_config=cfg.generation_graph)
    body = fsm.to_mermaid(title=title)
    meta = (
        f"<div>fsm_initial: <code>{fsm.initial_state}</code></div>"
        f"<div>fsm_states: <code>{', '.join(fsm.state_names)}</code></div>"
    )
    return body, meta


def _output_dir(launcher_yaml: str | None, fallback: str) -> str:
    """Pick the output directory: ``graphs/<parent_dir>_<launcher_stem>/``."""
    label = _yaml_label(launcher_yaml) if launcher_yaml else _yaml_label(fallback)
    return os.path.join("graphs", label)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args, launcher_yaml = parse_omni_args(
        Arguments,
        preload_path_fields=("model.modules", "infer.modules"),
        return_config_path=True,
    )
    fmt: OutputFormat = args.visualize.format

    if not args.model.modules:
        sys.exit("`model.modules` is missing (set in launcher YAML or pass via --model.modules).")
    if not args.model.train_graph:
        sys.exit("`model.train_graph` is missing (set in launcher YAML or pass via --model.train_graph).")
    infer_map: dict[str, str] = dict(args.infer.infer_graph or {})

    out_dir = _output_dir(launcher_yaml, args.model.train_graph)
    launcher_label = _yaml_label(launcher_yaml) if launcher_yaml else _yaml_label(args.model.train_graph)
    ext = ".html" if fmt == "html" else ".mmd"

    # 1. Training graph. Use the same loader path as trainer/inferencer so
    # visualisation stays in sync with the runtime config merge.
    cfg_train = args.load_omni_config()
    train_title = f"{launcher_label} — training"
    train_body, train_meta = _render_training(cfg_train, title=train_title)
    train_path = os.path.join(out_dir, "training" + ext)
    _write_diagram(train_path, fmt=fmt, title=train_title, body=train_body, meta=train_meta)
    print(f"wrote {train_path}", file=sys.stderr)

    # 2. One FSM per inference scenario.
    for infer_key in sorted(infer_map):
        args.infer.infer_type = infer_key
        cfg = args.load_omni_infer_config()
        fsm_title = f"{launcher_label} — {infer_key}"
        fsm_body, fsm_meta = _render_fsm(cfg, title=fsm_title)
        fsm_path = os.path.join(out_dir, infer_key + ext)
        _write_diagram(fsm_path, fmt=fmt, title=fsm_title, body=fsm_body, meta=fsm_meta)
        print(f"wrote {fsm_path}", file=sys.stderr)

    print(f"\nDone — {1 + len(infer_map)} {fmt} diagrams under {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
