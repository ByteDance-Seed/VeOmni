"""
Visualize OmniModel graphs from a VeOmni omni launcher YAML.

Contract
--------
Pass a single launcher ``base.yaml`` (e.g.
``configs/seed_omni/Janus/janus_1.3b/base.yaml``) as the positional
``config_file`` argument.  Its graph-pointing fields are consumed:

* ``model.model_config.modules`` / ``model.model_config.train_graph`` — training vocabulary + DAG
* ``model.model_config.infer_graph`` — dict of scenario → generation-graph YAML

and writes diagrams to ``graphs/<model_dir>_<stem>/`` (the launcher YAML's
parent directory name + stem, e.g.
``configs/seed_omni/Janus/janus_1.3b/base.yaml`` → ``graphs/janus_1.3b_base/``;
the parent prefix disambiguates the per-model ``base.yaml`` launchers):

1. ``training.{html|mmd}`` — training DAG from ``training_graph``
2. ``<infer_key>.{html|mmd}`` — one inference FSM per entry in ``model.model_config.infer_graph``

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
from veomni.models.seed_omni.utils.visualize import (
    render_generation_mermaid,
    render_training_mermaid,
    write_mermaid_file,
)


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


def _yaml_stem(yaml_path: str) -> str:
    return os.path.splitext(os.path.basename(yaml_path))[0]


def _yaml_label(yaml_path: str) -> str:
    """``<parent_dir>_<stem>`` — disambiguates per-model launchers."""
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
    if fmt == "html":
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        content = _HTML_TEMPLATE.format(title=title, meta=meta, body=body)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        write_mermaid_file(path, body)


def _training_meta(cfg_train) -> str:
    from veomni.models.seed_omni.graphs.training_graph import TrainingGraph

    graph = TrainingGraph(cfg_train.training_graph)
    return (
        f"<div>execution_order: <code>{', '.join(graph.execution_order)}</code></div>"
        f"<div>sources: <code>{', '.join(graph.sources)}</code></div>"
        f"<div>sinks: <code>{', '.join(graph.sinks)}</code></div>"
    )


def _generation_meta(cfg, infer_key: str) -> str:
    from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph

    fsm = GenerationGraph(cfg.generation_graphs[infer_key])
    return (
        f"<div>fsm_initial: <code>{fsm.initial_state}</code></div>"
        f"<div>fsm_states: <code>{', '.join(fsm.state_names)}</code></div>"
    )


def _output_dir(launcher_yaml: str | None, fallback: str) -> str:
    label = _yaml_label(launcher_yaml) if launcher_yaml else _yaml_label(fallback)
    return os.path.join("graphs", label)


def main() -> None:
    args, launcher_yaml = parse_omni_args(
        Arguments,
        preload_path_fields=("model.model_config.modules",),
        return_config_path=True,
    )
    fmt: OutputFormat = args.visualize.format

    if not args.model.launcher_config("modules"):
        sys.exit(
            "`model.model_config.modules` is missing (set in launcher YAML or pass via "
            "`--model.model_config.modules`)."
        )
    if not args.model.launcher_config("train_graph"):
        sys.exit(
            "`model.model_config.train_graph` is missing (set in launcher YAML or pass via "
            "`--model.model_config.train_graph`)."
        )

    train_graph = args.model.launcher_config("train_graph")
    out_dir = _output_dir(launcher_yaml, train_graph)
    launcher_label = _yaml_label(launcher_yaml) if launcher_yaml else _yaml_label(train_graph)
    ext = ".html" if fmt == "html" else ".mmd"

    # One config carries every scenario, so the FSMs below need no rebuild per key.
    # Diagrams only read graphs, so the runtime config serves directly — no need to
    # project onto an OmniConfig.
    cfg_train = args.resolve_model()
    train_title = f"{launcher_label} — training"
    train_body = render_training_mermaid(cfg_train, title=train_title)
    train_path = os.path.join(out_dir, "training" + ext)
    _write_diagram(
        train_path,
        fmt=fmt,
        title=train_title,
        body=train_body,
        meta=_training_meta(cfg_train),
    )
    print(f"wrote {train_path}", file=sys.stderr)

    infer_keys = sorted(cfg_train.infer_types)
    for infer_key in infer_keys:
        fsm_title = f"{launcher_label} — {infer_key}"
        fsm_body = render_generation_mermaid(cfg_train, title=fsm_title, infer_type=infer_key)
        fsm_path = os.path.join(out_dir, infer_key + ext)
        _write_diagram(
            fsm_path,
            fmt=fmt,
            title=fsm_title,
            body=fsm_body,
            meta=_generation_meta(cfg_train, infer_key),
        )
        print(f"wrote {fsm_path}", file=sys.stderr)

    print(f"\nDone — {1 + len(infer_keys)} {fmt} diagrams under {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
