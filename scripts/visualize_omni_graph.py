"""
Visualize an OmniModel config as Mermaid diagrams.

Two diagrams are produced from a single OmniConfig:

* **Training graph**: ``TrainingGraph.to_mermaid()`` — a ``flowchart TD`` over
  active call-site nodes and edges.  Active nodes are derived from the
  endpoints of ``training_graph.edges``; leaf nodes flowing to the virtual
  ``end`` keyword render as a dashed terminal.
* **Inference FSM**: ``GenerationGraph.to_mermaid()`` — a
  ``stateDiagram-v2`` over generation states.  Each state's label lists the
  derived node-execution sequence (unique edge endpoints, excluding
  ``end``, in declaration order).  Skipped when ``generation_graph`` is
  absent from the config.

Usage
-----
  # Print Mermaid syntax to stdout (paste into https://mermaid.live to view).
  # Both diagrams are emitted, separated by a blank line.
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b.yaml

  # Render a standalone HTML file (open in any browser) — both diagrams stacked.
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b.yaml -o /tmp/janus.html

  # Save to .md (two fenced blocks) or .mmd (single mermaid; also writes a
  # sibling .fsm.mmd when an FSM is present).
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b.yaml -o graph.mmd
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b.yaml -o graph.md

  # Compact view without the dashed raw_batch / losses pseudo-nodes (training
  # graph only — FSM diagram is unaffected).
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b.yaml --no-io

  # Restrict to one diagram.
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b.yaml --only train
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b.yaml --only fsm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

from veomni.models.seed_omni.configuration_seed_omni import OmniConfig
from veomni.models.seed_omni.generation_graph import GenerationGraph
from veomni.models.seed_omni.training_graph import TrainingGraph


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; color: #222; }}
    h1   {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
    h2   {{ font-size: 1.1rem; margin-top: 2rem; margin-bottom: 0.4rem; color: #333; }}
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
  <div class="meta">
    <div>execution_order: <code>{order}</code></div>
    <div>sources: <code>{sources}</code></div>
    <div>sinks: <code>{sinks}</code></div>
    {fsm_meta}
  </div>
{sections}
</body>
</html>
"""

_HTML_SECTION = """  <h2>{heading}</h2>
  <pre class="mermaid">
{body}
  </pre>
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "config",
        type=Path,
        help="Path to OmniModel config YAML (with modules / nodes / edges / training_graph / generation_graph)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file: .html (browser-renderable), .mmd (raw mermaid), .md (fenced). Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--no-io",
        action="store_true",
        help="Compact view: drop the dashed raw_batch and losses pseudo-nodes (training graph only).",
    )
    parser.add_argument("--title", type=str, default=None, help="Diagram title (default: config file stem).")
    parser.add_argument(
        "--only",
        choices=("train", "fsm", "both"),
        default="both",
        help="Restrict output to one diagram. Default: both (training + FSM when present).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg_path: Path = args.config
    if not cfg_path.exists():
        sys.exit(f"config not found: {cfg_path}")

    cfg = OmniConfig.from_dict(yaml.safe_load(cfg_path.read_text()))
    graph = TrainingGraph(
        nodes=cfg.nodes,
        edges=cfg.edges,
        training_edges=cfg.training_edges,
    )

    title = args.title or cfg_path.stem

    diagrams: List[Tuple[str, str]] = []  # (heading, mermaid_body)

    if args.only in ("train", "both"):
        diagrams.append(
            (
                "Training graph",
                graph.to_mermaid(show_io=not args.no_io, title=f"{title} — training"),
            )
        )

    fsm = None
    if cfg.has_generation_graph():
        fsm = GenerationGraph(fsm_config=cfg.generation_graph, nodes=cfg.nodes, edges=cfg.edges)
        if args.only in ("fsm", "both"):
            diagrams.append(("Inference FSM", fsm.to_mermaid(title=f"{title} — inference FSM")))
    elif args.only == "fsm":
        sys.exit("--only fsm requested but config has no `generation_graph` section.")

    if not diagrams:
        sys.exit("Nothing to render: --only train requested with no training_graph (impossible) — bug?")

    fsm_meta_text = ""
    if fsm is not None:
        fsm_meta_text = (
            f"<div>fsm_initial: <code>{fsm.initial_state}</code></div>"
            f"<div>fsm_states: <code>{', '.join(fsm.state_names)}</code></div>"
        )

    out = args.output

    if out is None:
        # stdout: emit each diagram, separated by a blank line; stats to stderr.
        print(f"# {title}", file=sys.stderr)
        print(f"  execution_order : {graph.execution_order}", file=sys.stderr)
        print(f"  sources         : {graph.sources}", file=sys.stderr)
        print(f"  sinks           : {graph.sinks}", file=sys.stderr)
        if fsm is not None:
            print(f"  fsm_initial     : {fsm.initial_state}", file=sys.stderr)
            print(f"  fsm_states      : {fsm.state_names}", file=sys.stderr)
        print("", file=sys.stderr)
        for i, (heading, body) in enumerate(diagrams):
            if i > 0:
                print("")
            print(f"# {heading}")
            print(body)
        return

    suffix = out.suffix.lower()
    if suffix == ".html":
        sections = "\n".join(_HTML_SECTION.format(heading=h, body=b) for h, b in diagrams)
        out.write_text(
            _HTML_TEMPLATE.format(
                title=title,
                order=", ".join(graph.execution_order),
                sources=", ".join(graph.sources),
                sinks=", ".join(graph.sinks),
                fsm_meta=fsm_meta_text,
                sections=sections,
            )
        )
        print(f"wrote {out}", file=sys.stderr)
    elif suffix == ".md":
        chunks = [f"# {title}\n"]
        for heading, body in diagrams:
            chunks.append(f"## {heading}\n\n```mermaid\n{body}\n```\n")
        out.write_text("\n".join(chunks))
        print(f"wrote {out}", file=sys.stderr)
    elif suffix == ".mmd":
        # .mmd carries a single mermaid diagram. Write training graph here;
        # write the FSM (if present and requested) to a sibling `.fsm.mmd`.
        train_body = next((b for h, b in diagrams if h == "Training graph"), None)
        fsm_body = next((b for h, b in diagrams if h == "Inference FSM"), None)
        if train_body is not None:
            out.write_text(train_body + "\n")
            print(f"wrote {out}", file=sys.stderr)
        if fsm_body is not None:
            fsm_path = out.with_suffix(".fsm.mmd") if train_body is not None else out
            fsm_path.write_text(fsm_body + "\n")
            print(f"wrote {fsm_path}", file=sys.stderr)
    else:
        sys.exit(f"unknown output suffix {suffix!r}; expected .html, .mmd, or .md")


if __name__ == "__main__":
    main()
