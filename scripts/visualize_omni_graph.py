"""
Visualize an OmniModel config as Mermaid diagrams.

The script is fixed to a **two-YAML contract**: always pass the master
training YAML first, then the inference YAML.  This mirrors how SeedOmni
V2 configs are organised:

* The master ``train_*.yaml`` declares ``modules`` / ``nodes`` / ``edges``
  / ``training_graph``.  It is the single source of truth for the node
  and edge pools.
* Each ``infer_*.yaml`` is a delta that carries only a ``generation_graph``
  referencing those names; ``OmniConfig.from_yamls`` deep-merges the
  inference YAML on top of the training YAML.

Two diagrams are produced from the merged config:

* **Training graph**: ``TrainingGraph.to_mermaid()`` — a left-to-right
  flowchart over the active call-site nodes and edges.  Leaf nodes
  flowing to the virtual ``end`` keyword render as a dashed terminal.
* **Inference FSM**: ``GenerationGraph.to_mermaid()`` — a ``flowchart LR``
  where each non-``done`` state is a labelled subgraph carrying the
  body's mini-flow (real data edges); a dashed self-loop on each
  subgraph encodes the iteration count (``×N`` for ``fixed:N``, no
  label for ``variable``).  State transitions are thick ``==>`` arrows
  carrying the firing condition.  The ``done`` state itself is not
  drawn — every transition that targets it lands on a small terminal
  node instead.

Usage
-----
  # Both diagrams (default).
  python scripts/visualize_omni_graph.py \\
      configs/seed_omni/janus_1.3b/train_joint.yaml \\
      configs/seed_omni/janus_1.3b/infer_interleave.yaml

  # Render a standalone HTML file (open in any browser) — both diagrams stacked.
  python scripts/visualize_omni_graph.py train.yaml infer.yaml -o /tmp/janus.html

  # Save to .md (two fenced blocks) or .mmd (single mermaid; also writes a
  # sibling .fsm.mmd alongside the training graph).
  python scripts/visualize_omni_graph.py train.yaml infer.yaml -o graph.mmd
  python scripts/visualize_omni_graph.py train.yaml infer.yaml -o graph.md

  # Compact view without the dashed raw_batch pseudo-node (training graph
  # only — FSM diagram is unaffected).  The single-loss protocol means
  # modules collect their own _loss, so no `losses` pseudo-node is drawn
  # in either view.
  python scripts/visualize_omni_graph.py train.yaml infer.yaml --no-io

  # Restrict to one diagram.
  python scripts/visualize_omni_graph.py train.yaml infer.yaml --only train
  python scripts/visualize_omni_graph.py train.yaml infer.yaml --only fsm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


_MASTER_KEYS = ("nodes", "edges", "training_graph")


def _safe_load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        sys.exit(f"failed to parse YAML `{path}`: {e}")
    return data if isinstance(data, dict) else {}


def _validate_yaml_roles(train_yaml: Path, infer_yaml: Path) -> None:
    """Enforce the two-YAML contract: first arg is master, second is FSM delta.

    Each file is checked once, on its own, against a minimal schema:

    * ``train_yaml`` MUST declare at least one of the master pool keys
      (``nodes`` / ``edges`` / ``training_graph``).  If it doesn't, the
      caller almost certainly swapped the two paths — exit with an error
      that suggests the fix instead of letting ``OmniConfig.from_yamls``
      raise something cryptic about an empty ``training_edges``.
    * ``infer_yaml`` MUST declare ``generation_graph``.  If it carries
      master keys too, that's allowed (a self-contained config) but warned
      about — usually the sign of a stray edit that drifted out of the
      delta-YAML pattern.
    """
    train_data = _safe_load_yaml(train_yaml)
    infer_data = _safe_load_yaml(infer_yaml)

    train_has_master = any(k in train_data for k in _MASTER_KEYS)
    infer_has_master = any(k in infer_data for k in _MASTER_KEYS)
    train_has_fsm = "generation_graph" in train_data
    infer_has_fsm = "generation_graph" in infer_data

    # Most common mistake: caller swapped train_yaml ↔ infer_yaml.
    if not train_has_master and infer_has_master:
        sys.exit(
            f"YAML role mismatch: the FIRST argument should be the master training "
            f"YAML (with `nodes` / `edges` / `training_graph`), but `{train_yaml.name}` "
            f"has none of those — and `{infer_yaml.name}` does.\n"
            f"Did you swap the two paths?  Try:\n"
            f"    python scripts/visualize_omni_graph.py {infer_yaml} {train_yaml}"
        )

    if not train_has_master:
        sys.exit(
            f"`{train_yaml}` (passed as the master training YAML) declares none of "
            f"`nodes` / `edges` / `training_graph`.  Pass a complete training YAML."
        )

    if not infer_has_fsm:
        sys.exit(
            f"`{infer_yaml}` (passed as the inference YAML) has no `generation_graph` "
            f"section.  Pass an inference YAML such as `infer_*.yaml`."
        )

    if infer_has_master:
        # Soft warning: usually the second file is a pure delta.  Carrying
        # master keys is allowed (deep-merge will overwrite the master) but
        # is surprising enough to flag.
        print(
            f"warning: `{infer_yaml.name}` carries master pool keys "
            f"({', '.join(k for k in _MASTER_KEYS if k in infer_data)}); these "
            f"will deep-merge over `{train_yaml.name}`.",
            file=sys.stderr,
        )

    if train_has_fsm:
        print(
            f"warning: `{train_yaml.name}` already carries `generation_graph`; "
            f"`{infer_yaml.name}` will deep-merge over it.",
            file=sys.stderr,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "train_yaml",
        type=Path,
        help=(
            "Master training YAML — declares `modules` / `nodes` / `edges` / "
            "`training_graph`.  E.g. configs/seed_omni/janus_1.3b/train_joint.yaml"
        ),
    )
    parser.add_argument(
        "infer_yaml",
        type=Path,
        help=(
            "Inference YAML — carries a `generation_graph` referencing nodes / "
            "edges from the master.  Deep-merged on top of the training YAML.  "
            "E.g. configs/seed_omni/janus_1.3b/infer_interleave.yaml"
        ),
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
        help="Compact view: drop the dashed raw_batch pseudo-node (training graph only).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Diagram title (default: '<train_stem>+<infer_stem>').",
    )
    parser.add_argument(
        "--only",
        choices=("train", "fsm", "both"),
        default="both",
        help="Restrict output to one diagram. Default: both.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    train_yaml: Path = args.train_yaml
    infer_yaml: Path = args.infer_yaml

    for p in (train_yaml, infer_yaml):
        if not p.exists():
            sys.exit(f"config not found: {p}")

    _validate_yaml_roles(train_yaml, infer_yaml)

    cfg = OmniConfig.from_yamls(train_yaml, infer_yaml)
    graph = TrainingGraph(
        nodes=cfg.nodes,
        edges=cfg.edges,
        training_edges=cfg.training_edges,
    )

    title = args.title or f"{train_yaml.stem}+{infer_yaml.stem}"

    diagrams: list[tuple[str, str]] = []  # (heading, mermaid_body)

    if args.only in ("train", "both"):
        diagrams.append(
            (
                "Training graph",
                graph.to_mermaid(show_io=not args.no_io, title=f"{title} — training"),
            )
        )

    # The two-YAML contract guarantees a `generation_graph` is present (we
    # validated `infer_yaml` above), so this branch is always taken when
    # the user asks for the FSM.
    fsm = GenerationGraph(fsm_config=cfg.generation_graph, nodes=cfg.nodes, edges=cfg.edges)
    if args.only in ("fsm", "both"):
        diagrams.append(("Inference FSM", fsm.to_mermaid(title=f"{title} — inference FSM")))

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
