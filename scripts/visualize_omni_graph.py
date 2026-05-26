"""
Visualize OmniModel graphs from a VeOmni launcher YAML.

Contract
--------
Pass a single launcher YAML (e.g. ``configs/seed_omni/janus_1.3b/veomni_janus.yaml``).
The script reads:

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
  python scripts/visualize_omni_graph.py configs/seed_omni/janus_1.3b/veomni_janus.yaml --format html
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Literal

import yaml

from veomni.models.seed_omni.configuration_seed_omni import OmniConfig, load_launcher_model_section
from veomni.models.seed_omni.generation_graph import GenerationGraph
from veomni.models.seed_omni.training_graph import TrainingGraph


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


def _output_dir(launcher_yaml: str) -> str:
    """``graphs/<yaml_stem>/`` — folder name matches the launcher YAML basename."""
    return os.path.join("graphs", _yaml_stem(launcher_yaml))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "launcher_yaml",
        help=(
            "VeOmni launcher YAML (e.g. configs/seed_omni/janus_1.3b/veomni_janus.yaml).  "
            "Must declare model.omni_train_yaml_path and model.omni_infer_yaml_path."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("html", "mmd"),
        default="mmd",
        help="Output format: raw Mermaid (.mmd, default) or browser HTML (.html).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    launcher_yaml: str = args.launcher_yaml
    fmt: OutputFormat = args.format
    if not os.path.isfile(launcher_yaml):
        sys.exit(f"config not found: {launcher_yaml}")

    model = load_launcher_model_section(launcher_yaml)
    train_yaml = model.get("omni_train_yaml_path") or ""
    infer_map: dict[str, str] = dict(model.get("omni_infer_yaml_path") or {})
    if not train_yaml:
        sys.exit(f"`model.omni_train_yaml_path` is missing in {launcher_yaml}")
    if not os.path.isfile(train_yaml):
        sys.exit(f"training YAML not found: {train_yaml}")
    if not infer_map:
        sys.exit(f"`model.omni_infer_yaml_path` is empty in {launcher_yaml}")

    _validate_train_yaml(train_yaml)
    for infer_key, infer_path in infer_map.items():
        if not os.path.isfile(infer_path):
            sys.exit(f"inference YAML not found for {infer_key!r}: {infer_path}")
        _validate_infer_yaml(infer_path)

    out_dir = _output_dir(launcher_yaml)
    launcher_label = _yaml_stem(launcher_yaml)
    ext = ".html" if fmt == "html" else ".mmd"

    # 1. Training graph (train YAML only).
    cfg_train = OmniConfig.from_yamls(train_yaml)
    train_title = f"{launcher_label} — training"
    train_body, train_meta = _render_training(cfg_train, title=train_title)
    train_path = os.path.join(out_dir, "training" + ext)
    _write_diagram(train_path, fmt=fmt, title=train_title, body=train_body, meta=train_meta)
    print(f"wrote {train_path}", file=sys.stderr)

    # 2. One FSM per inference scenario.
    for infer_key, infer_path in sorted(infer_map.items()):
        cfg = OmniConfig.from_yamls(train_yaml, infer_path)
        fsm_title = f"{launcher_label} — {infer_key}"
        fsm_body, fsm_meta = _render_fsm(cfg, title=fsm_title)
        fsm_path = os.path.join(out_dir, infer_key + ext)
        _write_diagram(fsm_path, fmt=fmt, title=fsm_title, body=fsm_body, meta=fsm_meta)
        print(f"wrote {fsm_path}", file=sys.stderr)

    print(f"\nDone — {1 + len(infer_map)} {fmt} diagrams under {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
