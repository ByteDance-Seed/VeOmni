"""Verify that every ``#L`` anchor in the SeedOmni docs lands on the symbol it names.

A doc link looks like ``[text](../../../veomni/path/file.py#L123)``. Line anchors
into files this branch edits drift silently — nothing in the docs build notices
when a `#L137` starts pointing at a comment. This script resolves the *link text*
to a definition and compares it to the anchored line.

Only anchors into `WATCHED` files are checked: those are the ones this branch
changes, and elsewhere the drift is someone else's.

Run: python scripts/ci/docs/check_line_anchors.py [--fix]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
LINK = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<path>(?:\.\./)+)(?P<target>[\w./-]+\.py)#L(?P<line>\d+)\)")

WATCHED = {
    "veomni/data/seed_omni/collator.py",
    "veomni/data/dataset.py",
    "veomni/utils/helper.py",
    "veomni/utils/omni_helper.py",
    "veomni/models/seed_omni/modules/qwen3vl/hf_layout.py",
    "veomni/models/seed_omni/modules/qwen3vl/convert_model.py",
    "veomni/models/seed_omni/modules/qwen3vl/packing.py",
}

# Link-text fragment -> the line that must hold it. Order matters: the first
# matching fragment wins, so put the specific ones before the general ones.
RULES: list[tuple[str, str]] = [
    (r"OmniEnvironMeter\.state_dict", r"^    def state_dict\("),
    (r"OmniEnvironMeter\.add", r"^    def add\(self, micro_batch"),
    (r"OmniEnvironMeter\.step", r"^    def step\("),
    (r"OmniEnvironMeter\._multisource_module", r"^    def _multisource_module\("),
    (r"class OmniEnvironMeter", r"^class OmniEnvironMeter\b"),
    (r"OmniModelRuntime\.metric_meter_collect", r"^    def metric_meter_collect\("),
    (r"OmniModel\.forward", r"^    def forward\(\s*$"),
    (r"_run_train_node", r"^    def _run_train_node\("),
    (r"OmniProcessor\.preprocess_batch", r"^    def preprocess_batch\("),
    (r"OmniProcessor\.preprocess`?$", r"^    def preprocess\("),
    (r"OmniProcessor`?$", r"^class OmniProcessor\b"),
    (r"OmniProcessor\.from_config", r"^    def from_config\("),
    (r"OmniProcessor\.bind_dummy_inputs", r"^    def bind_dummy_inputs\("),
    (r"SeedOmniCollator\.__call__", r"^    def __call__\(self, features"),
    (r"SeedOmniCollator`?$", r"^class SeedOmniCollator\b"),
    (r"get_length_by_labels_fn", r"^def get_length_by_labels_fn"),
    (r"get_length_fn`?$", r"^def get_length_fn"),
    (r"class EnvironMeter", r"^class EnvironMeter\b"),
    (r"compute_device_memory_metrics", r"^def compute_device_memory_metrics"),
    (r"One DP all-reduce", r"reduced = all_reduce\("),
    (r"flops_achieved\(T\)", r"flops_achieved = total_flops"),
    (r"flops_promised\(T\)", r"flops_promised = get_device_flops"),
    (r"^`?mfu`?$", r"mfu = flops_achieved"),
    (r"consumed_chunk_num", r'\{"consumed_chunk_num"'),
    (r"omitted", r"^        if names:$"),
    (r"tokens_per_second", r'"tokens_per_second\(M\)"'),
    (r"consume_tokens", r'"consume_tokens\(M\)"'),
    (r"avg_seq_len", r'"avg_seq_len"'),
    (r"logged as zero", r"per_sample_seqlens = \[0\]"),
]


def resolve(target: Path, text: str) -> int | None:
    """The line in ``target`` that holds the symbol ``text`` names, if known."""
    lines = target.read_text(encoding="utf-8").splitlines()
    for fragment, pattern in RULES:
        if re.search(fragment, text):
            for idx, line in enumerate(lines, start=1):
                if re.search(pattern, line):
                    return idx
            return None
    return None


def main(argv: list[str]) -> int:
    fix = "--fix" in argv
    stale = 0
    for doc in sorted((REPO / "docs").rglob("*.md")):
        source = doc.read_text(encoding="utf-8")
        pieces: list[str] = []
        cursor = 0
        for match in LINK.finditer(source):
            target = (doc.parent / (match["path"] + match["target"])).resolve()
            try:
                rel = target.relative_to(REPO).as_posix()
            except ValueError:
                continue
            if rel not in WATCHED:
                continue
            expected = resolve(target, match["text"])
            if expected is None or expected == int(match["line"]):
                continue
            stale += 1
            print(f"STALE {doc.relative_to(REPO)}  {match['text'][:40]}  L{match['line']} -> L{expected}")
            if fix:
                pieces.append(source[cursor : match.start()])
                pieces.append(f"[{match['text']}]({match['path']}{match['target']}#L{expected})")
                cursor = match.end()
        if fix and pieces:
            pieces.append(source[cursor:])
            doc.write_text("".join(pieces), encoding="utf-8")

    if stale:
        print(f"\n{stale} stale anchor(s)" + (" — fixed" if fix else "; rerun with --fix"))
    else:
        print("All doc line anchors resolve to the symbol they name.")
    return 0 if fix or not stale else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
