"""Verify that every ``#L`` anchor in the docs lands on the code it names.

A doc link looks like ``[text](../../../veomni/path/file.py#L123)``. Line anchors
drift silently: nothing in the docs build notices when a `#L137` starts pointing
at a comment.

Scope is derived, not hardcoded: an anchor is checked when its target is a Python
file this branch changes (against the merge base), because those are the anchors
whose line numbers moved underneath them. A hand-maintained list would rot exactly
the way the anchors do.

Resolution is a table because link text is prose. ``[`OmniEnvironMeter`]`` names a
class declaration; ``[`OmniProcessor.preprocess_batch`]`` names a method on one
class specifically; ``[omitted]`` names a branch. Each entry says what the
anchored line must contain. A link text the table does not know is a **failure**,
not a skip: a checker that cannot tell "correct" from "I don't know" must not
report success.

Run:
    python scripts/ci/docs/check_line_anchors.py           # check, exit 1 on stale
    python scripts/ci/docs/check_line_anchors.py --fix     # rewrite, then verify
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BASE = "github/main"
LINK = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<path>(?:\.\./)+)(?P<target>[\w./-]+\.py)#L(?P<line>\d+)\)")

# ``(link-text pattern, regex the anchored line must match)``. First match wins, so
# the specific entries come before the generic ones: `OmniProcessor.preprocess_batch`
# must not be captured by a bare `OmniProcessor` rule.
RULES: list[tuple[str, str]] = [
    # OmniEnvironMeter members (before the class itself).
    (r"OmniEnvironMeter\.state_dict", r"^\s*def state_dict\("),
    (r"OmniEnvironMeter\.add", r"^\s*def add\(self, micro_batch"),
    (r"OmniEnvironMeter\.step", r"^\s*def step\("),
    (r"OmniEnvironMeter\._multisource_module", r"^\s*def _multisource_module\("),
    (r"OmniModelRuntime\.metric_meter_collect", r"^\s*def metric_meter_collect\("),
    # OmniProcessor members (before the class). Longest first: `preprocess_batch`
    # and `preprocess(conversation_list)` both start with `preprocess`.
    (r"OmniProcessor\.preprocess_batch", r"^\s*def preprocess_batch\("),
    (r"OmniProcessor\.preprocess\(", r"^\s*def preprocess\("),
    (r"OmniProcessor\.preprocess`?$", r"^\s*def preprocess\("),
    (r"OmniProcessor\.from_config", r"^\s*def from_config\("),
    (r"OmniProcessor\.bind_dummy_inputs", r"^\s*def bind_dummy_inputs\("),
    (r"OmniProcessor\.__call__", r"^\s*def __call__\("),
    # SeedOmniCollator members (before the class).
    (r"SeedOmniCollator\.__call__", r"^\s*def __call__\(self, features"),
    # Class declarations, bare-name links.
    (r"^`?OmniEnvironMeter`?", r"^class OmniEnvironMeter\b"),
    (r"^`?EnvironMeter`?", r"^class EnvironMeter\b"),
    (r"^`?OmniProcessor`?", r"^class OmniProcessor\b"),
    (r"^`?SeedOmniCollator`?", r"^class SeedOmniCollator\b"),
    # Free functions.
    (r"get_length_by_labels_fn", r"^def get_length_by_labels_fn"),
    (r"get_length_fn`?$", r"^def get_length_fn"),
    (r"compute_device_memory_metrics", r"^def compute_device_memory_metrics"),
    (r"_run_train_node", r"^\s*def _run_train_node\("),
    (r"OmniModel\.forward", r"^\s*def forward\(\s*$"),
    # Statements and emitted metric keys, which the docs link to by name.
    (r"One DP all-reduce", r"reduced = all_reduce\("),
    (r"flops_achieved\(T\)", r"flops_achieved = total_flops"),
    (r"flops_promised\(T\)", r"flops_promised = get_device_flops"),
    (r"^`?mfu`?$", r"mfu = flops_achieved"),
    (r"consumed_chunk_num", r'\{"consumed_chunk_num"'),
    (r"omitted", r"^\s*if names:$"),
    (r"tokens_per_second", r'"tokens_per_second\(M\)"'),
    (r"consume_tokens", r'"consume_tokens\(M\)"'),
    (r"avg_seq_len", r'"avg_seq_len"'),
    (r"logged as zero", r"per_sample_seqlens = \[0\]"),
]

# Doc prose that links to a *test* by describing it: resolve via the test's name.
TEST_RULES: list[tuple[str, str]] = [
    (r"strict key alignment", r"^def test_collator_strict_key_alignment_rejects_mismatched_sample_keys"),
    (r"mixed-resolution images", r"^def test_collator_preserves_mixed_resolution_images_without_stacking"),
    (r"per-sample dummy anchor", r"^def test_module_preprocessor_appends_per_sample_dummy_when_modality_absent"),
    (r"`seedomni_transform` runs", r"^def test_seedomni_transform_runs_processor_and_supports_dyn_bsz_lengths"),
]


def changed_python_files() -> set[str]:
    """Repo-relative Python files this branch changes, versus its merge base."""
    result = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--name-only", f"{BASE}...HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"warning: `git diff {BASE}...HEAD` failed; checking every anchor", file=sys.stderr)
        return set()
    return {name for name in result.stdout.split() if name.endswith(".py")}


def pattern_for(text: str) -> str | None:
    for table in (RULES, TEST_RULES):
        for fragment, pattern in table:
            if re.search(fragment, text):
                return pattern
    return None


def find_line(lines: list[str], pattern: str) -> int | None:
    compiled = re.compile(pattern)
    for idx, line in enumerate(lines, start=1):
        if compiled.search(line):
            return idx
    return None


def check(fix: bool) -> int:
    scope = changed_python_files()
    stale = 0
    unchecked = 0
    checked = 0

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
            if scope and rel not in scope:
                continue
            if not target.is_file():
                print(f"MISSING   {doc.relative_to(REPO)}  {rel} does not exist")
                stale += 1
                continue

            pattern = pattern_for(match["text"])
            if pattern is None:
                # Not "fine" — unknown. Add a RULES entry saying what to look for.
                print(f"UNCHECKED {doc.relative_to(REPO)}  {match['text'][:52]!r} -> {rel}")
                unchecked += 1
                continue

            lines = target.read_text(encoding="utf-8").splitlines()
            claimed = int(match["line"])
            checked += 1
            if 0 < claimed <= len(lines) and re.search(pattern, lines[claimed - 1]):
                continue

            expected = find_line(lines, pattern)
            stale += 1
            print(
                f"STALE     {doc.relative_to(REPO)}  {match['text'][:46]:48s} "
                f"L{claimed} -> L{expected if expected is not None else '?'}"
            )
            if fix and expected is not None:
                pieces.append(source[cursor : match.start()])
                pieces.append(f"[{match['text']}]({match['path']}{match['target']}#L{expected})")
                cursor = match.end()

        if fix and pieces:
            pieces.append(source[cursor:])
            doc.write_text("".join(pieces), encoding="utf-8")

    state = f"{checked} checked"
    if stale:
        state += f", {stale} stale" + (" (rewritten)" if fix else "")
    if unchecked:
        state += f", {unchecked} unresolvable"
    print(state)

    if fix and stale and not unchecked:
        return check(fix=False)
    return 1 if stale or unchecked else 0


def main(argv: list[str]) -> int:
    return check(fix="--fix" in argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
