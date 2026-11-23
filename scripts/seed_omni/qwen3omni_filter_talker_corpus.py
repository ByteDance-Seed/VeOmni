# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Score a talker corpus by transcribing its speech back, and drop the rows that lie.

The talker is taught to say the assistant text, so a row whose waveform says
something else is a mislabeled example — and the duration check in
``qwen3omni_build_talker_corpus.py`` cannot see it. A clip can sit squarely
inside the plausible seconds-per-character band and still be unintelligible:
observed at roughly one row in six on a self-generated corpus, which is enough
to matter for a graph that has nothing else to learn from.

This reads each clip back through the model's own audio tower and compares the
transcript to the text on the row, word by word. Using the same model to judge
its own speech sounds circular but is not: the tower and the talker are separate
stacks, the tower's transcription is reliable where the speech is intelligible
(0.92-1.00 word overlap on clean rows), and the failure this catches is exactly
"the audio does not carry these words".

Two passes, so a slow scoring run is never repeated:

1. ``--score`` transcribes and writes one JSONL of scores per worker beside the
   corpus. Shardable across GPUs the same way the generator is.
2. ``--apply`` rewrites the shards keeping rows at or above ``--min-overlap``.
   Single process, and it refuses to run until every shard has been scored.

``--apply`` edits the corpus in place and runs exactly once — it drops a marker
beside the directory and refuses a second pass, because rewriting renumbers the
rows the scores are keyed on. Copy the corpus first if the generating run was
expensive; a rerun of ``--score`` against the filtered copy is the supported way
back.

Usage::

    for w in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$w python \\
      scripts/seed_omni/qwen3omni_filter_talker_corpus.py --score \\
      configs/seed_omni/Qwen/qwen3omni_30b_a3b/train/base.yaml \\
      --model.model_path /opt/tiger/models/qwen3omni_split \\
      --model.model_config.modules configs/seed_omni/Qwen/qwen3omni_30b_a3b/infer/modules_infer_eager.yaml \\
      --model.model_config.infer_type thinker_only \\
      --model.ops_implementation.attn_implementation flash_attention_3 \\
      --corpus /mnt/hdfs/veomni/datasets/qwen3omni_talker_selfspeech \\
      --worker $w --num-workers 8 & done

    python scripts/seed_omni/qwen3omni_filter_talker_corpus.py --apply \\
      --corpus /mnt/hdfs/veomni/datasets/qwen3omni_talker_selfspeech --min-overlap 0.6
"""

from __future__ import annotations

import argparse
import difflib
import glob
import io
import json
import os
import re
import sys
import time

from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata


TRANSCRIBE_PROMPT = "Transcribe this audio verbatim. Output only the words spoken."


def parse_script_args() -> argparse.Namespace:
    """Take this script's own flags off ``argv`` before the launcher parses it."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--corpus", required=True, help="Corpus directory of parquet shards.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--score", action="store_true", help="Transcribe and write scores (needs a GPU).")
    mode.add_argument("--apply", action="store_true", help="Rewrite the shards, dropping low-scoring rows.")
    parser.add_argument("--worker", type=int, default=0, help="This worker's index, for --score.")
    parser.add_argument("--num-workers", type=int, default=1, help="How many workers share the shards.")
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.6,
        help="Keep a row whose transcript overlaps its text by at least this word ratio. Clean rows score "
        "0.92-1.00 and unintelligible ones near 0, so the threshold sits in an empty region — it is not a "
        "quality dial.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Score at most this many rows per worker (0 = all).")
    mine, rest = parser.parse_known_args()
    if not 0 <= mine.worker < mine.num_workers:
        raise ValueError(f"--worker must be in [0, {mine.num_workers}) — got {mine.worker}.")
    sys.argv = [sys.argv[0], *rest]
    return mine


def scores_path(corpus: str, worker: int) -> str:
    """Beside the corpus, never inside it — the loader reads every file in there."""
    return f"{corpus.rstrip('/')}-scores-w{worker}.jsonl"


def _scored_rows(path: str) -> set[tuple[str, int]]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {(json.loads(line)["shard"], json.loads(line)["row"]) for line in handle if line.strip()}


def word_overlap(said: str, heard: str) -> float:
    """Word-level similarity, punctuation and case ignored.

    A ratio rather than WER: the transcript legitimately differs in contractions
    and hyphenation ("I'm" / "I am", "multimodal" / "multi-modal"), which WER
    would punish as errors while the words are the same.
    """
    tokens = (re.findall(r"[\w']+", said.lower()), re.findall(r"[\w']+", heard.lower()))
    return difflib.SequenceMatcher(None, *tokens).ratio()


def score_corpus(script_args: argparse.Namespace) -> None:
    import soundfile as sf
    import torch

    from veomni.arguments import OmniArguments, parse_omni_args
    from veomni.models.seed_omni.utils.conversation import ConversationItem
    from veomni.trainer.omni import OmniInferencer

    args = parse_omni_args(OmniArguments, preload_path_fields=("model.model_config.modules",))
    kernel = args.model.ops_implementation.attn_implementation
    if "flash_attention_3" not in str(kernel):
        raise ValueError(
            f"transcription needs FA3 here (got {kernel!r}): the pinned flash-attn 2 build returns NaN for "
            f"single-query decode past a 128-token cache, so every transcript past that length would be '!' "
            f"repeated and every row would score 0. See constraint 31."
        )

    shards = sorted(glob.glob(os.path.join(script_args.corpus, "*.parquet")))
    mine = shards[script_args.worker :: script_args.num_workers]

    # Rows already in this worker's score file are skipped and the file is
    # appended to, so a pass that died halfway — or one whose earlier run
    # dropped rows it could not read — costs only what it has left to do.
    already = _scored_rows(scores_path(script_args.corpus, script_args.worker))
    print(
        f"[filter] worker {script_args.worker}: {len(mine)} of {len(shards)} shard(s), "
        f"{len(already)} row(s) already scored",
        flush=True,
    )

    inferencer = OmniInferencer(args)
    generation_kwargs = inferencer._runtime_generation_kwargs()

    import pyarrow.parquet as pq

    written = 0
    with open(scores_path(script_args.corpus, script_args.worker), "a", encoding="utf-8") as handle:
        for shard in mine:
            for index, row in enumerate(pq.read_table(shard).to_pylist()):
                if script_args.limit and written >= script_args.limit:
                    break
                if (os.path.basename(shard), index) in already:
                    continue
                said = next((c["value"] for c in row["conversations"] if c["from"] == "gpt"), "")
                samples, rate = sf.read(io.BytesIO(row["audios"][0]), dtype="float32")
                conversation = [
                    ConversationItem(
                        type="audio",
                        value=samples,
                        role="user",
                        # Typed, because that is what the audio preprocessor reads
                        # to resample: a flat ``sampling_rate`` key leaves a 24 kHz
                        # corpus wav read as 16 kHz, which inflates every clip's
                        # apparent duration by 1.5x and pushes valid ones past the
                        # tower's 30 s window (they then score zero and get culled).
                        meta={
                            AUDIO_METADATA_KEY: AudioMetadata(
                                sampling_rate=int(rate), num_samples=int(samples.shape[0])
                            )
                        },
                    ),
                    ConversationItem(type="text", value=TRANSCRIBE_PROMPT, role="user"),
                ]
                try:
                    request = inferencer.processor.preprocess(conversation, inference=True, **generation_kwargs)
                    inferencer.model.reset()
                    with torch.no_grad():
                        generated = inferencer.model.generate(request, generation_kwargs=generation_kwargs)
                    heard = " ".join(
                        str(item["value"])
                        for item in generated
                        if item.get("type") == "text" and item.get("value") is not None
                    )
                except Exception as error:  # noqa: BLE001 - one bad clip must not end the pass
                    # Recorded as unverifiable rather than skipped: a skipped row
                    # would look unscored to ``--apply``, which then refuses to
                    # run at all. The common cause is a clip longer than the
                    # tower's 30 s window — a legitimate talker target that this
                    # check simply cannot read back, so it is dropped for lack
                    # of evidence rather than kept unverified.
                    print(f"[filter] {os.path.basename(shard)}#{index}: {type(error).__name__}: {error}", flush=True)
                    heard = ""
                    note = f"{type(error).__name__}: {error}"
                else:
                    note = ""
                record = {
                    "shard": os.path.basename(shard),
                    "row": index,
                    "overlap": 0.0 if note else round(word_overlap(said, heard), 4),
                    "heard": heard[:400],
                    "note": note[:200],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
                if written % 20 == 0:
                    print(f"[filter] worker {script_args.worker}: {written} row(s) scored", flush=True)
    print(f"[filter] worker {script_args.worker}: done, {written} row(s) scored", flush=True)


def load_scores(corpus: str) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    for path in sorted(glob.glob(f"{corpus.rstrip('/')}-scores-w*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                scores[(record["shard"], record["row"])] = record["overlap"]
    if not scores:
        raise FileNotFoundError(f"no score files beside {corpus}; run --score first")
    return scores


def apply_marker(corpus: str) -> str:
    """Records that this corpus has been filtered, beside it rather than in it."""
    return f"{corpus.rstrip('/')}-applied.json"


def apply_scores(script_args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    marker = apply_marker(script_args.corpus)
    if os.path.exists(marker):
        # A second pass would not just be redundant, it would be wrong: scores
        # are keyed on (shard, row index) and the first pass renumbered the
        # survivors, so every threshold check below would read some other row's
        # score. The row-count check cannot catch it either — the stale key set
        # is a superset of what is left.
        with open(marker, encoding="utf-8") as handle:
            previous = json.load(handle)
        raise SystemExit(
            f"{script_args.corpus} was already filtered at overlap >= {previous['min_overlap']} "
            f"on {previous['when']} ({previous['kept']} kept, {previous['dropped']} dropped). Scores are keyed "
            f"by row position, which that pass invalidated. Re-score, or start from the unfiltered copy."
        )

    scores = load_scores(script_args.corpus)
    shards = sorted(glob.glob(os.path.join(script_args.corpus, "*.parquet")))

    tables = {shard: pq.read_table(shard) for shard in shards}
    missing = [
        (os.path.basename(shard), index)
        for shard, table in tables.items()
        for index in range(table.num_rows)
        if (os.path.basename(shard), index) not in scores
    ]
    if missing:
        # Rewriting now would silently keep unscored rows, which is the thing
        # this pass exists to remove.
        raise ValueError(f"{len(missing)} row(s) have no score (e.g. {missing[:3]}). Finish scoring before applying.")

    kept_total = dropped_total = 0
    for shard, table in tables.items():
        name = os.path.basename(shard)
        rows = table.to_pylist()
        kept = [row for index, row in enumerate(rows) if scores[(name, index)] >= script_args.min_overlap]
        dropped_total += len(rows) - len(kept)
        kept_total += len(kept)
        if kept:
            pq.write_table(pa.Table.from_pylist(kept, schema=table.schema), shard)
        else:
            os.remove(shard)
        print(f"[filter] {name}: kept {len(kept)}/{len(rows)}", flush=True)

    with open(marker, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "min_overlap": script_args.min_overlap,
                "kept": kept_total,
                "dropped": dropped_total,
            },
            handle,
            indent=2,
        )

    values = sorted(scores.values())
    print(
        f"[filter] kept {kept_total}, dropped {dropped_total} "
        f"({100 * dropped_total / max(kept_total + dropped_total, 1):.1f}%) at overlap >= {script_args.min_overlap}",
        flush=True,
    )
    print(f"[filter] overlap deciles: {[round(values[int(q * (len(values) - 1) / 10)], 2) for q in range(11)]}")


def main() -> None:
    script_args = parse_script_args()
    if script_args.score:
        score_corpus(script_args)
    else:
        apply_scores(script_args)


if __name__ == "__main__":
    main()
