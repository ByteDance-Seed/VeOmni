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
"""Build a talker-training corpus by running Qwen3-Omni's own speech-output graph.

Talker training needs a `role="assistant"` audio turn -- the speech the talker
is taught to say -- paired with the assistant *text* it is speaking. No shipped
corpus has one: `voice_assistant` is the opposite shape (spoken question, text
answer), so its clip is an encoder input. This script produces the missing side
by asking the model to answer out loud and recording both halves of what comes
back.

**What this corpus is, and is not.** The speech is the model's own, so training
on it is self-distillation: it teaches the talker nothing it does not already
know, and its ceiling is the checkpoint that produced it. What it is good for is
everything up to that -- making the training round runnable, overfitting one
sample to prove the loss wires through, and shaking out the data path -- because
the one property talker training actually requires is that the text and the
waveform be the *same utterance*, which self-generated pairs satisfy by
construction. Swap in a real TTS corpus for a run whose result is meant to be a
better model.

Prompts default to `voice_assistant`'s human transcripts: real conversational
turns, and reusing them keeps this corpus in the same domain as the thinker mix.

The model loads once and the request loop reuses it, so cost scales with the
prompt count rather than with process launches. Rows are flushed in shards as
they are produced, so a crash at sample 900 keeps the first 899 -- but a rerun
starts over from the first prompt, and refuses a directory that already holds
shards unless told to append.

Output schema, matching what `voice_assistant_talker_preprocess` reads (see
`veomni/data/seed_omni/preprocess.py`):

    conversations  list[{from: "human"|"gpt", value: str}]
    audios         list[bytes]   one wav per gpt turn, in turn order
    source         str           informational; the loader dispatches on the
                                 config's `names:`, not on this column

No system column: the speech system turn is templated by the text encoder at
both ends (`add_speech_system_prompt`), so generation and talker training share
one source for it instead of a per-row copy that could drift.

Usage (single process, eager, `device_map="auto"` spreads the 30B over the
visible GPUs)::

    python scripts/seed_omni/qwen3omni_build_talker_corpus.py \\
        configs/seed_omni/Qwen/qwen3omni_30b_a3b/train/base.yaml \\
        --model.model_path /opt/tiger/models/qwen3omni_split \\
        --model.model_config.modules configs/seed_omni/Qwen/qwen3omni_30b_a3b/infer/modules_infer_eager_audio.yaml \\
        --model.model_config.infer_type audio_output \\
        --corpus-out /mnt/hdfs/veomni/datasets/qwen3omni_talker_selfspeech \\
        --num-samples 512
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from typing import Any

import torch

from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY


SOURCE_NAME = "voice_assistant_talker"
PROMPTS_DEFAULT = "/mnt/hdfs/veomni/datasets/voice_assistant"


def parse_script_args() -> argparse.Namespace:
    """Take this script's own flags off ``argv`` before the launcher parses it.

    ``parse_omni_args`` merges any undeclared ``--a.b`` flag into the config, so
    leaving these in would plant junk keys on ``OmniArguments``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--corpus-out", required=True, help="Output directory for the parquet shards.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="How many prompts to answer out loud, counted across all workers — the list is split round-robin, "
        "so each worker takes roughly num-samples/num-workers of them.",
    )
    parser.add_argument("--prompts-from", default=PROMPTS_DEFAULT, help="Parquet dir/file to read prompts from.")
    parser.add_argument("--shard-size", type=int, default=64, help="Rows per parquet shard.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Write into a directory that already holds shards. Off by default because shard numbering restarts "
        "at 0, so a rerun overwrites the earlier shards and mixes two generations — which is silent if the "
        "checkpoint or the generation settings changed between them.",
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=0,
        help="This worker's index. Prompts are split round-robin across workers so one process per GPU can run "
        "concurrently; each writes its own shard files.",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="How many workers share the prompt list.")
    parser.add_argument("--speaker", default=None, help="Talker voice; the module's default when unset.")
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=0.5,
        help="Drop a reply whose speech is shorter than this — the talker stopped early and the clip does not "
        "cover the text, which would teach it to stop early too.",
    )
    parser.add_argument(
        "--seconds-per-char",
        type=float,
        nargs=2,
        default=(0.02, 0.20),
        metavar=("MIN", "MAX"),
        help="Plausible band for speech duration divided by reply length. A row far outside it is a text/audio "
        "mismatch — truncated speech or a runaway tail — and training on it would teach the talker to say "
        "something other than the text it was given. The default is calibrated on Latin script (~0.06 s/char); "
        "Mandarin runs 0.2-0.3 s per character, so widen it for a corpus in a language that packs more sound "
        "into fewer characters, or every row will be dropped.",
    )
    mine, rest = parser.parse_known_args()
    if not 0 <= mine.worker < mine.num_workers:
        raise ValueError(
            f"--worker must be in [0, {mine.num_workers}) — got {mine.worker}. Out of range means either an "
            f"empty prompt slice (a run that writes nothing and exits happy) or two workers racing on the same "
            f"shard filenames."
        )
    sys.argv = [sys.argv[0], *rest]
    return mine


def read_prompts(path: str, limit: int) -> list[str]:
    """Human-side transcripts, deduplicated, in file order.

    Deduplicated because `voice_assistant` repeats near-identical "introduce
    yourself" turns, and a corpus of one question teaches one answer.
    """
    import pyarrow.parquet as pq

    files = []
    if os.path.isdir(path):
        files = sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".parquet"))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"no parquet found at {path}")

    prompts: list[str] = []
    seen: set[str] = set()
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=256, columns=["conversations"]):
            for row in batch.to_pylist():
                for turn in row["conversations"]:
                    if turn["from"] != "human":
                        continue
                    text = turn["value"].strip()
                    if text and text not in seen:
                        seen.add(text)
                        prompts.append(text)
            if len(prompts) >= limit:
                return prompts[:limit]
    return prompts[:limit]


def wav_bytes(waveform: torch.Tensor, sampling_rate: int) -> bytes:
    """Encode one mono utterance as a wav, the form a parquet row carries.

    16-bit PCM rather than float, which halves the corpus; the data layer reads
    it back as float32 at this same rate either way. The tradeoff is that PCM
    clips above full scale, so an overshooting utterance is refused here — the
    clipped waveform is what the codec turns into RVQ targets, and the
    distortion would be baked into the labels.
    """
    import soundfile as sf

    samples = waveform.detach().float().cpu().squeeze()
    if samples.ndim != 1:
        raise ValueError(f"expected one mono utterance, got shape {tuple(samples.shape)}")
    peak = float(samples.abs().max())
    if peak > 1.0:
        raise ValueError(f"utterance peaks at {peak:.3f}, above full scale; 16-bit PCM would clip it")
    buffer = io.BytesIO()
    sf.write(buffer, samples.numpy(), int(sampling_rate), format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def reply_of(generated: list[dict[str, Any]]) -> tuple[str, torch.Tensor | None, int | None]:
    """The text and the speech of one answer, as the FSM left them on ``ctx``."""
    texts = [str(item["value"]) for item in generated if item.get("type") == "text" and item.get("value") is not None]
    audio = next(
        (item for item in generated if item.get("type") == "audio" and item.get("value") is not None),
        None,
    )
    if audio is None:
        return "\n".join(texts), None, None
    # The codec declares its timeline as typed ``AudioMetadata`` under
    # ``AUDIO_METADATA_KEY`` — not a flat ``meta["sampling_rate"]``. Reading the
    # flat key finds nothing on every valid reply, so every row would be skipped.
    metadata = (audio.get("meta") or {}).get(AUDIO_METADATA_KEY)
    rate = getattr(metadata, "sampling_rate", None)
    if rate is None:
        # The rate is the vocoder's to declare; a corpus row without it cannot
        # be read back, since the data layer refuses to assume one.
        raise ValueError(f"generated audio carries no meta[{AUDIO_METADATA_KEY!r}].sampling_rate")
    return "\n".join(texts), audio["value"], int(rate)


class ShardWriter:
    """Flush rows to numbered parquet shards as they arrive.

    Sharding rather than one file at the end: generation is slow enough that a
    run is worth resuming, and a partial corpus is still a usable one.
    """

    def __init__(self, out_dir: str, shard_size: int, tag: str = "0") -> None:
        self._out_dir = out_dir
        self._shard_size = shard_size
        self._tag = tag
        self._rows: list[dict[str, Any]] = []
        self._shard = self._next_free_shard(out_dir, tag)
        self._written = 0
        os.makedirs(out_dir, exist_ok=True)

    def _next_free_shard(self, out_dir: str, tag: str) -> int:
        """First shard number this tag has not used yet.

        Counting from zero would silently overwrite the previous run's shards on
        ``--append`` — the very rows the flag exists to keep — because the tag is
        the worker index and does not change between runs.
        """
        pattern = re.compile(rf"^{re.escape(tag)}-(\d+)\.parquet$")
        used = (
            [int(m.group(1)) for name in os.listdir(out_dir) if (m := pattern.match(name))]
            if os.path.isdir(out_dir)
            else []
        )
        return max(used, default=-1) + 1

    def add(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        self._written += 1
        if len(self._rows) >= self._shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(
            self._rows,
            schema=pa.schema(
                [
                    ("conversations", pa.list_(pa.struct([("from", pa.string()), ("value", pa.string())]))),
                    ("audios", pa.list_(pa.binary())),
                    ("source", pa.string()),
                ]
            ),
        )
        path = os.path.join(self._out_dir, f"{self._tag}-{self._shard}.parquet")
        pq.write_table(table, path)
        print(f"[corpus] wrote {len(self._rows)} row(s) -> {path}", flush=True)
        self._rows = []
        self._shard += 1

    @property
    def written(self) -> int:
        return self._written


def main() -> None:
    script_args = parse_script_args()

    # Imported after argv is trimmed so the launcher parses only its own flags.
    from veomni.arguments import OmniArguments, parse_omni_args
    from veomni.trainer.omni import OmniInferencer
    from veomni.trainer.omni.omni_inferencer import InferenceRequest

    args = parse_omni_args(OmniArguments, preload_path_fields=("model.model_config.modules",))
    infer_type = args.model.launcher_config("infer_type")
    if infer_type != "audio_output":
        raise ValueError(
            f"this script records generated speech, so it needs the speech graph: "
            f"--model.model_config.infer_type audio_output (got {infer_type!r})."
        )
    modules = args.model.launcher_config("modules") or {}
    text_encoder_config = (modules.get("qwen3omni_text_encoder") or {}).get("model_config") or {}
    if not text_encoder_config.get("add_speech_system_prompt"):
        raise ValueError(
            "this script needs qwen3omni_text_encoder.model_config.add_speech_system_prompt: true — without "
            "that turn the thinker answers in bulleted prose that never reaches <|im_end|> in budget, so the "
            "talker never runs and every row is dropped for having no audio. Pass "
            "--model.model_config.modules configs/seed_omni/Qwen/qwen3omni_30b_a3b/infer/"
            "modules_infer_eager_audio.yaml."
        )
    kernel = args.model.ops_implementation.attn_implementation
    if "flash_attention_3" not in str(kernel):
        raise ValueError(
            f"generation needs FA3 here (got {kernel!r}): the pinned flash-attn 2 build returns NaN for every "
            f"single-query decode step past a 128-token cache, so a reply longer than that degenerates to '!' "
            f"repeated. That reply still carries text and speech and can pass the pace filter, so the run would "
            f"write plausible-looking rows of garbage rather than fail. See constraint 31. Pass "
            f"--model.ops_implementation.attn_implementation flash_attention_3."
        )

    existing = (
        [name for name in os.listdir(script_args.corpus_out) if name.endswith(".parquet")]
        if os.path.isdir(script_args.corpus_out)
        else []
    )
    if existing and not script_args.append:
        raise ValueError(
            f"{script_args.corpus_out} already holds {len(existing)} shard(s). A new run would add to them; "
            f"pass --append if that is what you want (shard numbering continues past the existing ones, so "
            f"nothing is overwritten), or choose a fresh directory."
        )

    prompts = read_prompts(script_args.prompts_from, script_args.num_samples)
    prompts = prompts[script_args.worker :: script_args.num_workers]
    print(
        f"[corpus] worker {script_args.worker}/{script_args.num_workers}: {len(prompts)} prompt(s) "
        f"from {script_args.prompts_from}",
        flush=True,
    )

    # `generate()` is one prompt per process; the loop below reuses the built
    # model, which is the whole reason this is a script and not a shell loop.
    inferencer = OmniInferencer(args)
    generation_kwargs = inferencer._runtime_generation_kwargs()
    if script_args.speaker is not None:
        generation_kwargs["speaker"] = script_args.speaker

    writer = ShardWriter(script_args.corpus_out, script_args.shard_size, tag=f"w{script_args.worker}")
    skipped: dict[str, int] = {}
    started = time.time()
    for index, prompt in enumerate(prompts):
        try:
            ctx = inferencer._run(
                InferenceRequest(
                    prompt=prompt,
                    images=[],
                    generation_kwargs=generation_kwargs,
                )
            )
            text, waveform, rate = reply_of(ctx["generated"])
        except Exception as error:  # noqa: BLE001 - one bad prompt must not end a long run
            skipped["error"] = skipped.get("error", 0) + 1
            print(f"[corpus] {index}: {type(error).__name__}: {error}", flush=True)
            continue

        if waveform is None or not text.strip():
            # Half an answer is worse than none here: the talker is scored on
            # speaking this text, so a row missing either side is not a pair.
            # Printed rather than counted silently: the usual cause is the text
            # phase running to its token budget without ever reaching
            # `<|im_end|>`, so the FSM never hands over to the talker — and the
            # reply is the only thing that distinguishes that from a real error.
            skipped["incomplete"] = skipped.get("incomplete", 0) + 1
            print(
                f"[corpus] {index}: incomplete — text={len(text)} chars, audio={'yes' if waveform else 'no'}"
                f"\n         reply: {text[:300]!r}",
                flush=True,
            )
            continue
        seconds = waveform.numel() / rate
        if seconds < script_args.min_speech_seconds:
            skipped["too_short"] = skipped.get("too_short", 0) + 1
            continue
        low, high = script_args.seconds_per_char
        pace = seconds / max(len(text), 1)
        if not low <= pace <= high:
            skipped["pace"] = skipped.get("pace", 0) + 1
            print(
                f"[corpus] {index}: dropped — {seconds:.1f}s of speech for {len(text)} chars "
                f"({pace:.3f}s/char, outside {low}-{high})",
                flush=True,
            )
            continue

        writer.add(
            {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": text},
                ],
                "audios": [wav_bytes(waveform, rate)],
                "source": SOURCE_NAME,
            }
        )
        if index % 10 == 0:
            elapsed = time.time() - started
            print(
                f"[corpus] {index + 1}/{len(prompts)} kept={writer.written} "
                f"{elapsed / (index + 1):.1f}s/sample last={seconds:.1f}s @{rate}Hz",
                flush=True,
            )
    writer.flush()

    manifest = {
        "source": SOURCE_NAME,
        "rows": writer.written,
        "prompts_from": script_args.prompts_from,
        "checkpoint": args.model.model_path,
        "infer_type": infer_type,
        "generation_kwargs": {k: v for k, v in generation_kwargs.items() if isinstance(v, (str, int, float, bool))},
        "skipped": skipped,
        "self_distilled": True,
    }
    # Beside the corpus directory, never inside it: ``get_data_files`` lists the
    # whole directory and infers the format from the first name it sees, so a
    # manifest among the shards makes the corpus load as JSON.
    manifest_path = f"{script_args.corpus_out.rstrip('/')}-manifest-w{script_args.worker}.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[corpus] done: {writer.written} row(s), skipped {skipped or 'nothing'}", flush=True)


if __name__ == "__main__":
    main()
