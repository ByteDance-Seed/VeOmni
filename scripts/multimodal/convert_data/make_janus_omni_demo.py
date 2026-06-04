"""Build a tiny conversation-structured SeedOmni V2 demo dataset (parquet).

The dataset pairs the two Janus-1.3B inference results under ``janus_out/``
into self-consistent (input → model-own-output) conversations so that
training on them drives the loss low — a clean correctness check for the
OmniTrainer build/forward/backward path.

Three base conversations, all in the unified ``veomni_omni_demo`` schema
(see ``docs/usage/seedomni_data_format.md``):

* **Understanding (I2T)** — user sends the image + "Describe this image in
  detail."; assistant replies with text.
* **Generation (T2I)** — user sends the prompt; assistant replies with
  ``type="image"`` (generation target).
* **Interleave (UG)** — two messages only (like a flat chat JSON): user
  ``text + image + text``, assistant ``image + text``; ``images[0]`` pairs
  the user image, ``images[1]`` pairs the assistant image.

Rows are stored with ``conversations`` as JSON-encoded bytes and ``images``
as a list of PNG bytes, so the parquet is fully self-contained (no external
file path dependencies at train time).

Usage::

    python scripts/multimodal/convert_data/make_janus_omni_demo.py \
        --gen_image janus_out/infer_gen/generated_image_0.png \
        --und_reply janus_out/infer_und/reply.txt \
        --gen_prompt "A close-up high-contrast photo of Sydney Opera House at night." \
        --out_dir outputs/janus_conversation_ipdb_debug/data \
        --only_interleave --num_repeat 4
"""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


_UND_PROMPT = "Describe this image in detail."
_DEFAULT_UND_REPLY = "The image shows a benchmark chart on the left and a generated scene on the right."
_DEFAULT_GEN_PROMPT = "A close-up high-contrast photo of Sydney Opera House at night."


def _synthetic_png_bytes(*, color: tuple[int, int, int] = (80, 120, 200), size: int = 128) -> bytes:
    """Minimal self-contained PNG when no real image file is available."""
    buf = BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


def _load_image_bytes(path: str) -> bytes:
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    print(f"[make_janus_omni_demo] warning: {path!r} missing — using synthetic PNG")
    return _synthetic_png_bytes()


def _load_und_reply(path: str) -> str:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    print(f"[make_janus_omni_demo] warning: {path!r} missing — using default caption")
    return _DEFAULT_UND_REPLY


def _build_understanding_row(user_image_bytes: bytes, und_reply: str) -> dict:
    return {
        "source_name": "veomni_omni_demo",
        "conversations": json.dumps(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": _UND_PROMPT},
                        {"type": "image"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "value": und_reply}]},
            ]
        ).encode("utf-8"),
        "images": [user_image_bytes],
    }


def _build_generation_row(gen_image_bytes: bytes, gen_prompt: str) -> dict:
    return {
        "source_name": "veomni_omni_demo",
        "conversations": json.dumps(
            [
                {"role": "user", "content": [{"type": "text", "value": gen_prompt}]},
                {"role": "assistant", "content": [{"type": "image"}]},
            ]
        ).encode("utf-8"),
        "images": [gen_image_bytes],
    }


def _build_interleave_row(
    user_image_bytes: bytes,
    gen_image_bytes: bytes,
    und_reply: str,
    gen_prompt: str,
) -> dict:
    """User ``text+image+text`` + assistant ``image+text`` (two messages total)."""
    return {
        "source_name": "veomni_omni_demo",
        "conversations": json.dumps(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": _UND_PROMPT},
                        {"type": "image"},
                        {"type": "text", "value": gen_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "value": und_reply},
                    ],
                },
            ]
        ).encode("utf-8"),
        "images": [user_image_bytes, gen_image_bytes],
    }


def _build_rows(
    user_image_bytes: bytes,
    gen_image_bytes: bytes,
    und_reply: str,
    gen_prompt: str,
    *,
    include_interleave: bool,
) -> list[dict]:
    rows = [
        _build_understanding_row(user_image_bytes, und_reply),
        _build_generation_row(gen_image_bytes, gen_prompt),
    ]
    if include_interleave:
        rows.append(_build_interleave_row(user_image_bytes, gen_image_bytes, und_reply, gen_prompt))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen_image", default="janus_out/infer_gen/generated_image_0.png")
    parser.add_argument("--und_image", default=None, help="User-side image; defaults to --gen_image")
    parser.add_argument("--und_reply", default="janus_out/infer_und/reply.txt")
    parser.add_argument("--gen_prompt", default=_DEFAULT_GEN_PROMPT)
    parser.add_argument("--out_dir", default="/mnt/hdfs/user_dir/veomni_omni/data")
    parser.add_argument(
        "--num_repeat",
        type=int,
        default=64,
        help="Repeat each selected base row this many times.",
    )
    parser.add_argument(
        "--include_interleave",
        action="store_true",
        help="Add the mixed user-image + assistant-image row to the dataset.",
    )
    parser.add_argument(
        "--only_interleave",
        action="store_true",
        help="Emit only the interleave row (handy for ipdb / encoder routing debug).",
    )
    args = parser.parse_args()

    und_image_path = args.und_image or args.gen_image
    user_image_bytes = _load_image_bytes(und_image_path)
    gen_image_bytes = _load_image_bytes(args.gen_image)
    und_reply = _load_und_reply(args.und_reply)

    if args.only_interleave:
        base_rows = [_build_interleave_row(user_image_bytes, gen_image_bytes, und_reply, args.gen_prompt)]
    else:
        base_rows = _build_rows(
            user_image_bytes,
            gen_image_bytes,
            und_reply,
            args.gen_prompt,
            include_interleave=args.include_interleave,
        )

    rows = [base_rows[i % len(base_rows)] for i in range(len(base_rows) * args.num_repeat)]

    table = pa.Table.from_pydict(
        {
            "source_name": pa.array([r["source_name"] for r in rows], type=pa.string()),
            "conversations": pa.array([r["conversations"] for r in rows], type=pa.binary()),
            "images": pa.array([r["images"] for r in rows], type=pa.list_(pa.binary())),
        }
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "janus_omni_demo.parquet")
    pq.write_table(table, out_path)
    print(f"[make_janus_omni_demo] wrote {table.num_rows} rows → {out_path}")
    print(f"[make_janus_omni_demo]   base row kinds: {len(base_rows)}")
    print(f"[make_janus_omni_demo]   understanding reply chars: {len(und_reply)}")
    print(f"[make_janus_omni_demo]   user image bytes: {len(user_image_bytes)}")
    print(f"[make_janus_omni_demo]   gen image bytes: {len(gen_image_bytes)}")


if __name__ == "__main__":
    main()
