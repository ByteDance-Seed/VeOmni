"""Build a tiny conversation-structured SeedOmni V2 demo dataset (parquet).

The dataset pairs the two Janus-1.3B inference results under ``janus_out/``
into self-consistent (input → model-own-output) conversations so that
training on them drives the loss low — a clean correctness check for the
OmniTrainer build/forward/backward path.

Two base conversations, both in the unified ``veomni_omni_demo`` schema
(see ``veomni/data/multimodal/preprocess.py``):

* **Understanding (I2T)** — user sends the generated image + "Describe this
  image in detail."; assistant replies with the model's own caption
  (regenerated via ``infer_und`` on that exact image).
* **Generation (T2I)** — user sends the prompt that produced the image;
  assistant replies with the image as a ``vq_image`` target.

Rows are stored with ``conversations`` as JSON-encoded bytes and ``images``
as a list of PNG bytes, so the parquet is fully self-contained (no external
file path dependencies at train time).

Usage::

    python scripts/multimodal/convert_data/make_janus_omni_demo.py \
        --gen_image janus_out/infer_gen/generated_image_0.png \
        --und_reply /tmp/janus_und_regen/infer_und/reply.txt \
        --gen_prompt "A close-up high-contrast photo of Sydney Opera House at night." \
        --out_dir /mnt/hdfs/user_dir/veomni_omni/data \
        --num_repeat 64
"""

from __future__ import annotations

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq


_UND_PROMPT = "Describe this image in detail."


def _build_rows(gen_image_bytes: bytes, und_reply: str, gen_prompt: str) -> list[dict]:
    """Return the two base conversation rows (understanding + generation)."""
    understanding = {
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
        "images": [gen_image_bytes],
    }

    generation = {
        "source_name": "veomni_omni_demo",
        "conversations": json.dumps(
            [
                {"role": "user", "content": [{"type": "text", "value": gen_prompt}]},
                {"role": "assistant", "content": [{"type": "vq_image"}]},
            ]
        ).encode("utf-8"),
        "images": [gen_image_bytes],
    }
    return [understanding, generation]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen_image", default="janus_out/infer_gen/generated_image_0.png")
    parser.add_argument("--und_reply", default="/tmp/janus_und_regen/infer_und/reply.txt")
    parser.add_argument(
        "--gen_prompt",
        default="A close-up high-contrast photo of Sydney Opera House at night.",
    )
    parser.add_argument("--out_dir", default="/mnt/hdfs/user_dir/veomni_omni/data")
    parser.add_argument(
        "--num_repeat",
        type=int,
        default=64,
        help="How many times to repeat each base conversation (dataset size = 2 * num_repeat).",
    )
    args = parser.parse_args()

    with open(args.gen_image, "rb") as f:
        gen_image_bytes = f.read()
    with open(args.und_reply, encoding="utf-8") as f:
        und_reply = f.read().strip()

    base_rows = _build_rows(gen_image_bytes, und_reply, args.gen_prompt)
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
    print(f"[make_janus_omni_demo]   understanding reply chars: {len(und_reply)}")
    print(f"[make_janus_omni_demo]   gen image bytes: {len(gen_image_bytes)}")


if __name__ == "__main__":
    main()
