# pip install (none beyond stdlib; COCO annotations json only)
"""COCO captions -> Kimi K3 conversation jsonl for the VeOmni VLM trainer.

Produces one JSONL record per caption annotation::

    {"image": "<abs-path>", "width": W, "height": H, "caption": "..."}

The `kimi_k3` data transform (veomni/data/data_transform.py) consumes this
shape: it renders the single-placeholder media template
``<|media_begin|>image {w}x{h}<|media_content|><|media_pad|><|media_end|> {caption}``,
supervises only the caption (+EOS), and preprocesses the image through the
KimiK3Processor. See docs/examples/kimi_k3.md for the full recipe.
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", required=True, help="COCO 2017 root (contains annotations/ and <split>/)")
    parser.add_argument("--split", default="val2017", help="COCO split dir name (default: val2017)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    ann_path = os.path.join(args.coco_root, "annotations", f"captions_{args.split}.json")
    with open(ann_path, encoding="utf-8") as f:
        d = json.load(f)
    meta = {im["id"]: (im["file_name"], im["width"], im["height"]) for im in d["images"]}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for a in d["annotations"]:
            fname, w, h = meta[a["image_id"]]
            rec = {
                "image": os.path.join(args.coco_root, args.split, fname),
                "width": int(w),
                "height": int(h),
                "caption": a["caption"].strip(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} samples -> {args.output}")


if __name__ == "__main__":
    main()
