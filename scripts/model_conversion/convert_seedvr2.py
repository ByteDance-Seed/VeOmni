"""Strictly import the official SeedVR2-3B checkpoint into VeOmni HF format."""

import argparse
import hashlib
import json
from pathlib import Path

from veomni.models.diffusers.seedvr2.conditioning_seedvr2 import SeedVR2ConditionConfig
from veomni.models.diffusers.seedvr2.inference import load_official_model


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    source_sha = checksum(args.source / "seedvr2_ema_3b.pth")
    expected_sha = "6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b"
    if source_sha != expected_sha:
        parser.error("source is not the pinned official SeedVR2-3B checkpoint (SHA256 mismatch)")
    model = load_official_model(args.source)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    SeedVR2ConditionConfig(base_model_path=str(args.source.resolve())).save_pretrained(args.output / "condition")
    report = {
        "source_revision": "e4de8c24441a67e1b7df56abea10645059bb1185",
        "weights_revision": "37255ff8cccfb01071b87f635a5948ca8d53117c",
        "source_sha256": source_sha,
        "mapping": "Every source key is prefixed with dit.; no tensor layout or dtype change.",
        "target_tensor_count": len(model.state_dict()),
        "target_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "strict_load": True,
        "dropped": [],
        "initialized": [],
    }
    (args.output / "conversion-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
