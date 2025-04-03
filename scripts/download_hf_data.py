import argparse

from huggingface_hub import snapshot_download


"""
python3 scripts/download_hf_data.py --repo_id HuggingFaceFW/fineweb --local_dir ./fineweb/ --allow_patterns sample/10BT/*
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--local_dir", type=str, default="./fineweb/")
    parser.add_argument("--allow_patterns", type=str, default=None)
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = args.local_dir
    allow_patterns = args.allow_patterns

    folder = snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
