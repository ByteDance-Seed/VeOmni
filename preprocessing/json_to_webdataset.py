"""convert huggingface json dataset to webdataset format
usage:
python preprocessing/json_to_webdataset.py --dataset_file path/to/dataset.json --dataset_path path/to/images --output_path path/to/output
example:
python preprocessing/json_to_webdataset.py --dataset_file data/AgentNet/agentnet_ubuntu_5k.jsonl --dataset_path data/AgentNet/ubuntu_images/images --output_path data/AgentNet/webdataset/ubuntu_5k
"""
import webdataset as wds
import io
from PIL import Image
import argparse
import os
from tqdm import tqdm
import json

checked = False

def encode_value(args, key, value):
    """根据值的类型，返回扩展名和内容"""
    if isinstance(value, str):
        # 如果是文件路径并且存在，就保存二进制
        if os.path.isfile(os.path.join(args.dataset_path, value)):
            ext = os.path.splitext(value)[-1].lstrip(".")  # 用文件扩展名
            with open(os.path.join(args.dataset_path, value), "rb") as f:
                return ext, f.read()
        else:
            return "txt", value.encode("utf-8")
    elif isinstance(value, (int, float)):
        return "txt", str(value).encode("utf-8")
    else:
        # dict/list 等复杂对象
        return "json", json.dumps(value, ensure_ascii=False).encode("utf-8")

def flatten(obj, parent_key="", sep="."):
    """
    递归展开 dict/list
    - dict: 正常展开
    - list: 展开成 index，继续递归
    """
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten(v, new_key, sep=sep))
    else:
        items[parent_key] = obj
    return items

def main(args):
    from datasets import load_dataset
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    dataset = load_dataset("json", data_files=args.dataset_file)["train"]

    sink = wds.ShardWriter(os.path.join(args.output_path, "shards-%06d.tar"), maxcount=10000)

    for idx, sample in tqdm(enumerate(dataset)):
        key = f"{idx:08d}"

        sample_out = {"__key__": key}
        flat_sample = flatten(sample)
        for k, v in flat_sample.items():
            try:
                ext, content = encode_value(args, k, v)
                sample_out[k + "." + ext] = content
            except Exception as e:
                print(f"Skip {k}: {e}")
        sink.write(sample_out)

    sink.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the JSON dataset file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory for webdataset")
    args = parser.parse_args()
    main(args)
