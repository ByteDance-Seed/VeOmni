# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Convert a normal VeOmni Qwen3 QAT HF export into a torchao MXFP4 artifact."""

import argparse

from veomni.models import build_config, build_foundation_model, build_tokenizer
from veomni.models.qat import export_mxfp4


def main():
    """Export on CPU; the source checkpoint must retain its QAT ops selections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Normal HF checkpoint exported by the QAT training job.")
    parser.add_argument("--output", required=True, help="Empty output directory for the torchao artifact.")
    args = parser.parse_args()
    config = build_config(args.input)
    # Offline conversion must not require the training job's CUDA-only kernels.
    config._set_ops_implementation(
        {
            "attn_implementation": "eager",
            "rms_norm_implementation": "eager",
            "swiglu_mlp_implementation": "eager",
            "rotary_pos_emb_implementation": "eager",
            "cross_entropy_loss_implementation": "eager",
        }
    )
    model = build_foundation_model(config, weights_path=args.input, init_device="cpu", torch_dtype="bfloat16")
    export_mxfp4(model, args.output)
    build_tokenizer(args.input).save_pretrained(args.output)


if __name__ == "__main__":
    main()
