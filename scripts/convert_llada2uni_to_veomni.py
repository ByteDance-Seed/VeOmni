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
"""Convert an upstream LLaDA2.0-Uni checkpoint (``model_type: llada2_moe``) for VeOmni's ``llada2_uni``.

The conversion itself lives in :mod:`veomni.models.transformers.llada2_uni.convert_from_raw`
so an installed wheel can run it; this CLI is a thin wrapper around it.

Usage:
    python scripts/convert_llada2uni_to_veomni.py --src /path/to/LLaDA2.0-Uni --dst /path/to/LLaDA2.0-Uni-veomni
    python scripts/convert_llada2uni_to_veomni.py --self-test
"""

import argparse

from veomni.models.transformers.llada2_uni.convert_from_raw import (  # noqa: F401  re-exported for importers
    convert_checkpoint,
    convert_config,
    convert_preprocessor_config,
    image_tokenizer_shard,
    merge_index,
    resolve_mask_token_id,
    self_test,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a LLaDA2.0-Uni checkpoint for VeOmni's llada2_uni model.")
    p.add_argument("--src", type=str, default=None, help="Source LLaDA2.0-Uni checkpoint dir.")
    p.add_argument("--dst", type=str, default=None, help="Output dir (model_type=llada2_uni).")
    p.add_argument("--no-tokenizer", action="store_true", help="Do not copy tokenizer/aux files.")
    p.add_argument("--self-test", action="store_true", help="Run config-mapping self-test (no checkpoint needed).")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.src or not args.dst:
        p.error("--src and --dst are required (or use --self-test)")
    convert_checkpoint(args.src, args.dst, copy_tokenizer=not args.no_tokenizer)


if __name__ == "__main__":
    main()
