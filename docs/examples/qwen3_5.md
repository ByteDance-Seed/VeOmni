# Qwen3.5 training guide

> **Note:** Qwen3.5 requires transformers v5 (now the project default).

## Install dependencies

Qwen3.5 depends on transformers v5, which is now the default install:

```shell
uv sync --extra gpu
```

## Download dataset

### Vision-Language Dataset

Download the [COCO2017](https://images.cocodataset.org/zips/train2017.zip) dataset and download the data annotation JSON file [sharegpt4v_instruct_gpt4-vision_cap100k.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main).

Modify the sharegpt4v_instruct_gpt4-vision_cap100k.json:

```python
import json
with open('sharegpt4v_instruct_gpt4-vision_cap100k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered_data = []
for item in data:
    if item.get('image', '').startswith('coco'):
        new_item = item.copy()
        image_path = new_item.pop('image')
        new_item['images'] = [image_path]
        filtered_data.append(new_item)
with open('sharegpt4v_instruct_gpt4-vision_cap100k_coco.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
```

### Text-only Dataset (Optional)

If you want to train on text-only data, download the [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset.

```python
import pyarrow.parquet as pq
input_path = "tulu-3-sft-mixture/data/train-00000-of-00006.parquet"
output_path = "tulu-first2000.parquet"
# Read parquet file and extract the first 2000 rows
table = pq.read_table(input_path)
table_first_2000 = table.slice(0, 2000)
pq.write_table(table_first_2000, output_path)
```

## Download Qwen3.5 model

Dense 9B:
```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3.5-9B \
    --local_dir ${HOME}/Qwen3.5-9B
```

MoE 35B:
```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3.5-35B-A3B \
    --local_dir ${HOME}/Qwen3.5-35B-A3B
```

## Start training on GPU

### Qwen3.5-9B VL Training

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5/qwen3_5_vl.yaml \
    --model.model_path ./Qwen/Qwen3.5-9B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --train.max_steps 20
```

### Qwen3.5 MoE 35B VL Training

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl.yaml \
    --model.model_path ./Qwen/Qwen3.5-35B-A3B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --train.max_steps 20
```

### Text-only training on GPU
Testing in 8x80GB GPUs.

Qwen3.5 Dense 9B:
```shell
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-9B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --model.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --model.accelerator.init_device meta \
    --train.max_steps 20 \
    --train.checkpoint.output_dir ./exp/qwen3_5_9b_sft
```

Qwen3.5-35B-A3B. 8X80GB GPU will likely OOM due to the model size. Use 8X192GB GPU or more GPUs.

```shell
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-35B-A3B \
    --model.ops_implementation.moe_implementation fused_triton \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --model.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --model.accelerator.init_device meta \
    --train.global_batch_size 16 \
    --train.checkpoint.output_dir ./exp/qwen3_5_35b_a3b_sft
```

### FlashQLA GatedDeltaNet backend on Hopper/Blackwell GPUs

On SM90-SM100 NVIDIA GPUs where `flash_qla` is installed, you can select the FlashQLA
`chunk_gated_delta_rule` backend:

```yaml
model:
  ops_implementation:
    chunk_gated_delta_rule_implementation: flash_qla
```

When `chunk_gated_delta_rule_implementation: flash_qla` is active (and not in recurrent cache
decode mode), Qwen3.5 and Qwen3.5-MoE pass grouped `query` and `key` heads
(`num_k_heads < num_v_heads`) directly to `flash_qla` instead of materializing `repeat_interleave`
copies before the kernel. Other backends (`fla`, `eager`, NPU `triton`, `fla_npu`) and recurrent
decode paths keep the expanded Q/K layout.

## Start training on NPU

Qwen3.5 runs on Ascend NPUs, but its GatedDeltaNet kernels are **not** auto-selected: all three
model-local kernels (`rms_norm_gated`, `causal_conv1d`, `chunk_gated_delta_rule`) default to `fla`,
which requires a GPU and raises on NPU. They must be set explicitly.

### Kernel backends

The Ascend rows are `torch_npu` for gated RMSNorm and `triton`, the vendored MindSpeed-MM
Triton-Ascend kernels, for the other two ops:

```yaml
model:
  ops_implementation:
    rms_norm_gated_implementation: torch_npu
    causal_conv1d_implementation: triton
    chunk_gated_delta_rule_implementation: triton
```

`rms_norm_gated` is the exception: it is a `torch_npu` op composition rather than Triton, so only
`causal_conv1d` and `chunk_gated_delta_rule` need `triton-ascend`. That package is not a declared
VeOmni dependency — install it on the NPU host. VeOmni main pins PyTorch and `torch_npu` 2.10.0; the corresponding
CANN 9.0.0 stack uses v3.2.1:

```bash
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

See the [triton-ascend quick-start](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/quick_start.md)
for the compatibility matrix and troubleshooting. Keep CANN, `torch_npu`,
and `triton-ascend` on a mutually compatible release set.

> **arch35:** the old `causal_conv1d` guard that raised `NotImplementedError` on
> `Ascend910_95` / `Ascend950` has been removed. Forward and backward parity on
> arch35 should be verified on the target hardware and software stack.

### `fla_npu` — AscendC fused backend (`chunk_gated_delta_rule` only)

`fla_npu` is a second NPU backend for `chunk_gated_delta_rule` that delegates the heavy GDN
compute to the external [`fla_npu`](https://github.com/flashserve/flash-linear-attention-npu)
operator library, driven through its stable Python entry `fla_npu.ops.ascendc`; only the Triton
glue stays vendored under `veomni/ops/kernels/gated_delta_rule/vendor/triton_core`. It coexists
with `triton` (pure vendored Triton-Ascend), which remains the fallback.
Set only `chunk_gated_delta_rule` to `fla_npu`; keep `rms_norm_gated` on `torch_npu` and
`causal_conv1d` on `triton`:

```yaml
model:
  ops_implementation:
    rms_norm_gated_implementation: torch_npu
    causal_conv1d_implementation: triton
    chunk_gated_delta_rule_implementation: fla_npu
```

`fla_npu` is not a declared VeOmni dependency (same as `triton-ascend`). It covers the A2
(`ascend910b`), A3 (`ascend910_93`) and A5 / arch35 (`ascend950`) targets; the library picks the
arch22 or arch35 kernel path from the SoC, and this adapter uses the BF16 K=V=128,
chunk_size=64 contract both paths share. The vendored `causal_conv1d` no longer rejects
arch35 (`Ascend910_95` / `Ascend950`); end-to-end arch35 validation remains necessary (see above).

These instance-local GDN selectors also apply to Qwen4-Exp. Keep the model's
configured output-gate activation: selecting AscendC does not change sigmoid
to SiLU. The NPU parity test covers ordinary sequences and packed samples with
boundaries that are not multiples of the recurrent chunk size. The vendored
convolution also selects a smaller channel tile for head shards such as
Qwen4-Exp U8CP2's 640 channels, preserving the existing 32-channel alignment
and UB budget.

The following prebuilt NPU images on [quay.io](https://quay.io/repository/ascend/veomni?tab=tags)
bundle the legacy `fla_npu` build from `c2e3d83f`, which does not provide the GDN API required
by this adapter:

- A2: `quay.io/ascend/veomni:v0.1.11-cann9.0.0-torch_npu2.10.0.post2-A2-ubuntu22.04-py3.11-veomni-latest`
- A3: `quay.io/ascend/veomni:v0.1.11-cann9.0.0-torch_npu2.10.0.post2-A3-ubuntu22.04-py3.11-veomni-latest`

Before selecting the `fla_npu` backend in either image, replace its legacy package using the
`a7c608f` wheel installation steps below. Follow the same steps on a bare host or an A5
system, selecting the matching SoC target. A compatible A5 image is not listed here.

Build and install the `a7c608f` wheel from the
[`flash-linear-attention-npu`](https://github.com/flashserve/flash-linear-attention-npu) sources.
The library does not install `torch`, `torch_npu` or `triton-ascend` for you; those must match
your CANN and Python versions, and `FLA_NPU_SOC` must match the chip — A2 → `ascend910b`,
A3 → `ascend910_93`, A5 → `ascend950`.

```bash
# CANN first: A2/A3 need >= 8.5.2, A5 needs >= 9.0.0. Re-source it in every new shell.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

git clone https://github.com/flashserve/flash-linear-attention-npu
cd flash-linear-attention-npu
git checkout a7c608f

# Build dependencies of the operator library only.
python -m pip install -r requirements.txt
python scripts/check_npu_env.py           # add --build-only on a host without an NPU

FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
python -m pip install --force-reinstall --no-cache-dir --no-deps dist/*.whl

# Verify the wheel registered its AscendC ops.
python -c "import fla_npu; print('ok')"
python -c "from fla_npu.ops import ascendc; assert all(callable(getattr(ascendc, n, None)) for n in ('chunk_gated_delta_rule_fwd', 'chunk_gated_delta_rule_bwd')), 'Missing required fla_npu GDN API'"
python scripts/check_packaged_wheel_api.py
```

`--no-deps` is required: without it pip re-resolves `torch` and can replace the Ascend
`torch_npu` stack with a CUDA build. The wheel embeds the AscendC operators, so no run package
(`build.sh --pkg`), no install into `$ASCEND_OPP_PATH/vendors` and no separate
`torch_custom/fla_npu` build are needed any more.

Coming from `a7c608f` or earlier, uninstall the old package and delete its leftovers under
`site-packages` (`fla_npu/`, `custom_aclnn_extension_lib*.so`, any self-built `libopapi.so`)
before running the steps above. The call entry moved from `torch.ops.npu.*` / `torch_npu.ops.*`
to `fla_npu.ops.ascendc`, and the legacy `torch.ops.npu.*` path is no longer enabled by default;
`FLA_NPU_INCREMENTAL_BUILD`, `FLA_NPU_SKIP_RUN_BUILD` and `FLA_NPU_SKIP_RUN_INSTALL` were
removed with it.

If the `fla_npu` package is missing when `chunk_gated_delta_rule_implementation: fla_npu` is
selected, the backend raises an actionable error when the model constructs its `VeomniOp` handle
(pointing back to the install step or to `triton` / `eager`).

The native convolution qualification uses independent FP64 segment references
computed from the exact BF16-rounded inputs and upstream gradient. Using separate
BF16 reference convolutions rounds each segment's shared-weight gradient before
addition; cancellation can make that reference less accurate than the fused
FP32 reduction. The output and gradient tolerances are unchanged.

### Qwen3.5-9B VL Training

There is no dedicated dense NPU config; reuse the GPU one and set the GatedDeltaNet kernels on the
command line. Nothing else needs overriding: the YAML's `attn_implementation: flash_attention_2` is
also the recommended NPU value (see [NPU-friendly operator
configurations](../hardware_support/typical_usage.md)), and `rms_norm`, `rotary_pos_emb`,
`swiglu_mlp` and `cross_entropy_loss` are left unset there, so `VeomniOp` resolves them to the NPU defaults
from `OP_REGISTRY.default_impl`. The three
GatedDeltaNet ops default to `"fla"` instead, which is why they must be explicit:

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5/qwen3_5_vl.yaml \
    --model.model_path ${HOME}/Qwen3.5-9B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --model.ops_implementation.rms_norm_gated_implementation torch_npu \
    --model.ops_implementation.causal_conv1d_implementation triton \
    --model.ops_implementation.chunk_gated_delta_rule_implementation fla_npu \
    --train.max_steps 20
```

Set `chunk_gated_delta_rule_implementation` to `triton` if `fla_npu` is not installed.

### Qwen3.5 MoE 35B VL Training

A ready-to-run MoE-VL config wired for `fla_npu` (plus `fused_npu` MoE, NPU RMSNorm/RoPE and
Ulysses SP) is provided at
[`configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_ascendc.yaml`](../../configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_ascendc.yaml):

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_ascendc.yaml \
    --model.model_path ${HOME}/Qwen3.5-35B-A3B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --train.max_steps 20
```

The config sets `ulysses_size: 4`, so the world size must be a multiple of 4. To run without
sequence parallelism, override `--model.accelerator.ulysses_size 1`; the attention kernels adapt
on their own.

### Qwen3.5 MoE 35B VL Muon training

The Ascend VL recipe uses UP1 and EP4, `fused_npu` MoE, the AscendC GatedDeltaNet
backend, and the pure-PyTorch `gram` Muon backend:

```shell
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_muon_ascendc.yaml \
    --model.model_path ${HOME}/Qwen3.5-35B-A3B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --train.max_steps 20
```

`muon_expert_zero_comm: true` selects whole-expert `Shard(0)` when the
EP-local expert count is divisible by the EP-FSDP size; otherwise VeOmni logs
a warning and falls back to the communication path.

## Shared-prefix training

In GRPO-style RL each prompt is sampled `n` times and the `n` rollouts are trained as `n` packed
`prompt + response` sequences, so the prompt is computed `n` times. With shared-prefix training the
text backbone computes each prompt once per micro-batch: GatedDeltaNet layers fork the recurrent
state at the end of the prompt, softmax-attention layers reuse the prompt's keys and values, and the
LM head scores each shared position once. Outputs and gradients match packed training to within the
difference that re-batching the same sequences produces by itself.

The work drops from `n * (P + R)` tokens to `P + n * R`, and on Qwen3.5-4B about 84% of that ratio is
realised. Long prompts gain most; under roughly 1k prompt tokens there is little to save.

It is a field on the **model config**, off by default, and it needs a non-eager linear-attention
backend: `triton` on Ascend, `fla` on GPU.

```json
{ "text_config": { "shared_prefix_training": true } }
```

Set it in the `config.json` of the model directory the trainer loads. Nothing else changes: detection
is internal to the patched modeling, so a trainer driving VeOmni needs no flag of its own and no
adapter code. When nothing in a micro-batch shares a prefix, or the batch carries image or video
inputs, the ordinary packed path runs unchanged.

What the caller has to get right is that a micro-batch holds **whole rollout groups**, i.e. a
multiple of `n` sequences, with no reordering between packing and the forward. A group split across
micro-batches is still correct, only slower, since just the rollouts that land together share work.
With `verl` (`model_engine=veomni`) that means the legacy trainer with group-aware batch balancing,
`use_dynamic_bsz` off, and `ppo_micro_batch_size_per_gpu` a multiple of `rollout.n`; the config field
travels through `actor_rollout_ref.model.hf_config_path`, which is why no verl-side change is needed.

Current limits: text-only prompts; dense Qwen3.5; no Ulysses SP, MTP or KV cache (each raises).

## Ulysses Sequence Parallelism

Qwen3.5 supports Ulysses sequence parallelism for both its softmax attention layers and
linear attention (GatedDeltaNet) layers. This enables training with longer sequences by
distributing the sequence across multiple GPUs.

To enable Ulysses SP, set `model.accelerator.ulysses_size`. VeOmni derives the effective
data-parallel size from the world size and the other parallel dimensions; set
`model.accelerator.dp_shard_size` only when you need to pin the FSDP shard degree explicitly.
For the example below, the total GPU count is `dp_shard_size * ulysses_size = 4 * 2 = 8`.

```shell
# Example: 8 GPUs, dp=4, sp=2
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-9B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --model.accelerator.dp_shard_size 4 \
    --model.accelerator.ulysses_size 2 \
    --model.ops_implementation.attn_implementation flash_attention_3
```

### Requirements

- `flash_attention_2` or `flash_attention_3` attention implementation (softmax layers use
  VeOmni's flash attention with built-in SP support).
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) installed
  (for GatedDeltaNet triton kernels).
- `num_k_heads` and `num_v_heads` (linear attention head counts) must be divisible by
  `ulysses_size`.

### Selecting linear-attention kernels

GatedDeltaNet has three instance-local kernels: `rms_norm_gated`, `causal_conv1d`, and
`chunk_gated_delta_rule`. All three default to `fla` — the FLA Triton kernels shipped under the
`gpu` extra, which is the recommended value on GPU and required for varlen training.

To switch `chunk_gated_delta_rule` to QwenLM's [`flash-qla`](https://github.com/QwenLM/FlashQLA)
kernel (also shipped under the `gpu` extra), set the field explicitly:

```yaml
model:
  ops_implementation:
    chunk_gated_delta_rule_implementation: flash_qla
```

On NPU the default `fla` raises, so all three fields must be set explicitly — see
[Kernel backends](#kernel-backends) under *Start training on NPU*.

### How It Works

Qwen3.5 is a hybrid model alternating between softmax and linear attention layers:

- **Softmax attention layers** — SP is handled transparently by VeOmni's `flash_attention_forward`,
  which performs all-to-all gather/scatter around the flash attention kernel.
- **Linear attention layers (GatedDeltaNet)** — SP is handled explicitly in the patched
  `Qwen3_5GatedDeltaNet.forward`. Q/K/V/b/a projections are all-to-all'd to gather the full
  sequence with local heads, the causal conv1d runs with sharded weights, the recurrent attention
  kernel runs on local heads, and the output is all-to-all'd back.

For detailed implementation notes, see the
[Ulysses documentation](../key_features/ulysses.md#-linear-attention-ulysses-gateddeltanet).
For Ascend 910B4 Triton code generation, export TRITON_ASCEND_ARCH=Ascend910B4
before the first Triton import. The NPU CI regression step sets this variable
explicitly; on a non-910B4 runner the regression test skips after checking the
runtime device name.
