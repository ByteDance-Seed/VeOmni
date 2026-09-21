#!/usr/bin/env bash
set -uo pipefail

source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /efs_rl/m00881603/veomni_dsv4/veomni_0918

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export PATH=/efs_rl/m00881603/veomni_dsv4/veomni_0918/.venv-tf516/bin:${PATH}

common_args=(tasks/train_text.py configs/text/deepseek_v4_npu.yaml
  --model.config_path=/efs_rl/m00881603/hf_weights/deepseek_v4-4layers_1
  --model.model_path=/efs_rl/m00881603/hf_weights/deepseek_v4-4layers_1
  --model.tokenizer_path=/efs_rl/m00881603/hf_weights/deepseek_v4-4layers_1
  --data.train_path=/efs_rl/m00881603/datasets/dapo-math-17k-text.parquet
  --data.datasets_type=iterable --data.train_sample=10000 --data.data_type=plaintext --data.text_keys=text
  --data.max_seq_len=4096 --data.dataloader.shuffle=false --data.dataloader.num_workers=0
  --train.global_batch_size=16 --train.micro_batch_size=1 --train.max_steps=20 --train.dyn_bsz=false
  --train.seed=42 --train.enable_full_determinism=true
  --model.optimizer.lr=3e-4 --model.optimizer.lr_warmup_ratio=0
  --train.checkpoint.save_steps=0 --train.checkpoint.save_epochs=0
  --train.checkpoint.save_hf_weights=false --train.wandb.enable=false)

run_case() {
  local name=$1 port=$2 socket_range=$3 indexer=$4 attention=$5 mhc=$6
  export HCCL_NPU_SOCKET_PORT_RANGE=${socket_range}
  torchrun --nnodes=1 --nproc-per-node=16 --node-rank=0 \
    --master-addr=127.0.0.1 --master-port="${port}" \
    "${common_args[@]}" \
    --model.ops_implementation.dsa_indexer_implementation="${indexer}" \
    --model.ops_implementation.dsa_attention_implementation="${attention}" \
    --model.ops_implementation.mhc_implementation="${mhc}" \
    --train.checkpoint.output_dir="dsv4_tf516_ablation_${name}_1step" \
    > "dsv4_242_tf516_ablation_${name}_1step_20260919.log" 2>&1
}

export DSV4_MHC_USE_EAGER_PRE=1
unset DSV4_MHC_USE_EAGER_POST
run_case fixed_mhc_eager_pre_20step 64430 59300-59350 eager npu npu
unset DSV4_MHC_USE_EAGER_PRE
export DSV4_MHC_USE_EAGER_POST=1
run_case fixed_mhc_eager_post_20step 64440 59400-59450 eager npu npu