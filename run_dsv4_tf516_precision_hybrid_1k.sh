#!/usr/bin/env bash
set -euo pipefail

source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /efs_rl/m00881603/veomni_dsv4/veomni_0918

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export PATH=/efs_rl/m00881603/veomni_dsv4/veomni_0918/.venv-tf516/bin:${PATH}
export HCCL_NPU_SOCKET_PORT_RANGE=59900-59950

torchrun --nnodes=1 --nproc-per-node=16 --node-rank=0 \
  --master-addr=127.0.0.1 --master-port=64490 \
  tasks/train_text.py configs/text/deepseek_v4_npu.yaml \
  --model.config_path=/efs_rl/m00881603/hf_weights/deepseek_v4-4layers_1 \
  --model.model_path=/efs_rl/m00881603/hf_weights/deepseek_v4-4layers_1 \
  --model.tokenizer_path=/efs_rl/m00881603/hf_weights/deepseek_v4-4layers_1 \
  --data.train_path=/efs_rl/m00881603/datasets/dapo-math-17k-text.parquet \
  --data.datasets_type=iterable --data.train_sample=10000 \
  --data.data_type=plaintext --data.text_keys=text \
  --data.max_seq_len=4096 --data.dataloader.shuffle=false --data.dataloader.num_workers=0 \
  --train.dyn_bsz=false --train.global_batch_size=16 --train.micro_batch_size=1 \
  --train.max_steps="${MAX_STEPS:-1000}" --train.seed=42 --train.enable_full_determinism=true \
  --model.optimizer.lr=3e-4 --model.optimizer.lr_warmup_ratio=0 \
  --model.ops_implementation.dsa_indexer_implementation=npu \
  --model.ops_implementation.dsa_attention_implementation=npu \
  --model.ops_implementation.mhc_implementation=npu \
  --model.ops_implementation.mhc_pre_implementation=eager \
  --model.ops_implementation.mhc_post_implementation=npu \
  --train.checkpoint.save_steps=0 --train.checkpoint.save_epochs=0 \
  --train.checkpoint.save_hf_weights=false --train.wandb.enable=false \
  --train.checkpoint.output_dir="${OUTPUT_DIR:-dsv4_tf516_precision_hybrid_1k}" \
  > "${LOG_FILE:-dsv4_242_tf516_precision_hybrid_1000step_20260921.log}" 2>&1
