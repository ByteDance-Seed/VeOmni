export USE_MOJO_OPSET=0
export USE_LIGER_KERNEL=0
export USE_SEED_KERNELS=0
export NCCL_DEBUG=ERROR

SP_SIZE=2
EXP_NAME=baseline_qwen3_sft_sp_${SP_SIZE}

bash train.sh tasks/train_torch.py configs/sft/qwen3_sft.yaml \
    --model.model_path /mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-0.6B-Base \
    --data.train_path /mnt/hdfs/veomni/dataset/tulu-3-sft-mixture/data \
    --train.enable_full_determinism true \
    --train.num_train_epochs 1 \
    --train.max_steps 20 \
    --train.ulysses_parallel_size ${SP_SIZE} \
    --train.use_wandb true \
    --train.output_dir /opt/tiger/exp/${EXP_NAME} \
    --train.wandb_project qwen3 \
    --train.wandb_name ${EXP_NAME} \
    --train.profile_trace_dir /opt/tiger/exp/${EXP_NAME}/trace
