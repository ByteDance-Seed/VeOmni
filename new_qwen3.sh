export USE_MOJO_OPSET=0
export USE_LIGER_KERNEL=0
export USE_SEED_KERNELS=0
export NCCL_DEBUG=ERROR

SP_SIZE=2
USE_RM_PAD=true
BS=1

EXP_NAME=trainer_qwen3_sft_SP${SP_SIZE}_rmpad${USE_RM_PAD}_bs${BS}

bash train.sh tasks/train_text.py configs/sft/qwen3_sft.yaml \
    --model.model_path /mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-0.6B-Base\
    --data.train_path /mnt/hdfs/veomni/dataset/tulu-3-sft-mixture/data \
    --data.max_seq_len 2048 \
    --data.source_name sharegpt4v_captioner_sft \
    --train.output_dir /opt/tiger/exp/${EXP_NAME} \
    --train.data_parallel_mode fsdp1 \
    --train.use_wandb true \
    --train.wandb_project qwen3 \
    --train.wandb_name ${EXP_NAME} \
    --train.ulysses_parallel_size ${SP_SIZE} \
    --train.global_batch_size 8 \
    --train.rmpad_with_pos_ids ${USE_RM_PAD} \
    --train.num_train_epochs 1 \
    --train.micro_batch_size ${BS} \
    --train.max_steps 20 \
    --train.profile_trace_dir /opt/tiger/exp/${EXP_NAME}/trace \
    --train.enable_full_determinism true
