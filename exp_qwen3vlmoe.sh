export USE_MOJO_OPSET=0
export USE_LIGER_KERNEL=0
export USE_SEED_KERNELS=0
export NCCL_DEBUG=ERROR

EP_SIZE=2
SP_SIZE=2
USE_RM_PAD=false
moe_imple=fused
BS=2

EXP_NAME=patch_qwen3_vl_moe_IMPLE${moe_imple}_EP${EP_SIZE}_SP${SP_SIZE}_rmpad${USE_RM_PAD}_bs${BS}

bash train.sh tasks/vlm/train_qwen3_vl_moe.py configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml \
    --model.model_path /mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-VL-30B-A3B-Instruct\
    --model.moe_implementation ${moe_imple} \
    --data.train_path /mnt/hdfs/veomni/dataset/sharegpt4v_cap_100k \
    --data.max_seq_len 2048 \
    --data.source_name sharegpt4v_captioner_sft \
    --train.output_dir /opt/tiger/exp/${EXP_NAME} \
    --train.data_parallel_mode fsdp1 \
    --train.use_wandb true \
    --train.wandb_name ${EXP_NAME} \
    --train.expert_parallel_size ${EP_SIZE} \
    --train.ulysses_parallel_size ${SP_SIZE} \
    --train.global_batch_size 8 \
    --train.rmpad_with_pos_ids ${USE_RM_PAD} \
    --train.num_train_epochs 1 \
    --train.micro_batch_size ${BS} \
    --train.max_steps 20 \
    --train.profile_trace_dir /opt/tiger/exp/${EXP_NAME}/trace
