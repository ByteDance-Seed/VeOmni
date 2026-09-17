source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH="/usr/local/python3.11.15/bin:$PATH"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11

mkdir -p logs
LOG_FILE=logs/gemma3_12b_npu.log

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345
export HCCL_NPU_SOCKET_PORT_RANGE=17000-18000

torchrun --nnodes=1 --nproc-per-node=4 --node-rank=0 --standalone \
    tasks/train_text.py configs/text/gemma3_12b_npu.yaml \
    --model.model_path ../models/gemma-3-12b-it \
    --data.train_path ../data/tulu-first2000.parquet \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta \
    --train.max_steps 60 \
    2>&1 | tee "${LOG_FILE}"
