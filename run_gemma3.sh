source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH="/usr/local/python3.11.15/bin:$PATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export HCCL_NPU_SOCKET_PORT_RANGE=17000-18000

mkdir -p logs

bash train.sh tasks/train_text.py configs/text/gemma3_12b_npu.yaml \
    --model.model_path ../models/gemma-3-12b-it \
    --data.train_path ../data/tulu-first2000.parquet \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta \
    2>&1 | tee logs/gemma3_npu.log
