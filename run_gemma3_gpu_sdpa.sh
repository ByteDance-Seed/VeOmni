export CUDA_VISIBLE_DEVICES=0,1

mkdir -p logs

bash train.sh tasks/train_text.py configs/text/gemma3_gpu_sdpa.yaml \
    --model.model_path ../models/gemma-3-270m \
    --data.train_path ../data/tulu-first2000.parquet \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta \
    2>&1 | tee logs/gemma3_gpu_sdpa.log
