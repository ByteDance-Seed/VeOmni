source /app/VeOmni/submodules/Open-VeOmni/.venv/bin/activate
export PYTHONPATH=/opt/tiger/projects/Open-VeOmni:${PYTHONPATH:-}
python tasks/omni/infer_omni.py \
    configs/seed_omni/janus_1.3b/veomni_janus.yaml \
    --model.omni_infer_type infer_und \
    --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/Janus-1.3B \
    --infer.prompt "What do you see in this image?" \
    --infer.image /mnt/hdfs/user_dir/veomni_omni/models/transformers/Janus-1.3B/teaser.png \
    --infer.output_dir janus_out \
    --infer.generation_kwargs.max_new_tokens 1024
