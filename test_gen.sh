source /app/VeOmni/submodules/Open-VeOmni/.venv/bin/activate
export PYTHONPATH=/opt/tiger/projects/Open-VeOmni:${PYTHONPATH:-}
python tasks/omni/infer_omni.py \
    configs/seed_omni/janus_1.3b/veomni_janus.yaml \
    --model.omni_infer_type infer_gen \
    --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/Janus-1.3B \
    --infer.prompt "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue." \
    --infer.output_dir janus_out \
    --infer.generation_kwargs.max_new_tokens 2048 \
    --infer.generation_kwargs.guidance_scale 5.0
