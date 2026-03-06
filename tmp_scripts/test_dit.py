from veomni.models.auto import build_foundation_model
from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import WanTransformer3DModel


model_path = "/mnt/hdfs/veomni/models/diffusers/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer"
model = build_foundation_model(
    config_path=model_path,
    weights_path=model_path,
    torch_dtype="bfloat16",
    init_device="cuda",
)

model.save_pretrained("wan2.1")

model_path = "wan2.1"
model = WanTransformer3DModel.from_pretrained(model_path, torch_dtype="bfloat16")
model = build_foundation_model(
    config_path=model_path,
    weights_path=model_path,
    torch_dtype="bfloat16",
    init_device="cuda",
)
print(model)
