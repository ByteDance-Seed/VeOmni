import copy
import json
import tempfile

import torch

from veomni.models.diffusers.seedvr2.conditioning_seedvr2 import SeedVR2ConditionConfig, SeedVR2ConditionModel
from veomni.models.diffusers.seedvr2.configuration_seedvr2 import SeedVR2Config
from veomni.models.diffusers.seedvr2.modeling_seedvr2 import SeedVR2Model


torch.set_num_threads(4)
torch.manual_seed(1030)
config = SeedVR2Config.from_pretrained("tests/toy_config/seedvr2_toy")
model = SeedVR2Model(config).train()
condition = SeedVR2ConditionModel(SeedVR2ConditionConfig(), meta_init=True)
batch = condition.process_condition(
    latents=[torch.randn(3, 8, 8, 4), torch.randn(1, 4, 4, 4)],
    condition_latents=[torch.randn(3, 8, 8, 4), torch.randn(1, 4, 4, 4)],
    context=[torch.randn(3, 24), torch.randn(5, 24)],
)
other = copy.deepcopy(model)
other.gradient_checkpointing_enable()
loss = model(**batch).loss["mse_loss"]
loss.backward()
checkpoint_loss = other(**batch).loss["mse_loss"]
checkpoint_loss.backward()
torch.testing.assert_close(loss, checkpoint_loss, atol=1e-6, rtol=1e-5)
max_gradient_difference = 0.0
for (name, param), (other_name, other_param) in zip(model.named_parameters(), other.named_parameters()):
    assert name == other_name
    assert param.grad is not None and torch.isfinite(param.grad).all(), name
    torch.testing.assert_close(param.grad, other_param.grad, atol=1e-6, rtol=1e-5)
    max_gradient_difference = max(max_gradient_difference, (param.grad - other_param.grad).abs().max().item())
before = model.dit.vid_in.proj.weight.detach().clone()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.step()
assert not torch.equal(before, model.dit.vid_in.proj.weight)
model.eval()
expected = model(**batch).predictions
with tempfile.TemporaryDirectory() as tmp:
    model.save_pretrained(tmp)
    restored = SeedVR2Model.from_pretrained(tmp).eval()
    actual = restored(**batch).predictions
    # Reloading must preserve every tensor bit for bit. The forward outputs of the reloaded
    # model are only as reproducible as the accelerator kernels allow, so they are compared
    # with a bounded tolerance instead of demanding bit equality from the whole graph.
    saved_state = model.state_dict()
    restored_state = restored.state_dict()
    assert saved_state.keys() == restored_state.keys()
    for key, value in saved_state.items():
        torch.testing.assert_close(restored_state[key], value, atol=0, rtol=0)
    max_output_difference = 0.0
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
        max_output_difference = max(max_output_difference, (a - b).abs().max().item())
print(
    json.dumps(
        {
            "loss": loss.item(),
            "checkpoint_loss": checkpoint_loss.item(),
            "max_gradient_difference": max_gradient_difference,
            "finite_gradients": True,
            "optimizer_update": True,
            "hf_save_load_weights_exact": True,
            "hf_save_load_max_output_difference": max_output_difference,
            "hf_save_load_atol": 1e-5,
            "hf_save_load_rtol": 1e-4,
            "training_resume_tested": False,
        },
        indent=2,
    )
)
