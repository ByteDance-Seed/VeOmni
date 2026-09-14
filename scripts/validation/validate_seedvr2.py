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
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
print(
    json.dumps(
        {
            "loss": loss.item(),
            "checkpoint_loss": checkpoint_loss.item(),
            "max_gradient_difference": max_gradient_difference,
            "finite_gradients": True,
            "optimizer_update": True,
            "hf_save_load_exact": True,
            "training_resume_tested": False,
        },
        indent=2,
    )
)
