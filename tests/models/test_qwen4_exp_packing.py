import copy
import importlib

import torch
from transformers import Qwen4ExpConfig

from veomni.ops import apply_ops_config
from veomni.utils.device import IS_NPU_AVAILABLE

from ..tools.training_utils import make_eager_ops_config


def _load_text_model_class():
    backend = "npu" if IS_NPU_AVAILABLE else "gpu"
    module = importlib.import_module(
        f"veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}"
    )
    return module.Qwen4ExpTextModel


def _forward_varlen(model, input_ids: torch.Tensor, packed_seq_lens: tuple[int, ...]) -> torch.Tensor:
    text_positions = torch.cat([torch.arange(length) for length in packed_seq_lens]).view(1, -1)
    position_ids = text_positions.unsqueeze(0).expand(4, -1, -1)
    cu_seq_lens_q = torch.tensor(
        [0, *torch.tensor(packed_seq_lens).cumsum(dim=0).tolist()],
        dtype=torch.int32,
        device=input_ids.device,
    )
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        use_cache=False,
        cu_seq_lens_q=cu_seq_lens_q,
    ).last_hidden_state


def test_qwen4_exp_packed_matches_separate_outputs_and_gradients():
    """QSA, GDN, and PLE must reset state at every explicit packed boundary."""
    apply_ops_config(make_eager_ops_config())
    torch.manual_seed(0)

    config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json").text_config
    config._attn_implementation = "eager"
    config.use_cache = False
    text_model_cls = _load_text_model_class()
    packed_model = text_model_cls(config).float().eval()
    separate_model = copy.deepcopy(packed_model)

    packed_seq_lens = (5, 7)
    input_ids = torch.randint(0, config.vocab_size, (1, sum(packed_seq_lens)))
    packed_output = _forward_varlen(packed_model, input_ids, packed_seq_lens)
    separate_output = torch.cat(
        [
            _forward_varlen(separate_model, segment, (segment.shape[1],))
            for segment in input_ids.split(packed_seq_lens, dim=1)
        ],
        dim=1,
    )

    torch.testing.assert_close(packed_output, separate_output, rtol=1e-5, atol=1e-6)

    loss_weights = torch.randn_like(packed_output)
    (packed_output * loss_weights).sum().backward()
    (separate_output * loss_weights).sum().backward()
    packed_grads = {name: parameter.grad for name, parameter in packed_model.named_parameters()}
    separate_grads = {name: parameter.grad for name, parameter in separate_model.named_parameters()}
    assert packed_grads.keys() == separate_grads.keys()
    for name in packed_grads:
        assert (packed_grads[name] is None) == (separate_grads[name] is None), name
        if packed_grads[name] is None:
            continue
        torch.testing.assert_close(packed_grads[name], separate_grads[name], rtol=1e-4, atol=1e-5)


def test_qwen4_exp_text_model_accepts_explicit_packing_metadata():
    apply_ops_config(make_eager_ops_config())
    config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json").text_config
    config._attn_implementation = "eager"
    config.use_cache = False
    model = _load_text_model_class()(config).float().eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    position_ids = torch.arange(4).view(1, 1, -1).expand(4, -1, -1)

    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        use_cache=False,
        cu_seq_lens_q=torch.tensor([0, input_ids.shape[1]], dtype=torch.int32, device=input_ids.device),
    )
    assert output.last_hidden_state.shape[:2] == input_ids.shape
