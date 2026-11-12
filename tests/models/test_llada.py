"""LLaDA under DllmTrainer: masking transform, templates, packed bidirectional loss, converter."""

import importlib.util
import os
import types

import pytest
import torch

from tests.tools.training_utils import make_eager_ops_config
from veomni.data.chat_template import build_chat_template
from veomni.data.data_collator import DataCollateInfo, MainCollator
from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY, apply_dllm_masking
from veomni.models import build_foundation_model
from veomni.models.loss_utils import dllm_masked_cross_entropy
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.device import IS_CUDA_AVAILABLE


TOY_CONFIG = "./tests/toy_config/llada_toy"
MASK_ID = 126336
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class _CharTokenizer:
    """One id per character; special markers get their own ids, as in the real vocabulary."""

    bos_token = "<|startoftext|>"
    chat_template = None
    specials = ["<|startoftext|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|role_end|>"]

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            for index, special in enumerate(self.specials):
                if text.startswith(special):
                    ids.append(1000 + index)
                    text = text[len(special) :]
                    break
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids

    def decode(self, ids):
        return "".join(self.specials[i - 1000] if i >= 1000 else chr(i) for i in ids)


def _fake_ps():
    return types.SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)


def _messages():
    return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo!"}]


def _toy_model(attn_implementation="eager", device="cpu", dtype="float32", config=TOY_CONFIG):
    torch.manual_seed(0)
    return build_foundation_model(
        config,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        init_device=device,
        ops_implementation=make_eager_ops_config(attn_implementation=attn_implementation),
    ).to(device)


def _sample(length, answer, seed):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 1000, (length,), generator=generator)
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[answer] = ids[answer]
    ids = ids.masked_fill(labels != IGNORE_INDEX, MASK_ID)
    weight = torch.zeros(length)
    weight[answer] = 1.0 / (0.5 * 8)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones(length, dtype=torch.long),
        "labels": labels,
        "loss_weight": weight,
    }


def test_llada_template_supervises_answer_and_eot_only():
    template = build_chat_template("llada", _CharTokenizer())
    out = template.encode_messages(_messages())
    text = _CharTokenizer().decode(out["input_ids"])
    assert text == (
        "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nyo!<|eot_id|>"
    )
    supervised = [i for i, label in zip(out["input_ids"], out["labels"]) if label != IGNORE_INDEX]
    assert _CharTokenizer().decode(supervised) == "yo!<|eot_id|>"


def test_llada_moe_template_folds_system_and_supervises_answer():
    template = build_chat_template("llada_moe", _CharTokenizer())
    out = template.encode_messages([{"role": "system", "content": "be brief"}, *_messages()])
    assert _CharTokenizer().decode(out["input_ids"]) == (
        "<role>SYSTEM</role>be brief\ndetailed thinking off<|role_end|>"
        "<role>HUMAN</role>hi<|role_end|><role>ASSISTANT</role>yo!<|role_end|>"
    )
    supervised = [i for i, label in zip(out["input_ids"], out["labels"]) if label != IGNORE_INDEX]
    assert _CharTokenizer().decode(supervised) == "yo!<|role_end|>"


def test_masking_touches_answer_only_with_one_weight_per_sample():
    torch.manual_seed(0)
    input_ids = torch.arange(100)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[40:90] = input_ids[40:90]
    for _ in range(50):
        ids, new_labels, weight = apply_dllm_masking(input_ids, labels, MASK_ID)
        masked = new_labels != IGNORE_INDEX
        assert torch.equal(ids[:40], input_ids[:40]) and torch.equal(ids[90:], input_ids[90:])
        assert bool((ids[masked] == MASK_ID).all()) and torch.equal(new_labels[masked], labels[masked])
        assert bool((weight[~masked] == 0).all())
        if masked.any():
            assert torch.unique(weight[masked]).numel() == 1


def test_masking_weight_is_unbiased():
    # E[sum_i w_i] = E[#masked / (t N)] = 1 per sample, whatever t turns out to be
    torch.manual_seed(0)
    input_ids = torch.arange(64)
    labels = input_ids.clone()
    labels[:16] = IGNORE_INDEX
    totals = [float(apply_dllm_masking(input_ids, labels, MASK_ID)[2].sum()) for _ in range(4000)]
    assert abs(sum(totals) / len(totals) - 1.0) < 0.05


def test_masking_without_answer_yields_no_supervision():
    ids, labels, weight = apply_dllm_masking(torch.arange(5), torch.full((5,), IGNORE_INDEX), MASK_ID)
    assert torch.equal(ids, torch.arange(5)) and bool((labels == IGNORE_INDEX).all()) and weight.sum() == 0


def test_transform_requires_mask_token_id():
    template = build_chat_template("llada", _CharTokenizer())
    with pytest.raises(ValueError, match="mask_token_id"):
        DATA_TRANSFORM_REGISTRY["llada"]({"messages": _messages()}, chat_template=template, max_seq_len=64)
    (out,) = DATA_TRANSFORM_REGISTRY["llada"](
        {"messages": _messages()}, chat_template=template, max_seq_len=64, mask_token_id=MASK_ID
    )
    assert set(out) == {"input_ids", "attention_mask", "labels", "loss_weight", "answer_mask"}
    assert out["loss_weight"].dtype == torch.float32 and out["loss_weight"].shape == out["input_ids"].shape
    # "yo!" plus <|eot_id|>, whether or not the draw masked them
    assert int(out["answer_mask"].sum()) == 4 and bool((out["answer_mask"][out["loss_weight"] > 0] == 1).all())


def test_transform_draws_from_its_own_keyed_generator():
    template = build_chat_template("llada", _CharTokenizer())
    example = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a long answer " * 8}]}

    def draw(seed, epoch):
        (out,) = DATA_TRANSFORM_REGISTRY["llada"](
            example, chat_template=template, max_seq_len=512, mask_token_id=MASK_ID, mask_seed=seed, mask_epoch=epoch
        )
        return out["input_ids"]

    rng_state = torch.get_rng_state()
    first = draw(0, 0)
    assert torch.equal(torch.get_rng_state(), rng_state)
    torch.rand(10)  # a moved process RNG must not change the draw
    assert torch.equal(draw(0, 0), first)
    assert not torch.equal(draw(0, 1), first)
    assert not torch.equal(draw(1, 0), first)


def test_attention_is_bidirectional():
    model = _toy_model().eval()
    ids = torch.randint(0, 1000, (1, 10))
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 1000
    with torch.no_grad():
        first = model(input_ids=ids).logits[0, 0]
        second = model(input_ids=changed).logits[0, 0]
    assert not torch.allclose(first, second)


def test_labels_without_loss_weight_are_refused():
    model = _toy_model()
    ids = torch.randint(0, 1000, (1, 4))
    with pytest.raises(ValueError, match="loss_weight"):
        model(input_ids=ids, labels=ids)


def _packed_vs_separate(monkeypatch, attn_implementation, device, dtype, config=TOY_CONFIG):
    import veomni.data.data_collator as collator_module

    monkeypatch.setattr(collator_module, "get_parallel_state", _fake_ps)
    model = _toy_model(attn_implementation, device, dtype, config).eval()
    features = [_sample(9, slice(4, 8), 1), _sample(7, slice(2, 6), 2), _sample(5, slice(1, 3), 3)]
    collator = MainCollator(data_collate_info={"loss_weight": DataCollateInfo(-1, True, 0, 1)})

    def run(batch):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            return model(**batch, use_cache=False).loss.float()

    separate = sum(run(collator([dict(f)])) for f in features)
    packed_batch = collator([dict(f) for f in features])
    packed = run(packed_batch)
    # one segment for the whole row: samples see each other, so the loss must move
    unsegmented = dict(packed_batch, position_ids=torch.arange(packed_batch["input_ids"].shape[-1]).unsqueeze(0))
    for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
        unsegmented.pop(key)
    leaked = run(unsegmented)
    return separate, packed, leaked


def test_packed_loss_matches_separate_samples_sdpa(monkeypatch):
    separate, packed, leaked = _packed_vs_separate(monkeypatch, "sdpa", "cpu", "float32")
    torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(leaked, separate, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="flash varlen needs CUDA")
def test_packed_loss_matches_separate_samples_flash_varlen(monkeypatch):
    pytest.importorskip("flash_attn")
    separate, packed, _ = _packed_vs_separate(monkeypatch, "veomni_flash_attention_2", "cuda", "bfloat16")
    torch.testing.assert_close(packed, separate, rtol=2e-2, atol=2e-2)


def test_loss_matches_full_logit_reference():
    """Masked-position LM head equals Open-VeOmni's full-logit `dllm_masked_cross_entropy(sum)`."""
    model = _toy_model()
    sample = _sample(12, slice(3, 11), 7)
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}
    loss = model(**batch, use_cache=False).loss
    hidden = model.model(input_ids=batch["input_ids"]).last_hidden_state
    reference = dllm_masked_cross_entropy(
        model.lm_head(hidden), batch["labels"], batch["loss_weight"], reduction="sum"
    )
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_no_masked_token_keeps_graph():
    model = _toy_model()
    ids = torch.randint(0, 1000, (1, 6))
    loss = model(input_ids=ids, labels=torch.full_like(ids, IGNORE_INDEX), loss_weight=torch.zeros(1, 6)).loss
    assert float(loss.detach()) == 0.0
    loss.backward()


def test_converter_self_test():
    spec = importlib.util.spec_from_file_location(
        "convert_llada_to_veomni", os.path.join(REPO_ROOT, "scripts", "convert_llada_to_veomni.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.self_test()
    assert module.convert_key("transformer.blocks.1.attn_out.weight") == "model.layers.1.self_attn.o_proj.weight"
