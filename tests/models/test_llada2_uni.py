"""LLaDA2.0-Uni under DllmTrainer: understand_image template, VQ-id transform, packed loss, converter."""

import importlib.util
import json
import os
import types

import pytest
import torch
from PIL import Image

from veomni.data.chat_template import build_chat_template
from veomni.data.data_collator import DataCollateInfo, MainCollator
from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY
from veomni.utils.constants import IGNORE_INDEX, IMAGE_INPUT_INDEX

from .test_llada import REPO_ROOT, _fake_ps, _toy_model


TOY_CONFIG = "./tests/toy_config/llada2_uni_toy"
MASK_ID = 156895
REAL_CKPT = "/mnt/hdfs/user_dir/omni_v2/ckpt/LLaDA2.0-Uni"
PATCH = 16
# Fake processor + tokenizer without reserved_token_2 round a 2x2 smart-resize up to 8x8.
IMAGE_SEQ_LEN = 64


class _UniTokenizer:
    """Character ids plus the understand_image markers the template looks up by name."""

    bos_token = None
    chat_template = None
    specials = [
        "<role>",
        "</role>",
        "<|endoftext|>",
        "<|image|>",
        "<|/image|>",
        "<boi>",
        *[f"<|reserved_token_{n}|>" for n in (8, 10, 16, 22, 28)],
    ]

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

    def convert_tokens_to_ids(self, token):
        try:
            return 1000 + self.specials.index(token)
        except ValueError:
            return None

    def convert_ids_to_tokens(self, idx):
        if 1000 <= idx < 1000 + len(self.specials):
            return self.specials[idx - 1000]
        return chr(idx) if 0 <= idx < 128 else None


class _FakeProcessor:
    """Packed GlmImage-shaped patches; honors `do_resize` so the round-up path stays testable."""

    patch_size = PATCH
    merge_size = 1
    temporal_patch_size = 1
    size = types.SimpleNamespace(shortest_edge=256, longest_edge=4096)

    def __call__(self, images, return_tensors="pt", do_resize=True):
        del return_tensors
        image = images[0]
        width, height = image.size
        if do_resize:
            height = width = 2 * PATCH
        grid_h, grid_w = height // PATCH, width // PATCH
        return {
            "pixel_values": torch.zeros(grid_h * grid_w, 3 * PATCH * PATCH),
            "image_grid_thw": torch.tensor([[1, grid_h, grid_w]]),
        }


def _image(value=128, size=(64, 64)):
    return Image.new("RGB", size, (value, 0, 255 - value))


def _example(question="What?", answer="A cat.", image_value=128):
    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [_image(image_value)],
    }


def _transform(example, max_seq_len=512, seed=0, tokenizer=None):
    (out,) = DATA_TRANSFORM_REGISTRY["llada2_uni"](
        example,
        chat_template=build_chat_template("llada2_uni", tokenizer or _UniTokenizer()),
        max_seq_len=max_seq_len,
        mask_token_id=MASK_ID,
        image_processor=_FakeProcessor(),
        mask_seed=seed,
    )
    return out


def test_config_defaults_match_the_16b_checkpoint():
    from veomni.models.transformers.llada2_uni.configuration_llada2_uni import LLaDA2UniConfig

    config = LLaDA2UniConfig()
    assert config.rotary_dim == 64 and config.num_experts == 256 and config.first_k_dense_replace == 1
    assert config.image_token_offset + config.vq_config.num_embeddings == config.vocab_size
    assert config.mask_token_id == MASK_ID


def test_template_wraps_vq_slots_in_the_understand_image_header():
    template = build_chat_template("llada2_uni", _UniTokenizer())
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "\nWhat?"}]},
        {"role": "assistant", "content": "A cat."},
    ]
    out = template.encode_multimodal_messages(messages, [(8, 8)])
    ids, labels = out["input_ids"], out["labels"]
    tok = _UniTokenizer()
    start = ids.index(IMAGE_INPUT_INDEX)
    assert ids[start - 4 : start] == [
        tok.convert_tokens_to_ids("<|image|>"),
        tok.convert_tokens_to_ids("<|reserved_token_8|>"),
        tok.convert_tokens_to_ids("<|reserved_token_8|>"),
        tok.convert_tokens_to_ids("<boi>"),
    ]
    assert ids[start : start + 64] == [IMAGE_INPUT_INDEX] * 64
    assert ids[start + 64] == tok.convert_tokens_to_ids("<|/image|>")
    assert tok.decode(ids[: start - 4]).startswith(
        "<role>SYSTEM</role> You are a multimodal understanding assistant. <role>HUMAN</role>"
    )
    supervised = [i for i, label in zip(ids, labels) if label != IGNORE_INDEX]
    assert tok.decode(supervised) == "A cat.<|endoftext|>"


def test_template_matches_the_text_only_encoding_without_images():
    template = build_chat_template("llada2_uni", _UniTokenizer())
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo!"}]
    assert template.encode_multimodal_messages(messages, []) == template.encode_messages(messages)


def test_template_refuses_an_image_in_the_answer():
    template = build_chat_template("llada2_uni", _UniTokenizer())
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "image"}]}]
    with pytest.raises(ValueError, match="text only"):
        template.encode_multimodal_messages(messages, [(8, 8)])


def test_transform_never_masks_or_supervises_image_slots():
    out = _transform(_example(answer="a long answer " * 4))
    image = out["image_mask"]
    assert int(image.sum()) == IMAGE_SEQ_LEN
    assert out["image_grid_thw"].tolist() == [[1, 8, 8]]
    assert bool((out["input_ids"][image] == 0).all())
    assert bool((out["labels"][image] == IGNORE_INDEX).all()) and bool((out["loss_weight"][image] == 0).all())
    assert bool((out["answer_mask"][image] == 0).all())
    assert out["pixel_values"].shape[0] == IMAGE_SEQ_LEN
    assert bool((out["input_ids"] != MASK_ID)[~out["answer_mask"].bool()].all())


def test_a_lone_placeholder_moves_to_the_front_as_in_llava():
    first = _transform(_example(question="What?"))
    for content in ("What?\n<image>", "<image>What?", "  What? <image> "):
        example = _example()
        example["messages"][0]["content"] = content
        assert torch.equal(_transform(example)["image_mask"], first["image_mask"]), content
        assert torch.equal(_transform(example)["input_ids"], first["input_ids"]), content


def test_unspellable_grid_is_rounded_to_the_next_reserved_token():
    out = _transform(_example())
    assert out["image_grid_thw"].tolist() == [[1, 8, 8]]


def test_a_missing_checkpoint_parameter_is_refused():
    model = _toy_model("sdpa", config=TOY_CONFIG)
    model._validate_checkpoint_missing_parameters(set())
    with pytest.raises(ValueError, match="missing"):
        model._validate_checkpoint_missing_parameters({"model.word_embeddings.weight"})


def test_transform_checks_images_against_placeholders():
    example = _example()
    example["images"] = [_image(), _image()]
    with pytest.raises(ValueError, match="image part"):
        _transform(example)


def test_text_only_sample_has_no_pixels():
    out = _transform({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo!"}]})
    assert "pixel_values" not in out and not bool(out["image_mask"].any())


def test_over_long_sample_is_cut_from_the_answer_side():
    full = _transform(_example(answer="x" * 40))
    cut = _transform(_example(answer="x" * 40), max_seq_len=full["input_ids"].numel() - 10)
    assert int(cut["image_mask"].sum()) == IMAGE_SEQ_LEN
    assert torch.equal(cut["image_mask"], full["image_mask"][: cut["image_mask"].numel()])


def test_sample_whose_image_does_not_fit_is_left_unsupervised():
    out = _transform(_example(), max_seq_len=IMAGE_SEQ_LEN)
    assert out["input_ids"].numel() == 1 and "pixel_values" not in out
    assert int(out["answer_mask"].sum()) == 0 and float(out["loss_weight"].sum()) == 0.0


def test_vision_and_vq_are_frozen():
    model = _toy_model("sdpa", config=TOY_CONFIG)
    assert not any(p.requires_grad for p in model.model.visual.parameters())
    assert not any(p.requires_grad for p in model.model.vqmodel.parameters())
    assert model.lm_head.weight.requires_grad
    model.train()
    assert model.training and not model.model.visual.training and not model.model.vqmodel.training
    assert not any(
        name.startswith(("model.visual.", "model.vqmodel.")) for name, p in model.named_parameters() if p.requires_grad
    )


def _batch(features, monkeypatch):
    import veomni.data.data_collator as collator_module

    monkeypatch.setattr(collator_module, "get_parallel_state", _fake_ps)
    collator = MainCollator(
        data_collate_info={
            "loss_weight": DataCollateInfo(-1, True, 0, 1),
            "answer_mask": DataCollateInfo(-1, True, 0, 1),
        },
        seq_classification=True,
    )
    batch = collator([dict(f) for f in features])
    batch.pop("answer_mask")
    return batch


def _features():
    return [
        _transform(_example(answer="a cat on a mat", image_value=10), seed=1),
        _transform({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo!"}]}, seed=2),
        _transform(_example(question="Who?", answer="a dog", image_value=200), seed=3),
    ]


def test_packed_multimodal_loss_matches_separate_samples(monkeypatch):
    model = _toy_model("sdpa", config=TOY_CONFIG).eval()
    features = _features()
    with torch.no_grad():
        separate = sum(model(**_batch([f], monkeypatch), use_cache=False).loss for f in features)
        packed = model(**_batch(features, monkeypatch), use_cache=False).loss
    torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-5)


def test_the_image_tokens_condition_the_loss(monkeypatch):
    model = _toy_model("sdpa", config=TOY_CONFIG).eval()
    batch = _batch([_transform(_example(answer="a cat on a mat"), seed=1)], monkeypatch)
    with torch.no_grad():
        loss = model(**batch, use_cache=False).loss
        other = model(**dict(batch, pixel_values=1 - batch["pixel_values"]), use_cache=False).loss
    assert not torch.allclose(loss, other)


def test_backward_runs_with_and_without_images(monkeypatch):
    model = _toy_model("sdpa", config=TOY_CONFIG).train()
    batch = _batch(_features()[:1], monkeypatch)
    model(**batch, use_cache=False).loss.backward()
    assert model.lm_head.weight.grad is not None and float(model.lm_head.weight.grad.abs().sum()) > 0
    assert all(p.grad is None for p in model.model.visual.parameters())
    model.zero_grad(set_to_none=True)
    text_only = _batch(_features()[1:2], monkeypatch)
    model(**text_only, use_cache=False).loss.backward()
    assert model.lm_head.weight.grad is not None


def test_placeholder_count_must_match_the_images(monkeypatch):
    model = _toy_model("sdpa", config=TOY_CONFIG).eval()
    batch = _batch(_features()[:1], monkeypatch)
    batch["image_mask"][0, batch["image_mask"][0].nonzero()[0]] = False
    with pytest.raises(ValueError, match="image slots"):
        model(**batch, use_cache=False)


def _converter():
    spec = importlib.util.spec_from_file_location(
        "convert_llada2uni_to_veomni", os.path.join(REPO_ROOT, "scripts", "convert_llada2uni_to_veomni.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_self_test():
    _converter().self_test()


@pytest.mark.skipif(not os.path.exists(REAL_CKPT), reason="LLaDA2.0-Uni checkpoint not mounted")
def test_model_matches_the_real_checkpoint_layout():
    from safetensors import safe_open

    from veomni.models.transformers.llada2_uni.configuration_llada2_uni import LLaDA2UniConfig
    from veomni.models.transformers.llada2_uni.modeling_llada2_uni import LLaDA2UniForConditionalGeneration
    from veomni.models.transformers.llada_moe.checkpoint_tensor_converter import convert_llada_moe_fqn_to_index_mapping

    converter = _converter()
    with open(os.path.join(REAL_CKPT, "config.json")) as f:
        src_cfg = json.load(f)
    with open(os.path.join(REAL_CKPT, "image_tokenizer", "config.json")) as f:
        image_cfg = json.load(f)
    with open(os.path.join(REAL_CKPT, "model.safetensors.index.json")) as f:
        backbone = json.load(f)
    with safe_open(os.path.join(REAL_CKPT, "image_tokenizer", "model.safetensors"), "pt") as f:
        image_keys = list(f.keys())
    cfg = converter.convert_config(src_cfg, image_cfg, converter.resolve_mask_token_id(REAL_CKPT))
    index = converter.merge_index(backbone, image_keys)["weight_map"]
    skip = {"model_type", "architectures", "torch_dtype", "transformers_version"}
    config = LLaDA2UniConfig(**{k: v for k, v in cfg.items() if k not in skip})
    with torch.device("meta"):
        model = LLaDA2UniForConditionalGeneration(config)
    mapped = convert_llada_moe_fqn_to_index_mapping(index)
    assert set(mapped) == set(model.state_dict())
    params = dict(model.named_parameters()) | dict(model.named_buffers())
    with safe_open(os.path.join(REAL_CKPT, "image_tokenizer", "model.safetensors"), "pt") as f:
        for key in ("model.visual.patch_embed.proj.weight", "model.vqmodel.quantize.embedding.weight"):
            assert tuple(f.get_slice(key).get_shape()) == tuple(params[key].shape), key
    assert params["model.layers.0.mlp.gate_proj.weight"].shape[0] == config.intermediate_size
    assert params["model.layers.1.mlp.experts.gate_up_proj"].shape[0] == config.num_experts
