"""LLaDA-V under DllmTrainer: image-aware template and transform, packed multimodal loss, converter."""

import importlib.util
import json
import os

import pytest
import torch
from PIL import Image
from transformers import SiglipImageProcessor

from tests.models.test_llada import MASK_ID, REPO_ROOT, _CharTokenizer, _fake_ps, _toy_model
from veomni.data.chat_template import build_chat_template
from veomni.data.data_collator import DataCollateInfo, MainCollator
from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY
from veomni.utils.constants import IGNORE_INDEX, IMAGE_INPUT_INDEX


TOY_CONFIG = "./tests/toy_config/llada_v_toy"
# 56 px / 14 px patches = a 4x4 grid, plus one newline per row.
IMAGE_SEQ_LEN = 20
REAL_CKPT = "/mnt/hdfs/user_dir/omni_v2/ckpt/LLaDA-V"


def _processor():
    return SiglipImageProcessor(size={"height": 56, "width": 56}, image_mean=[0.5] * 3, image_std=[0.5] * 3)


def _image(value=128):
    return Image.new("RGB", (80, 60), (value, 0, 255 - value))


def _example(question="What?", answer="A cat.", image_value=128):
    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [_image(image_value)],
    }


def _transform(example, max_seq_len=512, seed=0):
    (out,) = DATA_TRANSFORM_REGISTRY["llada_v"](
        example,
        chat_template=build_chat_template("llada", _CharTokenizer()),
        max_seq_len=max_seq_len,
        mask_token_id=MASK_ID,
        image_processor=_processor(),
        image_seq_len=IMAGE_SEQ_LEN,
        mask_seed=seed,
    )
    return out


def test_config_defaults_to_the_so400m_single_tile():
    from veomni.models.transformers.llada_v.configuration_llada_v import LLaDAVConfig

    config = LLaDAVConfig()
    assert config.image_seq_len == 27 * 28 and config.mm_hidden_size == 1152


def test_template_places_image_slots_after_the_user_header():
    template = build_chat_template("llada", _CharTokenizer())
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What? "}]},
        {"role": "assistant", "content": " A cat."},
    ]
    out = template.encode_multimodal_messages(messages, IMAGE_SEQ_LEN)
    ids, labels = out["input_ids"], out["labels"]
    start = ids.index(IMAGE_INPUT_INDEX)
    assert ids[start : start + IMAGE_SEQ_LEN] == [IMAGE_INPUT_INDEX] * IMAGE_SEQ_LEN
    assert IMAGE_INPUT_INDEX not in ids[start + IMAGE_SEQ_LEN :]
    tokenizer = _CharTokenizer()
    assert tokenizer.decode(ids[:start]) == "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n"
    assert tokenizer.decode(ids[start + IMAGE_SEQ_LEN :]) == (
        "\nWhat?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nA cat.<|eot_id|>"
    )
    supervised = [i for i, label in zip(ids, labels) if label != IGNORE_INDEX]
    assert tokenizer.decode(supervised) == "A cat.<|eot_id|>"


def test_template_matches_the_text_only_encoding_without_images():
    template = build_chat_template("llada", _CharTokenizer())
    messages = [{"role": "user", "content": " hi "}, {"role": "assistant", "content": "yo!"}]
    assert template.encode_multimodal_messages(messages, IMAGE_SEQ_LEN) == template.encode_messages(messages)


def test_template_refuses_an_image_in_the_answer():
    template = build_chat_template("llada", _CharTokenizer())
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "image"}]}]
    with pytest.raises(ValueError, match="text only"):
        template.encode_multimodal_messages(messages, IMAGE_SEQ_LEN)


def test_transform_never_masks_or_supervises_image_slots():
    out = _transform(_example(answer="a long answer " * 4))
    image = out["image_mask"]
    assert int(image.sum()) == IMAGE_SEQ_LEN
    assert bool((out["input_ids"][image] == 0).all())
    assert bool((out["labels"][image] == IGNORE_INDEX).all()) and bool((out["loss_weight"][image] == 0).all())
    assert bool((out["answer_mask"][image] == 0).all())
    assert int(out["answer_mask"].sum()) == len("a long answer " * 4) - 1 + 1  # trimmed answer + <|eot_id|>
    assert out["pixel_values"].shape == (1, 3, 56, 56)
    assert bool((out["input_ids"] != MASK_ID)[~out["answer_mask"].bool()].all())


def test_a_lone_placeholder_moves_to_the_front_as_in_llava():
    first = _transform(_example(question="What?"))
    for content in ("What?\n<image>", "<image>What?", "  What? <image> "):
        example = _example()
        example["messages"][0]["content"] = content
        assert torch.equal(_transform(example)["image_mask"], first["image_mask"]), content
        assert torch.equal(_transform(example)["input_ids"], first["input_ids"]), content


def test_a_missing_checkpoint_parameter_is_refused():
    model = _toy_model("sdpa", config=TOY_CONFIG)
    model._validate_checkpoint_missing_parameters(set())
    with pytest.raises(ValueError, match="image_newline"):
        model._validate_checkpoint_missing_parameters({"model.image_newline"})


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


def test_ulysses_shards_fill_exactly_the_unsharded_slots(monkeypatch):
    import types

    from veomni.models.transformers.llada import dllm_utils

    # two rows over two shards of four: row 0's image straddles the boundary, and row 1's
    # sits wholly in shard 1, so shard 0 holds none of it
    image_mask = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1]], dtype=torch.bool)
    rows = torch.arange(18.0).view(6, 3).requires_grad_(True)
    expected = torch.zeros(2, 8, 3).masked_scatter(image_mask.unsqueeze(-1), rows)

    shards = []
    for rank in range(2):
        state = types.SimpleNamespace(ulysses_enabled=True, ulysses_size=2, ulysses_rank=rank)
        monkeypatch.setattr(dllm_utils, "get_parallel_state", lambda state=state: state)
        slots, local_rows = dllm_utils.ulysses_local_image_slots(image_mask, rows, local_len=4)
        shards.append(torch.zeros(2, 4, 3).masked_scatter(slots.unsqueeze(-1), local_rows))
    torch.testing.assert_close(torch.cat(shards, dim=1), expected)

    # a shard with no slot still reaches the rows in the graph, with a zero gradient
    state = types.SimpleNamespace(ulysses_enabled=True, ulysses_size=2, ulysses_rank=0)
    monkeypatch.setattr(dllm_utils, "get_parallel_state", lambda: state)
    slots, local_rows = dllm_utils.ulysses_local_image_slots(image_mask[1:], rows[4:], local_len=4)
    assert local_rows.shape == (0, 3)
    torch.zeros(1, 4, 3).masked_scatter(slots.unsqueeze(-1), local_rows).sum().backward()
    assert rows.grad is not None and float(rows.grad.abs().sum()) == 0.0

    with pytest.raises(ValueError, match="full padded sequence"):
        dllm_utils.ulysses_local_image_slots(image_mask[:, :4], rows, local_len=4)


def test_the_image_conditions_the_loss(monkeypatch):
    model = _toy_model("sdpa", config=TOY_CONFIG).eval()
    batch = _batch([_transform(_example(answer="a cat on a mat"), seed=1)], monkeypatch)
    with torch.no_grad():
        loss = model(**batch, use_cache=False).loss
        other = model(**dict(batch, pixel_values=-batch["pixel_values"]), use_cache=False).loss
    assert not torch.allclose(loss, other)


def test_backward_reaches_the_vision_tower_with_and_without_images(monkeypatch):
    model = _toy_model("sdpa", config=TOY_CONFIG).train()
    patch = model.model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight
    batch = _batch(_features()[:1], monkeypatch)
    model(**batch, use_cache=False).loss.backward()
    assert patch.grad is not None and float(patch.grad.abs().sum()) > 0
    assert model.model.image_newline.grad is not None
    model.zero_grad(set_to_none=True)
    text_only = _batch(_features()[1:2], monkeypatch)
    model(**text_only, use_cache=False).loss.backward()
    # A rank without images still produces (zero) tower gradients, so FSDP's reductions line up.
    assert patch.grad is not None and float(patch.grad.abs().sum()) == 0.0


def test_placeholder_count_must_match_the_images(monkeypatch):
    model = _toy_model("sdpa", config=TOY_CONFIG).eval()
    batch = _batch(_features()[:1], monkeypatch)
    batch["image_mask"][0, batch["image_mask"][0].nonzero()[0]] = False
    with pytest.raises(ValueError, match="placeholder"):
        model(**batch, use_cache=False)


def _converter():
    spec = importlib.util.spec_from_file_location(
        "convert_lladav_to_veomni", os.path.join(REPO_ROOT, "scripts", "convert_lladav_to_veomni.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_self_test():
    _converter().self_test()


@pytest.mark.skipif(not os.path.exists(REAL_CKPT), reason="LLaDA-V checkpoint not mounted")
def test_model_matches_the_real_checkpoint_layout():
    from safetensors import safe_open

    from veomni.models.transformers.llada_v.configuration_llada_v import LLaDAVConfig
    from veomni.models.transformers.llada_v.modeling_llada_v import LLaDAVForConditionalGeneration

    converter = _converter()
    with open(os.path.join(REAL_CKPT, "config.json")) as f:
        src_cfg = json.load(f)
    with open(os.path.join(REAL_CKPT, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    cfg = converter.convert_config(
        src_cfg,
        converter.count_vision_layers(weight_map),
        converter.resolve_mask_token_id(REAL_CKPT),
        converter.resolve_bos_token_id(REAL_CKPT),
    )
    assert cfg["mask_token_id"] == MASK_ID and cfg["bos_token_id"] < cfg["vocab_size"]
    config = LLaDAVConfig(**{k: v for k, v in cfg.items() if k not in ("model_type", "architectures", "torch_dtype")})
    assert config.image_seq_len == 756
    with torch.device("meta"):
        model = LLaDAVForConditionalGeneration(config)
    params = dict(model.named_parameters())
    assert set(params) == set(weight_map)
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(REAL_CKPT, shard), "pt") as f:
            for key in f.keys():
                assert tuple(f.get_slice(key).get_shape()) == tuple(params[key].shape), key
