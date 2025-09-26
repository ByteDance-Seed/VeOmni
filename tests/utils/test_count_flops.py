import pytest
import torch
from transformers import AutoConfig

from veomni.utils.count_flops import VeomniFlopsCounter


@pytest.fixture(autouse=True)
def patch_get_device(monkeypatch):
    # Force a known device name so get_device_flops() returns a stable value
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args, **kwargs: "A100")
    # Patch the get_device_flops in the same module where VeomniFlopsCounter lives:
    monkeypatch.setattr(
        "veomni.utils.count_flops.get_device_flops",
        lambda unit="T": 312.0,
    )


# Make sure the pure text training does not include ViT flops.
def test_qwen2_vl_no_image():
    config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    config.model_type = "qwen2_vl"
    fc = VeomniFlopsCounter(config)

    batch_seqlens = [128, 64]
    delta_time = 2.0
    est, prom = fc.estimate_flops(batch_seqlens, delta_time)

    assert prom == 312.0
    assert est > 0
    expected = fc._estimate_qwen2_vl_flops(sum(batch_seqlens), batch_seqlens, delta_time)
    assert pytest.approx(expected, rel=1e-6) == est


# Make sure the image ViT flops is counted.
def test_qwen2_vl_with_image():
    config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    config.model_type = "qwen2_vl"
    fc = VeomniFlopsCounter(config)

    batch_seqlens = [64]
    image_seqlens = [14 * 14]  # 196 patches
    delta_time = 1.0

    est, prom = fc.estimate_flops(batch_seqlens, delta_time, image_seqlens=image_seqlens)
    assert prom == 312.0

    base = fc._estimate_qwen2_vl_flops(sum(batch_seqlens), batch_seqlens, delta_time)
    assert est > base

    vit = fc._estimate_qwen_vit_flop(image_seqlens, config.vision_config)
    combined = (base * 1e12 + vit) / 1e12
    assert pytest.approx(combined, rel=1e-6) == est


@pytest.mark.parametrize(
    "total_tokens,image_ratio,delta_time,expected_qwen2,expected_qwen25,execpted_qwen2_vit_ratio,execpted_qwen25_vit_ratio",
    [
        # 32784 total; 90% images → 29505 img, 3279 text; delta=2.0
        # Note that the ViT actually process 29505 x 4 = 118020 img tokens due to the spatial_merge_size=2.
        # Without the window attention optimization, the Qwen2 has a higher flops and ViT ratio.
        (8196 * 4, 0.90, 1.0, 10115.557602361345, 4762.382498463744, 0.72396525438, 0.41368729403),
        # 8192 total; 50% images → 4096/4096; delta=1.0
        (1024 * 8, 0.50, 1.0, 653.290295525376, 626.312532197376, 0.30334109108, 0.27333323047),
        # 4096 total; 25% images → 1024 img, 3072 text; delta=0.5
        (2048 * 2, 0.25, 1.0, 232.160162217984, 247.062356557824, 0.10683760683, 0.16071096798),
    ],
)
def test_estimate_various_ratios_exact(
    total_tokens,
    image_ratio,
    delta_time,
    expected_qwen2,
    expected_qwen25,
    execpted_qwen2_vit_ratio,
    execpted_qwen25_vit_ratio,
):
    # Qwen2‑VL
    cfg2 = AutoConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    # ViT has (spatial_merge_size)^2 times image tokens compared to the LM.
    image_tokens2 = int(total_tokens * image_ratio) * (cfg2.vision_config.spatial_merge_size**2)
    cfg2.model_type = "qwen2_vl"
    fc2 = VeomniFlopsCounter(cfg2)
    est2, prom2 = fc2.estimate_flops(
        [total_tokens],
        delta_time,
        image_seqlens=[image_tokens2],
    )
    raw_vit2 = fc2._estimate_qwen_vit_flop([image_tokens2], cfg2.vision_config) / 1e12 / delta_time
    assert prom2 == 312.0
    assert pytest.approx(expected_qwen2, rel=1e-9) == est2
    assert pytest.approx(execpted_qwen2_vit_ratio, rel=1e-9) == raw_vit2 / est2

    # Qwen2.5‑VL
    cfg25 = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    image_tokens25 = int(total_tokens * image_ratio) * (cfg25.vision_config.spatial_merge_size**2)
    cfg25.model_type = "qwen2_5_vl"
    fc25 = VeomniFlopsCounter(cfg25)
    est25, prom25 = fc25.estimate_flops(
        [total_tokens],
        delta_time,
        image_seqlens=[image_tokens25],
    )
    raw_vit25 = fc25._estimate_qwen_vit_flop([image_tokens25], cfg25.vision_config) / 1e12 / delta_time
    assert prom25 == 312.0
    assert pytest.approx(expected_qwen25, rel=1e-9) == est25
    assert pytest.approx(execpted_qwen25_vit_ratio, rel=1e-9) == raw_vit25 / est25

    # optional sanity: they should differ
    assert est2 != est25
