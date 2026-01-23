import pytest
import torch

from veomni.data.data_collator import DataCollatorWithPositionIDs, DataCollatorWithPositionIDsAndPadding
from veomni.models import build_foundation_model
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def _skip_if_no_flash_attn():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for flash-attn.")
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except Exception as exc:
        pytest.skip(f"flash-attn is not available: {exc}")


@pytest.mark.parametrize("pad_to_length", [16])
def test_qwen3_loss_match_with_padded_packed_input(monkeypatch, pad_to_length):
    _skip_if_no_flash_attn()
    monkeypatch.setattr(
        "veomni.data.data_collator.get_parallel_state",
        lambda: type("PS", (), {"sp_enabled": False, "sp_size": 1, "sp_rank": 0})(),
    )

    device = torch.device(get_device_type())
    torch.manual_seed(0)

    model = build_foundation_model(
        config_path="./tests/models/toy_config/qwen3_toy.json",
        weights_path=None,
        torch_dtype="float16",
        attn_implementation="veomni_flash_attention_2_with_sp",
        init_device=get_device_type(),
    ).to(device)
    model.eval()

    features = [
        {
            "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([2], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([21, 22], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "labels": torch.tensor([1], dtype=torch.long),
        },
    ]

    base_collator = DataCollatorWithPositionIDs(mask_boundary_labels=False)
    padded_collator = DataCollatorWithPositionIDsAndPadding(
        pad_to_length=pad_to_length,
        position_id_pad_value=0,
        attention_mask_pad_value=1,
    )

    batch_unpadded = base_collator(features)
    batch_padded = padded_collator(features)

    def to_device(batch):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    batch_unpadded = to_device(batch_unpadded)
    batch_padded = to_device(batch_padded)

    with torch.no_grad():
        out_unpadded = model(
            input_ids=batch_unpadded["input_ids"],
            attention_mask=batch_unpadded.get("attention_mask"),
            position_ids=batch_unpadded.get("position_ids"),
            cu_seq_lens_q=batch_unpadded.get("cu_seq_lens_q"),
            cu_seq_lens_k=batch_unpadded.get("cu_seq_lens_k"),
            max_length_q=batch_unpadded.get("max_length_q"),
            max_length_k=batch_unpadded.get("max_length_k"),
            labels=batch_unpadded.get("labels"),
        )
        out_padded = model(
            input_ids=batch_padded["input_ids"],
            attention_mask=batch_padded.get("attention_mask"),
            position_ids=batch_padded.get("position_ids"),
            cu_seq_lens_q=batch_padded.get("cu_seq_lens_q"),
            cu_seq_lens_k=batch_padded.get("cu_seq_lens_k"),
            max_length_q=batch_padded.get("max_length_q"),
            max_length_k=batch_padded.get("max_length_k"),
            labels=batch_padded.get("labels"),
        )

    torch.testing.assert_close(
        out_padded.loss,
        out_unpadded.loss,
        rtol=1e-3,
        atol=1e-3,
    )
