from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset

from veomni.arguments import (
    AcceleratorConfig,
    CheckpointConfig,
    DataloaderConfig,
    FSDPConfig,
    GradientCheckpointingConfig,
    MixedPrecisionConfig,
    OptimizerConfig,
)
from veomni.data.multimodal.multimodal_chat_template import MiniMaxM3VLChatTemplate
from veomni.models import build_foundation_model
from veomni.models.loader import get_model_config
from veomni.trainer.base import BaseTrainer
from veomni.trainer.vlm_trainer import (
    VeOmniVLMArguments,
    VLMMDataArguments,
    VLMMModelArguments,
    VLMTrainer,
    VLMTrainingArguments,
    _get_vlm_visual_module,
)
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to
from veomni.utils.loss_utils import count_loss_token

from ..tools.launch_utils import find_free_port
from ..tools.training_utils import make_eager_ops_config


_FREEZE_VIT_VLM_CASES = [
    pytest.param("./tests/toy_config/qwen2vl_toy/config.json", id="qwen2_vl"),
    pytest.param("./tests/toy_config/qwen3_5_toy/config.json", id="qwen3_5"),
    pytest.param("./tests/toy_config/qwen3_5_moe_toy/config.json", id="qwen3_5_moe"),
    pytest.param("./tests/toy_config/qwen25vl_toy/config.json", id="qwen2_5_vl"),
    pytest.param("./tests/toy_config/qwen3vl_toy/config.json", id="qwen3_vl"),
    pytest.param("./tests/toy_config/qwen3vlmoe_toy/config.json", id="qwen3_vl_moe"),
    pytest.param(
        "./tests/toy_config/minimax_m3_vl_toy/config.json",
        marks=pytest.mark.skipif(
            not is_transformers_version_greater_or_equal_to("5.12.0"),
            reason="MiniMax M3 VL modeling is generated from transformers>=5.12.0",
        ),
        id="minimax_m3_vl",
    ),
]


class _MiniMaxTrainerTokenizer:
    eos_token = "<eos>"
    eos_token_id = 200027

    _token_ids = {
        MiniMaxM3VLChatTemplate.IMAGE_TOKEN: 200025,
        MiniMaxM3VLChatTemplate.VIDEO_TOKEN: 200026,
        "<eos>": eos_token_id,
    }

    def convert_tokens_to_ids(self, token):
        return self._token_ids[token]

    def encode(self, text, add_special_tokens=False):
        token_ids = []
        idx = 0
        special_tokens = (
            MiniMaxM3VLChatTemplate.IMAGE_TOKEN,
            MiniMaxM3VLChatTemplate.VIDEO_TOKEN,
            "<eos>",
        )
        while idx < len(text):
            for token in special_tokens:
                if text.startswith(token, idx):
                    token_ids.append(self._token_ids[token])
                    idx += len(token)
                    break
            else:
                token_ids.append(1000 + ord(text[idx]) % 1000)
                idx += 1
        return token_ids


class _MiniMaxTrainerProcessor:
    image_token_id = 200025
    video_token_id = 200026
    IMAGE_TOKEN = MiniMaxM3VLChatTemplate.IMAGE_TOKEN
    VIDEO_TOKEN = MiniMaxM3VLChatTemplate.VIDEO_TOKEN
    VISION_START_TOKEN = MiniMaxM3VLChatTemplate.VISION_START_TOKEN
    VISION_END_TOKEN = MiniMaxM3VLChatTemplate.VISION_END_TOKEN

    def __init__(self):
        from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor
        from transformers.models.minimax_m3_vl.video_processing_minimax_m3_vl import MiniMaxM3VLVideoProcessor

        self.tokenizer = _MiniMaxTrainerTokenizer()
        self.image_processor = MiniMaxM3VLImageProcessor()
        self.video_processor = MiniMaxM3VLVideoProcessor()

    def replace_image_token(self, image_inputs, image_idx):
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = int(image_inputs["image_grid_thw"][image_idx].prod() // merge_length)
        return self.VISION_START_TOKEN + self.IMAGE_TOKEN * num_image_tokens + self.VISION_END_TOKEN

    def replace_video_token(self, video_inputs, video_idx):
        merge_length = self.video_processor.merge_size**2
        grid_thw = video_inputs["video_grid_thw"][video_idx]
        grid_t = int(grid_thw[0])
        frame_seqlen = int(grid_thw[1:].prod() // merge_length)
        return "".join(
            self.VISION_START_TOKEN + self.VIDEO_TOKEN * frame_seqlen + self.VISION_END_TOKEN for _ in range(grid_t)
        )


class _MiniMaxTrainerInitDataset(Dataset):
    def __init__(self, data_transform, media_case: str, image_input=None, video_input=None):
        self.data_transform = data_transform
        self.media_case = media_case
        self.image_input = image_input
        self.video_input = video_input

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        sample = {
            "source_name": f"minimax_trainer_init_{self.media_case}",
            "conversations": [{"from": "human", "value": "Describe the media."}],
        }
        if self.media_case in ("image", "mixed"):
            sample["images"] = [self.image_input]
        if self.media_case in ("video", "mixed"):
            sample["videos"] = [self.video_input]
        return self.data_transform(sample)


class _MiniMaxTrainerInitSmoke(VLMTrainer):
    def _build_model_assets(self):
        self.base.processor = _MiniMaxTrainerProcessor()
        self.base.chat_template = MiniMaxM3VLChatTemplate(self.base.processor.tokenizer)
        self.base.model_assets = [self.base.processor, self.base.chat_template]


class _CpuDeviceModule:
    def set_device(self, device):
        self.device = torch.device(device)

    def synchronize(self):
        pass

    def current_device(self):
        return 0

    def empty_cache(self):
        pass


@pytest.mark.parametrize(
    "freeze_vit",
    [
        pytest.param(False, id="freeze_vit_disabled"),
        pytest.param(True, id="freeze_vit_enabled"),
    ],
)
@pytest.mark.parametrize("config_path", _FREEZE_VIT_VLM_CASES)
def test_freeze_vit_on_vlm_model(config_path, freeze_vit):
    # This test only constructs the model on `meta` and verifies freeze
    # behaviour — it never runs forward. Use an all-eager ops config so the
    # build works everywhere: it pins every per-op field (including the
    # Qwen3.5 GatedDeltaNet trio that has no FLA backend on NPU and the
    # GPU-only liger/triton defaults that fail NPU validation). Eager paths
    # that raise only at forward time are fine because this test never
    # forwards.
    ops_implementation = make_eager_ops_config()
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=ops_implementation,
    )
    visual = _get_vlm_visual_module(model)
    assert visual is not None

    args = VeOmniVLMArguments(
        model=VLMMModelArguments(
            config_path=config_path,
            ops_implementation=make_eager_ops_config(),
        ),
        data=VLMMDataArguments(train_path="dummy"),
    )
    args.train.freeze_vit = freeze_vit

    trainer = VLMTrainer.__new__(VLMTrainer)
    trainer.base = SimpleNamespace(
        args=args,
        model=model,
        model_config=model.config,
    )

    trainer._freeze_model_module()

    if freeze_vit:
        assert all(not param.requires_grad for param in visual.parameters())
    else:
        assert all(param.requires_grad for param in visual.parameters())


@pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="MiniMax M3 VL modeling is generated from transformers>=5.12.0",
)
def test_minimax_m3_vl_vlm_trainer_transform_collate_forward_backward(monkeypatch):
    """Exercise MiniMax through VLMTrainer transform/collate glue and BaseTrainer backward."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(find_free_port()))

    initialized_dist = False
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
        initialized_dist = True

    try:

        def fake_conv_preprocess(source, conversations, **kwargs):
            assert source == "minimax_trainer_test"
            return [
                ["user", ("image", None), ("video", None), ("text", "Describe both.")],
                ["assistant", ("text", "Done.")],
            ]

        def fake_fetch_images(images, **kwargs):
            assert images == ["image://one"]
            return [Image.new("RGB", (28, 28), color=(127, 64, 32))]

        def fake_fetch_videos_metadata(videos, **kwargs):
            assert videos == ["video://one"]
            frames = [Image.new("RGB", (28, 28), color=(idx * 20, 64, 32)) for idx in range(2)]
            return [frames], [{"fps": 1.0, "total_num_frames": len(frames)}], None, None

        monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", fake_conv_preprocess)
        monkeypatch.setattr("veomni.data.multimodal.image_utils.fetch_images", fake_fetch_images)
        monkeypatch.setattr("veomni.data.multimodal.video_utils.fetch_videos_metadata", fake_fetch_videos_metadata)

        torch.manual_seed(20260618)
        config = get_model_config("./tests/toy_config/minimax_m3_vl_toy/config.json")
        config.text_config.vocab_size = max(
            config.text_config.vocab_size,
            config.image_token_id + 8,
            config.video_token_id + 8,
            _MiniMaxTrainerTokenizer.eos_token_id + 8,
        )
        model = build_foundation_model(
            config_path=config,
            weights_path=None,
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=make_eager_ops_config(),
        )
        model.train()

        processor = _MiniMaxTrainerProcessor()

        args = VeOmniVLMArguments(
            model=VLMMModelArguments(
                config_path="./tests/toy_config/minimax_m3_vl_toy/config.json",
                ops_implementation=make_eager_ops_config(),
            ),
            data=VLMMDataArguments(
                train_path="dummy",
                chat_template="minimax_m3_vl",
            ),
        )

        trainer = VLMTrainer.__new__(VLMTrainer)
        trainer.base = BaseTrainer.__new__(BaseTrainer)
        trainer.base.args = args
        trainer.base.model = model
        trainer.base.model_config = model.config
        trainer.base.processor = processor
        trainer.base.chat_template = MiniMaxM3VLChatTemplate(processor.tokenizer)
        trainer.base.model_assets = [processor, trainer.base.chat_template]
        trainer.base.device = torch.device("cpu")
        trainer.base.model_fwd_context = nullcontext()
        trainer.base.model_bwd_context = nullcontext()
        trainer.base.LOG_SAMPLE = False

        trainer._build_data_transform()
        trainer._build_collate_fn()

        features = trainer.base.data_transform(
            {
                "source_name": "minimax_trainer_test",
                "conversations": [{"from": "human", "value": "<image><video>Describe both."}],
                "images": ["image://one"],
                "videos": ["video://one"],
            }
        )
        micro_batch = trainer.base.collate_fn(features)

        image_token_count = int(micro_batch["image_grid_thw"][0].prod() // processor.image_processor.merge_size**2)
        video_token_count = int(micro_batch["video_grid_thw"][0].prod() // processor.video_processor.merge_size**2)
        assert (
            micro_batch["input_ids"][micro_batch["image_mask"]].tolist() == [config.image_token_id] * image_token_count
        )
        assert (
            micro_batch["input_ids"][micro_batch["video_mask"]].tolist() == [config.video_token_id] * video_token_count
        )
        assert micro_batch["multimodal_metadata"] == {
            "image_grid_thw_list": micro_batch["image_grid_thw"].tolist(),
            "video_grid_thw_list": micro_batch["video_grid_thw"].tolist(),
        }

        trainer.base.micro_batch_token_len = count_loss_token(micro_batch)
        trainer.base.micro_batches_token_len = count_loss_token([micro_batch])
        loss, loss_dict = trainer.base.forward_backward_step(micro_batch)

        assert torch.isfinite(loss)
        assert set(loss_dict) == {"foundation_loss"}
        assert model.model.vision_tower.embeddings.proj.weight.grad.abs().sum() > 0
        assert model.model.multi_modal_projector.merge_linear_2.weight.grad.abs().sum() > 0
        assert model.lm_head.weight.grad.abs().sum() > 0
    finally:
        if initialized_dist:
            dist.destroy_process_group()


@pytest.mark.parametrize("media_case", ["image", "video", "mixed"])
@pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="MiniMax M3 VL modeling is generated from transformers>=5.12.0",
)
def test_minimax_m3_vl_vlm_trainer_init_dataloader_optimizer_smoke(monkeypatch, tmp_path, media_case):
    """Exercise MiniMax through VLMTrainer.__init__, dataloader, backward, optimizer, and scheduler."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(find_free_port()))

    import veomni.distributed.parallel_state as parallel_state
    import veomni.trainer.base as base_trainer
    import veomni.trainer.vlm_trainer as vlm_trainer

    cpu_device = _CpuDeviceModule()
    monkeypatch.setattr(base_trainer, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(base_trainer, "get_dist_comm_backend", lambda: "gloo")
    monkeypatch.setattr(base_trainer, "get_torch_device", lambda: cpu_device)
    monkeypatch.setattr(parallel_state, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(vlm_trainer, "synchronize", lambda: None)

    def build_noop_parallelized_model(base):
        base.model.train()

    monkeypatch.setattr(base_trainer.BaseTrainer, "_build_parallelized_model", build_noop_parallelized_model)

    def fake_conv_preprocess(source, conversations, **kwargs):
        if "mixed" in source:
            media = [("image", None), ("video", None)]
        elif "image" in source:
            media = [("image", None)]
        elif "video" in source:
            media = [("video", None)]
        else:
            raise AssertionError(f"unexpected MiniMax init-smoke source: {source}")
        return [["user", *media, ("text", "Describe the media.")], ["assistant", ("text", "Done.")]]

    monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", fake_conv_preprocess)

    image_input = None
    video_input = None
    if media_case in ("image", "mixed"):
        image_path = tmp_path / f"minimax_m3_vl_{media_case}.jpg"
        Image.new("RGB", (28, 28), color=(127, 64, 32)).save(image_path)
        image_input = str(image_path)

    if media_case in ("video", "mixed"):
        pytest.importorskip("av")
        from veomni.data.multimodal.video_utils import save_video_tensors_to_file

        video = torch.zeros((4, 3, 28, 28), dtype=torch.uint8)
        video[:, 0] = torch.arange(4, dtype=torch.uint8).view(4, 1, 1) * 40
        video[:, 1] = 80
        video[:, 2] = 32
        video_path = tmp_path / f"minimax_m3_vl_{media_case}.mp4"
        save_video_tensors_to_file(video, str(video_path), fps=2)
        video_input = str(video_path) if media_case == "video" else video_path.read_bytes()

    def build_minimax_init_smoke_dataset(base):
        base.train_dataset = _MiniMaxTrainerInitDataset(
            base.data_transform,
            media_case,
            image_input=image_input,
            video_input=video_input,
        )
        base.args.compute_train_steps(len(base.train_dataset))
        base.train_steps = base.args.train_steps

    monkeypatch.setattr(base_trainer.BaseTrainer, "_build_dataset", build_minimax_init_smoke_dataset)

    parallel_state._PARALLEL_STATE = None
    if dist.is_initialized():
        dist.destroy_process_group()

    try:
        torch.manual_seed(20260618)
        config = get_model_config("./tests/toy_config/minimax_m3_vl_toy/config.json")
        config.text_config.vocab_size = max(
            config.text_config.vocab_size,
            config.image_token_id + 8,
            config.video_token_id + 8,
            _MiniMaxTrainerTokenizer.eos_token_id + 8,
        )
        args = VeOmniVLMArguments(
            model=VLMMModelArguments(
                config_path=config,
                ops_implementation=make_eager_ops_config(),
            ),
            data=VLMMDataArguments(
                train_path="dummy",
                chat_template="minimax_m3_vl",
                datasets_type="mapping",
                max_seq_len=256,
                dataloader=DataloaderConfig(num_workers=0, drop_last=False, pin_memory=False),
            ),
            train=VLMTrainingArguments(
                dyn_bsz=False,
                micro_batch_size=1,
                global_batch_size=1,
                init_device="cpu",
                checkpoint=CheckpointConfig(output_dir=str(tmp_path / f"minimax_m3_vl_{media_case}")),
                optimizer=OptimizerConfig(lr=1e-3),
                accelerator=AcceleratorConfig(
                    fsdp_config=FSDPConfig(
                        fsdp_mode="ddp",
                        mixed_precision=MixedPrecisionConfig(enable=True),
                    ),
                ),
                gradient_checkpointing=GradientCheckpointingConfig(enable=False),
            ),
        )

        trainer = _MiniMaxTrainerInitSmoke(args)
        assert trainer.base.train_dataloader is not None
        assert trainer.base.optimizer is not None
        assert trainer.base.lr_scheduler is not None
        assert trainer.base.state is not None

        micro_batches = next(iter(trainer.base.train_dataloader))
        assert len(micro_batches) == 1
        micro_batch = micro_batches[0]
        assert "multimodal_metadata" in micro_batch

        has_image = media_case in ("image", "mixed")
        has_video = media_case in ("video", "mixed")
        assert ("pixel_values" in micro_batch) is has_image
        assert ("pixel_values_videos" in micro_batch) is has_video
        assert micro_batch["image_mask"].any().item() is has_image
        assert micro_batch["video_mask"].any().item() is has_video

        trainer.base.micro_batch_token_len = count_loss_token(micro_batch)
        trainer.base.micro_batches_token_len = count_loss_token([micro_batch])
        before = trainer.base.model.lm_head.weight.detach().clone()
        loss, loss_dict = trainer.base.forward_backward_step(micro_batch)
        trainer.base.optimizer.step()
        trainer.base.lr_scheduler.step()
        trainer.base.optimizer.zero_grad()

        assert torch.isfinite(loss)
        assert set(loss_dict) == {"foundation_loss"}
        assert not torch.equal(before, trainer.base.model.lm_head.weight.detach())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        parallel_state._PARALLEL_STATE = None
