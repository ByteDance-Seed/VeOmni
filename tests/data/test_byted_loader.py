import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from veomni.arguments import DataArguments
from veomni.data import build_dataloader
from veomni.data.byted_loader import (
    BYTED_LOADER_TYPE,
    BytedLoaderDatasetSpec,
    build_byted_loader_dataset_spec,
    set_byted_loader_client_factory_for_tests,
)


def _fake_ps(sp_size: int = 1):
    return types.SimpleNamespace(
        global_rank=0,
        local_rank=0,
        world_size=1,
        dp_size=1,
        dp_rank=0,
        sp_size=sp_size,
        sp_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
    )


class FakeBytedClient:
    def __init__(self, *, transforms_dict, microbatch_transforms, seq_len_extract, total_steps, **kwargs):
        self.transforms_dict = transforms_dict
        self.microbatch_transforms = microbatch_transforms
        self.seq_len_extract = seq_len_extract
        self.total_steps = total_steps
        self.index = 0
        self.loaded = {}

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.total_steps:
            raise StopIteration
        self.index += 1
        raw_sample = {
            "messages": [{"role": "user", "content": "hello"}],
            "source_name": "source_a",
            "byted_private": "drop-me",
        }
        samples = self.transforms_dict["text_transform"][0](raw_sample)
        assert self.seq_len_extract(samples[0]) == 4
        return [self.microbatch_transforms[0](samples)]

    def __len__(self):
        return self.total_steps

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state):
        self.loaded = dict(state)
        self.index = int(state.get("index", 0))


@pytest.fixture(autouse=True)
def reset_fake_client():
    set_byted_loader_client_factory_for_tests(None)
    yield
    set_byted_loader_client_factory_for_tests(None)


def test_byted_loader_registration_does_not_import_real_package():
    assert "bytedance.dataloader" not in sys.modules


def test_native_dataloader_ignores_byted_only_config_fields(monkeypatch):
    import veomni.data.data_collator as m_col
    import veomni.data.data_loader as m_dl

    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: _fake_ps())
    monkeypatch.setattr(m_col, "get_parallel_state", lambda: _fake_ps())

    class TinyDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return [{"input_ids": [idx], "labels": [idx], "attention_mask": [1]}]

    dataloader = build_dataloader(
        "native",
        dataset=TinyDataset(),
        micro_batch_size=1,
        global_batch_size=1,
        dataloader_batch_size=1,
        max_seq_len=4,
        train_steps=1,
        dyn_bsz=False,
        dyn_bsz_runtime="main",
        dyn_bsz_count_mode="total",
        dyn_bsz_physical_overflow_ratio=1.5,
        dyn_bsz_buffer_size=1,
        seed=0,
        num_workers=0,
        prefetch_factor=None,
        collate_fn=lambda samples: samples[0],
        file_type="lance",
        worker_subprocess_num=4,
    )

    batch = next(iter(dataloader))
    assert batch[0]["input_ids"] == [0]

    data_args = DataArguments(train_path="dummy")
    assert data_args.dataloader.type == "native"
    assert data_args.dataloader.file_type == "lance"

    # Registration happens through veomni.data import, but the real package is
    # loaded only after type=byted_loader is selected and the builder runs.
    assert "bytedance.dataloader" not in sys.modules


def test_build_byted_loader_with_fake_client(monkeypatch):
    import veomni.data.byted_loader as m_byted

    monkeypatch.setattr(m_byted, "get_parallel_state", lambda: _fake_ps())
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    set_byted_loader_client_factory_for_tests(
        lambda **kwargs: FakeBytedClient(**kwargs),
    )

    def transform(sample, source_name=None):
        return {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "byted_private": sample["byted_private"],
            "source_name": source_name,
        }

    def collate(samples):
        assert samples == [
            {
                "input_ids": [1, 2, 3, 4],
                "labels": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
            }
        ]
        return {key: torch.tensor([value]) for key, value in samples[0].items()}

    dataset = BytedLoaderDatasetSpec(
        train_path="catalog.namespace.demo_lance",
        raw_train_path="catalog.namespace.demo_lance",
        source_config_path="",
        transform=transform,
        is_multisource_yaml=False,
        dataset_name_before_bypass="mapping",
        data_type="conversation",
        shuffle=True,
        shuffle_seed=None,
        shuffle_shard_nums=1,
        ckpt_dir="/tmp/byted_loader_ckpt",
        save_ckpt_interval=10,
    )
    dataloader = build_dataloader(
        BYTED_LOADER_TYPE,
        dataset=dataset,
        micro_batch_size=1,
        global_batch_size=1,
        dataloader_batch_size=1,
        max_seq_len=128,
        train_steps=2,
        dyn_bsz=True,
        dyn_bsz_buffer_size=1,
        bsz_warmup_ratio=0,
        seed=123,
        collate_fn=collate,
        strict_api_check=False,
    )

    batch = next(iter(dataloader))
    assert len(batch) == 1
    assert set(batch[0]) == {"input_ids", "labels", "attention_mask"}
    assert dataloader.state_dict()["client_state"] == {"index": 1}
    dataloader.load_state_dict({"client_state": {"index": 0}, "adapter_epoch": 3})
    assert dataloader.state_dict()["adapter_epoch"] == 3


def test_byted_loader_transform_without_source_name_preserves_custom_inputs(monkeypatch):
    import veomni.data.byted_loader as m_byted

    monkeypatch.setattr(m_byted, "get_parallel_state", lambda: _fake_ps())
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    set_byted_loader_client_factory_for_tests(
        lambda **kwargs: FakeBytedClient(**kwargs),
    )

    def transform(sample):
        return {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "token_type_ids": [0, 0, 0, 0],
            "source_name": sample["source_name"],
            "ds_idx": 7,
            "byted_private": sample["byted_private"],
        }

    def collate(samples):
        assert samples == [
            {
                "input_ids": [1, 2, 3, 4],
                "labels": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "token_type_ids": [0, 0, 0, 0],
            }
        ]
        return {key: torch.tensor([value]) for key, value in samples[0].items()}

    dataset = BytedLoaderDatasetSpec(
        train_path="catalog.namespace.demo_lance",
        raw_train_path="catalog.namespace.demo_lance",
        source_config_path="",
        transform=transform,
        is_multisource_yaml=False,
        dataset_name_before_bypass="mapping",
        data_type="conversation",
        shuffle=True,
        shuffle_seed=None,
        shuffle_shard_nums=1,
        ckpt_dir="/tmp/byted_loader_ckpt",
        save_ckpt_interval=10,
    )
    dataloader = build_dataloader(
        BYTED_LOADER_TYPE,
        dataset=dataset,
        micro_batch_size=1,
        global_batch_size=1,
        dataloader_batch_size=1,
        max_seq_len=128,
        train_steps=1,
        dyn_bsz=True,
        dyn_bsz_buffer_size=1,
        bsz_warmup_ratio=0,
        seed=123,
        collate_fn=collate,
        strict_api_check=False,
    )

    batch = next(iter(dataloader))
    assert len(batch) == 1
    assert set(batch[0]) == {"input_ids", "labels", "attention_mask", "token_type_ids"}


def test_dataset_bypass_disables_veomni_multisource_meter():
    args = SimpleNamespace(
        data=DataArguments(train_path="sources.yaml"),
        train=SimpleNamespace(checkpoint=SimpleNamespace(output_dir="/tmp/out", save_steps=10)),
    )
    args.data.dataloader.type = BYTED_LOADER_TYPE
    assert args.data.enable_multisource is True

    spec = build_byted_loader_dataset_spec(args, lambda sample, source_name=None: sample)

    assert spec.source_config_path == "sources.yaml"
    assert spec.is_multisource_yaml is True
    assert spec.dataset_name_before_bypass == "interleave"
    assert args.data.enable_multisource is False
    assert args.data._byted_loader_original_enable_multisource is True


def test_byted_loader_parquet_concurrency_fast_fails(monkeypatch):
    import veomni.data.byted_loader as m_byted

    monkeypatch.setattr(m_byted, "get_parallel_state", lambda: _fake_ps())
    dataset = BytedLoaderDatasetSpec(
        train_path="catalog.namespace.demo_parquet",
        raw_train_path="catalog.namespace.demo_parquet",
        source_config_path="",
        transform=lambda sample, source_name=None: sample,
        is_multisource_yaml=False,
        dataset_name_before_bypass="mapping",
        data_type="conversation",
        shuffle=True,
        shuffle_seed=None,
        shuffle_shard_nums=1,
        ckpt_dir="/tmp/byted_loader_ckpt",
        save_ckpt_interval=10,
    )

    with pytest.raises(ValueError, match="parquet is restricted"):
        build_dataloader(
            BYTED_LOADER_TYPE,
            dataset=dataset,
            micro_batch_size=1,
            global_batch_size=1,
            dataloader_batch_size=1,
            max_seq_len=128,
            train_steps=1,
            dyn_bsz=True,
            dyn_bsz_buffer_size=1,
            bsz_warmup_ratio=0,
            seed=123,
            collate_fn=lambda samples: samples[0],
            file_type="parquet",
            worker_subprocess_num=2,
            worker_parallel_read_num=1,
            strict_api_check=False,
        )


def test_forbidden_write_roots_are_env_driven(monkeypatch):
    import veomni.data.byted_loader as m_byted

    monkeypatch.setenv("VEOMNI_FORBIDDEN_WRITE_ROOTS", "/old/root,hdfs://old/root")
    args = SimpleNamespace(
        data=SimpleNamespace(
            train_path="data.yaml",
            dataloader=SimpleNamespace(ckpt_dir="hdfs://old/root/ckpt", save_ckpt_interval=-1),
            enable_multisource=True,
            dataset_name="interleave",
            data_type="conversation",
        ),
        train=SimpleNamespace(checkpoint=SimpleNamespace(output_dir="/tmp/out", save_steps=10)),
    )

    with pytest.raises(ValueError, match="forbidden write root"):
        m_byted.build_byted_loader_dataset_spec(args, lambda sample, source_name=None: sample)
