"""Data-pipeline dry-run entry point.

Drives a trainer's data loop without any model forward/backward, to check the data
pipeline cheaply.

It reuses the *unmodified* training YAML (passed as the positional config file, exactly
like ``tasks/train_*.py``) and takes the dry-run knobs separately on the CLI

    bash train.sh tasks/data_sim.py /tmp/training_config.yaml \
        --data_sim.trainer_type vlm \
        --data_sim.max_step 100 \
        --data_sim.data_step_range 0 4

Knobs (``--data_sim.*``, consumed here and stripped before the trainer's own parse_args):
  - ``trainer_type``:    text | vlm | dit. Which trainer's pipeline to simulate.
  - ``max_step``:        Iterate at most up to this step per epoch. Defaults to train_steps.
  - ``data_step_range``: ``[a, b)`` steps for which to dump the loaded batch's concrete
                         content and a content hash (for cross-run / resume comparison).
"""

import argparse
import hashlib
import importlib
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from veomni.arguments import parse_args
from veomni.utils import helper, logging
from veomni.utils.device import synchronize


logger = logging.get_logger(__name__)


# trainer_type -> (module, trainer_class, args_class)
_TRAINERS = {
    "text": ("veomni.trainer.text_trainer", "TextTrainer", "VeOmniArguments"),
    "vlm": ("veomni.trainer.vlm_trainer", "VLMTrainer", "VeOmniVLMArguments"),
    "dit": ("veomni.trainer.dit_trainer", "DiTTrainer", "VeOmniDiTArguments"),
}


@dataclass
class DataSimArguments:
    """Dry-run knobs, parsed from ``--data_sim.*`` CLI flags."""

    trainer_type: str = "text"
    max_step: Optional[int] = None  # iterate up to this step per epoch; None -> train_steps
    data_step_range: Optional[List[int]] = None  # [a, b): dump content + hash for these steps


def _hash_batch(micro_batch: Any) -> str:
    """Order-stable SHA256 over one step's data.

    Recurses through dicts / lists / tuples. Tensors are hashed by their device-independent
    cpu bytes + dtype + shape (never by repr, which would embed ``device=`` and truncate);
    leaf scalars (e.g. ``padding_flag``) by their repr. Stable across runs / devices, so a
    step can be compared cross-run or before-vs-after resume.
    """
    hasher = hashlib.sha256()

    def feed(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            if t.dtype == torch.bfloat16:  # numpy has no bfloat16; widen deterministically
                t = t.to(torch.float32)
            hasher.update(f"{t.dtype}{tuple(t.shape)}".encode())
            hasher.update(t.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                hasher.update(repr(key).encode())
                feed(value[key])
        elif isinstance(value, (list, tuple)):
            hasher.update(f"{type(value).__name__}{len(value)}".encode())
            for item in value:
                feed(item)
        else:
            hasher.update(repr(value).encode())

    feed(micro_batch)
    return hasher.hexdigest()


class DataSim:
    """Wrap an already-constructed trainer and replay its data loop without training.

    Only the trainer's ``.base`` (a ``BaseTrainer``) is used — the dataloader, callbacks,
    ``state`` and distributed teardown all live there, and ``TextTrainer`` / ``VLMTrainer``
    / ``DiTTrainer`` all build it the same way.
    """

    def __init__(self, trainer: Any, sim_args: DataSimArguments) -> None:
        self.trainer = trainer
        self.base = trainer.base
        self.sim_args = sim_args

    def dry_run(self) -> None:
        base = self.base
        args = base.args
        local_rank = args.train.local_rank

        end_step = self.sim_args.max_step if self.sim_args.max_step is not None else args.train_steps

        # [dump_lo, dump_hi) steps get their content + hash dumped; (0, 0) means none.
        data_step_range = self.sim_args.data_step_range
        if data_step_range is not None and len(data_step_range) != 2:
            raise ValueError(f"data_sim.data_step_range must be [start, end], got {data_step_range}.")
        dump_lo, dump_hi = data_step_range if data_step_range else (0, 0)

        base.on_train_begin()
        logger.info(
            f"Rank{local_rank} Start data dry-run. "
            f"Start step: {base.start_step}. "
            f"Max step: {end_step}. "
            f"Dump range: {self.sim_args.data_step_range}. "
            f"Start epoch: {base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(base.start_epoch, args.train.num_train_epochs):
            if hasattr(base.train_dataloader, "set_epoch"):
                base.train_dataloader.set_epoch(epoch)
            base.state.epoch = epoch

            base.on_epoch_begin()

            # Create a batch generator
            data_iterator = iter(base.train_dataloader)

            consumed = 0
            for step in range(base.start_step, end_step):
                try:
                    micro_batch = next(data_iterator)
                    # No train_step: this is a data dry-run, only pull from the dataloader.
                    consumed += 1
                    if dump_lo <= step < dump_hi:
                        self._dump_step_content(local_rank, epoch, step, micro_batch)
                except StopIteration:
                    logger.info(
                        f"Rank{local_rank} epoch:{epoch} Dataloader finished with "
                        f"drop_last {args.data.dataloader.drop_last} after {consumed} step(s)."
                    )
                    break

            base.on_epoch_end()

            base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        base.on_train_end()

        synchronize()

        base.destroy_distributed()

    def _dump_step_content(self, local_rank: int, epoch: int, step: int, micro_batch: Any) -> None:
        """Dump the concrete loaded content and a content hash for one step (every rank).

        The hash fingerprints all tensor bytes + structure, so it can be compared across
        runs / before-vs-after resume to verify the data pipeline is deterministic.
        """
        digest = _hash_batch(micro_batch)
        logger.info(f"Rank{local_rank} epoch:{epoch} step:{step} hash={digest}")
        logger.info(f"Rank{local_rank} epoch:{epoch} step:{step} content={micro_batch}")


def _parse_sim_args(argv: List[str]) -> "tuple[DataSimArguments, List[str]]":
    """Pull the ``--data_sim.*`` flags out of argv, returning (sim_args, remaining_argv).

    The remaining argv (config file + trainer overrides) is handed to ``parse_args`` as-is.
    """
    peeker = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    peeker.add_argument("--data_sim.trainer_type", dest="trainer_type", default=None)
    peeker.add_argument("--data_sim.max_step", dest="max_step", type=int, default=None)
    peeker.add_argument("--data_sim.data_step_range", dest="data_step_range", nargs="+", type=int, default=None)
    known, remaining = peeker.parse_known_args(argv)

    trainer_type = known.trainer_type if known.trainer_type is not None else "text"
    if trainer_type not in _TRAINERS:
        raise ValueError(f"Unknown data_sim.trainer_type {trainer_type!r}, expected one of {list(_TRAINERS)}.")

    sim_args = DataSimArguments(
        trainer_type=trainer_type,
        max_step=known.max_step,
        data_step_range=known.data_step_range,
    )
    return sim_args, remaining


if __name__ == "__main__":
    sim_args, remaining_argv = _parse_sim_args(sys.argv[1:])

    module_name, trainer_cls_name, args_cls_name = _TRAINERS[sim_args.trainer_type]
    module = importlib.import_module(module_name)
    trainer_cls = getattr(module, trainer_cls_name)
    args_cls = getattr(module, args_cls_name)

    sys.argv = [sys.argv[0], *remaining_argv]
    args = parse_args(args_cls)

    trainer = trainer_cls(args)
    DataSim(trainer, sim_args).dry_run()
