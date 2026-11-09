"""Typed-decision training and evaluation on VeOmni's shared lifecycle."""

import json
import math
from dataclasses import asdict
from functools import partial
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from veomni.data.data_collator import MainCollator, MakeMicroBatchCollator
from veomni.data.data_loader import DistributedDataloader
from veomni.data.decision.collator import DecisionCollator, encode_request
from veomni.data.decision.dataset import CaseEncoder, DecisionDataset, DecisionSampler
from veomni.data.decision.prepare import prepare_decision_data
from veomni.data.decision.typed_decisions import shuffled_request
from veomni.distributed.parallel_state import use_parallel_state
from veomni.models.decision.accelerated.model_runtime import DecisionModelRuntime
from veomni.utils.decision_metrics import finalize_metrics, metric_totals
from veomni.utils.device import get_device_type

from .base import BaseTrainer


class DecisionTrainer(BaseTrainer):
    """Generic decision orchestration; architecture behavior comes from the spec."""

    def __init__(self, args):
        super().__init__(args)
        self.rank = args.train.global_rank
        self.output = Path(args.train.checkpoint.output_dir)

    def on_train_begin(self):
        self.model.train()
        super().on_train_begin()

    def on_step_begin(self, micro_batches=None, **kwargs):
        # Count real requests only, including mixed real/padding microbatches.
        features = []
        for batch in micro_batches:
            offset = 0
            for length, padding in zip(
                batch["decision_request_lengths"], batch["decision_supervision"]["padding"], strict=True
            ):
                if not padding and not batch.get("padding_flag", False):
                    features.append(
                        {
                            "input_ids": batch["input_ids"][:, offset : offset + length],
                            "attention_mask": batch["attention_mask"][:, offset : offset + length],
                        }
                    )
                offset += length
        super().on_step_begin(micro_batches=features, **kwargs)

    def on_step_end(self, **kwargs):
        super().on_step_end(**kwargs)
        self._log({"step": self.state.global_step, "epoch": self.state.epoch, **self.step_train_metrics})

    def _build_model_runtime(self, model_name="base"):
        task = self.args.decision
        model = DecisionModelRuntime(
            self.args.model,
            model_name,
            train=self.args.train,
            decision_model=task.decision_model,
            decision_model_config=task.decision_model_config,
        )
        spec = model.spec
        self.loss = spec.build_loss(
            task.loss_type or spec.default_loss,
            smoothing=task.label_smoothing,
            brier=task.brier_weight,
            rps=task.rps_weight,
        )
        return model

    def _build_data_transform(self):
        self.data_transform = partial(
            encode_request,
            preprocessor=self.model.preprocessor,
            max_request_tokens=self.args.data.max_request_tokens,
        )

    def _build_dataset(self):
        self._prepare_data()
        self.case_encoder = CaseEncoder(
            self.data_transform,
            shuffled_request,
            self.args.train.seed,
            self.args.data.shuffle_choices,
        )
        self.train_dataset = DecisionDataset(self.decision_data.train_cases, transform=self.case_encoder)
        if self.args.train.dyn_bsz:
            self.args.compute_train_steps()
        else:
            self.args._train_steps = self.steps_per_epoch
        self.train_steps = self.args.train_steps

    def _build_collate_fn(self):
        padding_case = self.data_transform(self.decision_data.padding_case.request)
        module = self.model.unwrapped_module
        feature_collator = MainCollator(
            pad_to_length=self.args.train.pad_to_length,
            metadata_collate_func=getattr(module, "get_metadata_collate_func", lambda: None)(),
            data_collate_info=getattr(module, "get_extra_collate_infos", lambda: {})(),
        )
        self.collate_fn = DecisionCollator(padding_case=padding_case, feature_collator=feature_collator)

    def _build_dataloader(self):
        train = self.args.train
        if train.dyn_bsz:
            return super()._build_dataloader()
        state = self.model.parallel_state
        options = self.args.data.dataloader
        sampler = DecisionSampler(
            self.train_dataset, train.global_batch_size, state.dp_size, state.dp_rank, train.seed
        )
        kwargs = {}
        if options.num_workers:
            kwargs.update(prefetch_factor=options.prefetch_factor, persistent_workers=False)
        self.train_dataloader = DistributedDataloader(
            self.train_dataset,
            batch_size=train.global_batch_size // state.dp_size,
            sampler=sampler,
            collate_fn=MakeMicroBatchCollator(
                num_micro_batch=train.global_batch_size // (state.dp_size * train.micro_batch_size),
                internal_data_collator=self.collate_fn,
            ),
            num_workers=options.num_workers,
            pin_memory=options.pin_memory,
            drop_last=False,
            **kwargs,
        )

    def _prepare_data(self):
        train = self.args.train
        options = SimpleNamespace(
            **asdict(self.args.data), mode=self.args.decision.mode, eval_split=self.args.decision.eval_split
        )
        self.decision_data = prepare_decision_data(options, seed=train.seed)
        self.steps_per_epoch = math.ceil(len(self.decision_data.train_cases) / train.global_batch_size)

    @staticmethod
    def _write_json(path, data):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)

    def _log(self, data):
        if self.rank == 0:
            with (self.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(data, allow_nan=False) + "\n")

    def prepare_loss_reduction(self, batches):
        denominator = sum(
            float((batch["decision_supervision"]["weights"] * (batch["decision_supervision"]["counts"] > 1)).sum())
            for batch in batches
            if not batch.get("padding_flag", False)
        )
        denominator = torch.tensor(denominator, device=self.device, dtype=torch.float32)
        dist.all_reduce(denominator, group=self.model.parallel_state.fsdp_group)
        if denominator <= 0:
            raise ValueError("Optimizer step has no effective supervised questions")
        self.global_weight = denominator

    def prepare_micro_batch_loss(self, micro_batch):
        pass

    def _forward(self, batch):
        return self.model(batch)

    def forward_backward_step(self, batch):
        with use_parallel_state(self.model.parallel_state), self.model_fwd_context:
            output = self._forward(batch)
        with use_parallel_state(self.model.parallel_state):
            loss, loss_dict, aux_metrics = self.postforward(output, batch)
        with use_parallel_state(self.model.parallel_state), self.model_bwd_context:
            loss.backward()
        return loss, loss_dict, aux_metrics

    def postforward(self, output, batch):
        numerator, _ = self.loss(output, batch, self.device)
        # SP/CP is rejected: every DP rank contributes unique questions. The
        # dp_sp gradient average cancels dp_size in this weighted global mean.
        loss = numerator * (self.model.parallel_state.dp_size / self.global_weight)
        if batch.get("padding_flag", False):
            loss = loss * 0
        return loss, {"decision_loss": loss}, {"question_weight": self.global_weight.detach()}

    @torch.no_grad()
    def evaluate(self, split=None):
        split = split or self.args.decision.eval_split
        cases = self.decision_data.evaluation_cases(split)
        state = self.model.parallel_state
        dtype = torch.float32 if get_device_type() == "npu" else torch.float64
        totals = torch.zeros(7, device=self.device, dtype=dtype)
        was_training = self.model.model.training
        self.model.eval()
        path = self.output / f"predictions_{split}_step_{self.state.global_step}_rank_{self.rank}.jsonl"
        source = islice(iter(cases), state.dp_rank, None, state.dp_size)
        try:
            with path.open("w") as handle:
                # Equal forward counts are required by FSDP even on the last batch.
                for _ in range(math.ceil(len(cases) / state.dp_size)):
                    case = next(source, None)
                    batch = self.collate_fn([self.data_transform(case.request)] if case is not None else [None])
                    with use_parallel_state(state), self.model_fwd_context:
                        output = self.model(batch)
                    if case is not None:
                        totals += metric_totals(output, batch).to(dtype)
                        handle.write(
                            json.dumps(
                                {"case_id": case.case_id, "questions": self.model.spec.decode(output, batch)},
                                allow_nan=False,
                            )
                            + "\n"
                        )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=state.dp_group)
            metrics = finalize_metrics(totals)
            if self.rank == 0:
                self._write_json(self.output / f"evaluation_{split}_step_{self.state.global_step}.json", metrics)
            return metrics
        finally:
            self.model.train(was_training)
