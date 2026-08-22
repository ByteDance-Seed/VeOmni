# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import TYPE_CHECKING, Any, Dict, List

from tqdm import trange

from ...utils import helper
from ...utils.dist_utils import all_reduce
from ...utils.logging import get_logger
from .base import Callback, TrainerState


logger = get_logger(__name__)


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


class MoERouterMonitorCallback(Callback):
    """Monitors MoE expert load distribution and logs heatmaps to wandb.

    Activation is gated only by ``moe_load_balance_monitor_interval > 0``; the
    monitor itself does not require wandb. Logging to wandb is gated by
    ``wandb.enable`` and ``global_rank == 0``.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self.monitor = None

        args: "VeOmniArguments" = self.trainer.args
        if args.train.moe_load_balance_monitor_interval <= 0:
            logger.info_rank0("MoE router monitor disabled (moe_load_balance_monitor_interval=0).")
            return

        config = self.trainer.model_config
        num_experts = getattr(config, "num_experts", None)
        if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
            text_config = getattr(config, "text_config", None)
            num_experts = getattr(text_config, "num_experts", None)
        if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
            logger.warning_rank0(
                "moe_load_balance_monitor_interval > 0 but model config has no positive "
                "'num_experts' or 'text_config.num_experts'. "
                "MoE router monitor not activated."
            )
            return

        from ...utils.moe_monitor import MoERouterMonitor, set_active_monitor

        # Process groups are read lazily in on_train_begin once the device
        # mesh is guaranteed to be initialized.
        self.monitor = MoERouterMonitor(num_experts=num_experts)
        set_active_monitor(self.monitor)
        ps = self.parallel_state
        logger.info_rank0(
            f"MoE router monitor created: num_experts={num_experts}, "
            f"interval={args.train.moe_load_balance_monitor_interval}, "
            f"ep_size={ps.ep_size if ps.ep_enabled else 1}"
        )

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        if self.monitor is None:
            return
        from ...utils.moe_monitor import attach_moe_router_monitor

        # fsdp_group is the dp_sp mesh dim — exactly the set of ranks that
        # hold distinct token slices. EP is intentionally not in this group;
        # see MoERouterMonitor.__init__ docstring.
        self.monitor.dp_group = self.parallel_state.fsdp_group

        attached = attach_moe_router_monitor(self.trainer.model, self.monitor)
        if attached == 0:
            logger.warning_rank0(
                "MoE router monitor: no recognized router modules found in the model. "
                "Disabling monitor. To add support for a new router class, register an "
                "extractor in veomni/utils/moe_monitor.py (see ROUTER_EXTRACTORS)."
            )
            from ...utils.moe_monitor import set_active_monitor

            set_active_monitor(None)
            self.monitor = None
        else:
            logger.info_rank0(f"MoE router monitor: attached to {attached} router module(s).")

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if self.monitor is None or state.global_step % args.train.moe_load_balance_monitor_interval != 0:
            return

        # Every rank participates in the cached DP+SP/FSDP collective and
        # resets its interval. Only global rank 0 formats scalars and images.
        metrics = self.monitor.compute_metrics(
            current_step=state.global_step,
            format_only_on=args.train.global_rank == 0,
        )
        if not metrics:
            return

        scalar_metrics = {key: value for key, value in metrics.items() if not key.endswith("_heatmap")}
        if not hasattr(self.trainer, "step_env_metrics") or self.trainer.step_env_metrics is None:
            self.trainer.step_env_metrics = {}
        self.trainer.step_env_metrics.update(scalar_metrics)

        if args.train.wandb.enable:
            import wandb

            wandb_metrics = {}
            for k, v in metrics.items():
                if not k.endswith("_heatmap"):
                    continue
                start, end = self.monitor._last_step_range
                wandb_metrics[k] = wandb.Image(v, caption=f"Steps {start}-{end}")
            if wandb_metrics:
                # The following WandbTraceCallback commits the scalar payload
                # for this same step; keep the image write in that open row.
                wandb.log(wandb_metrics, step=state.global_step, commit=False)

        start, end = self.monitor._last_step_range
        summaries = []
        for key in (
            "moe/avg_vio/avg",
            "moe/ep_rank_imbalance_before/avg",
            "moe/ep_rank_imbalance_after/avg",
            "moe/ep_moved_tokens/sum",
        ):
            if key in scalar_metrics:
                summaries.append(f"{key.rsplit('/', 2)[-2:]!s}={scalar_metrics[key]:.4f}")
        summary = ", ".join(summaries) if summaries else "no scalar summaries"
        logger.info_rank0(f"Step {state.global_step}: collected MoE monitor metrics (steps {start}-{end}); {summary}.")

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        from ...utils.moe_monitor import set_active_monitor

        set_active_monitor(None)
        if self.monitor is not None:
            logger.info_rank0("MoE router monitor disabled.")
        self.monitor = None


class WandbTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0 and args.train.wandb.enable:
            from dataclasses import asdict

            import wandb

            wandb.init(
                project=args.train.wandb.project,
                name=args.train.wandb.name,
                id=args.train.wandb.id,
                resume="allow" if args.train.wandb.id else None,
                config={**asdict(args.model), **asdict(args.data), **asdict(args.train)},
            )

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args

        if args.train.global_rank == 0 and args.train.wandb.enable:
            import wandb

            wandb.log(self.trainer.step_env_metrics, step=state.global_step)


class ProfileTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            self.profiler = helper.create_profiler(
                start_step=args.train.profile.start_step,
                end_step=args.train.profile.end_step,
                trace_dir=args.train.profile.trace_dir,
                record_shapes=args.train.profile.record_shapes,
                profile_memory=args.train.profile.profile_memory,
                with_stack=args.train.profile.with_stack,
                with_modules=args.train.profile.with_modules,
                global_rank=args.train.global_rank,
            )
            self.profiler.start()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            if state.global_step <= args.train.profile.end_step:
                self.profiler.step()

            if state.global_step == args.train.profile.end_step:
                self.profiler.stop()


class EnvironMeterCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)

        args: "VeOmniArguments" = self.trainer.args
        self.lora_config = trainer.model.get_lora_config() if hasattr(trainer.model, "get_lora_config") else None
        self.freeze_vit = getattr(args.train, "freeze_vit", None) if self.lora_config is None else None
        self.trainer.environ_meter = helper.EnvironMeter(
            config=trainer.model_config,
            global_batch_size=args.train.global_batch_size,
            empty_cache_steps=args.train.empty_cache_steps,
            enable_multisource=args.data.enable_multisource,
            dataloader=trainer.train_dataloader,
            data_path=args.data.train_path,
            gc_steps=args.train.gc_steps,
            parallel_state=self.parallel_state,
        )

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        for micro_batch in micro_batches:
            self.trainer.environ_meter.add(micro_batch)
        self.start_time = time.time()

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        delta_time = time.time() - self.start_time
        step_env_metrics = self.trainer.environ_meter.step(
            delta_time,
            global_step=state.global_step,
            lora_config=self.lora_config,
            freeze_vit=self.freeze_vit,
        )

        step_train_metrics = {
            "total_loss": loss,
        }
        step_train_metrics.update(loss_dict)
        step_train_metrics["grad_norm"] = grad_norm

        # gather training_step_info from all ranks
        step_train_metrics = {
            f"training/{k}": all_reduce(v, group=self.parallel_state.fsdp_group) for k, v in step_train_metrics.items()
        }

        if self.trainer.lr_scheduler is not None:
            lr = max(self.trainer.lr_scheduler.get_last_lr())
            step_train_metrics["training/lr"] = lr

        step_env_metrics.update(step_train_metrics)

        self.trainer.step_train_metrics = step_train_metrics
        self.trainer.step_env_metrics = step_env_metrics


class TqdmCallback(Callback):
    def on_epoch_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        self.data_loader_tqdm = trange(
            args.train_steps,
            desc=f"Epoch {state.epoch + 1}/{args.train.num_train_epochs}",
            total=args.train_steps,
            initial=self.trainer.start_step,
            disable=args.train.local_rank != 0,
        )

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        self.data_loader_tqdm.close()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        postfix = ", ".join(f"{k.split('/', 1)[-1]}: {v:.2f}" for k, v in self.trainer.step_train_metrics.items())
        self.data_loader_tqdm.set_postfix_str(postfix, refresh=False)
        self.data_loader_tqdm.update()
