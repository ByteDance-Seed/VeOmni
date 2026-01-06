from typing import TYPE_CHECKING

from ...utils import helper
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import Arguments


class WandbTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "Arguments" = self.trainer.args
        if args.train.global_rank == 0 and args.train.use_wandb:
            import wandb

            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "Arguments" = self.trainer.args

        if args.train.global_rank == 0 and args.train.use_wandb:
            import wandb

            wandb.log(self.trainer.step_train_metrics, step=state.global_step)


class ProfileTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "Arguments" = self.trainer.args
        if args.train.profile_this_rank:
            self.profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
                global_rank=args.train.global_rank,
            )
            self.profiler.start()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "Arguments" = self.trainer.args
        if args.train.profile_this_rank:
            if state.global_step <= args.train.profile_end_step:
                self.profiler.step()

            if state.global_step == args.train.profile_end_step:
                self.profiler.stop()
