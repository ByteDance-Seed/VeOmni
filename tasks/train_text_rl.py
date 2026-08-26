from veomni.arguments import parse_args
from veomni.trainer.base_rl_trainer import BaseRLTrainer
from veomni.trainer.text_trainer import TextTrainer, VeOmniArguments


class TextRLTrainer(TextTrainer):
    base: BaseRLTrainer

    def __init__(self, args: VeOmniArguments) -> None:
        # BaseRLTrainer.__init__ is NOT called here; we call its private
        # helpers one-by-one so the sequence is explicit.
        self.base = BaseRLTrainer.__new__(BaseRLTrainer)
        self.base.args = args

        self.base.device = self.base.setup_distributed(args)  # registers ParallelState("base") before seed
        self.base.model = self.base.build_model_runtime()

        # rewrite build_data_transform to support conversation dataset
        self._build_data_transform()

        self.base._build_dataset()
        self.base._build_collate_fn()
        self.base._build_dataloader()
        self.base.build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()

        self.base._build_preforward_postforward()

    def train(self):
        super().train()


if __name__ == "__main__":
    args = parse_args(VeOmniArguments)
    trainer = TextRLTrainer(args)
    trainer.train()
