from veomni.arguments import parse_args
from veomni.trainer.base_rl_trainer import BaseRLTrainer
from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMTrainer


class VLMRLTrainer(VLMTrainer):
    base: BaseRLTrainer

    def __init__(self, args: VeOmniVLMArguments):
        # BaseRLTrainer.__init__ is NOT called here; we call its private
        # helpers one-by-one so the sequence is explicit.
        self.base = BaseRLTrainer.__new__(BaseRLTrainer)
        self.base.args = args

        self.base._setup()

        # rewrite build model to support data balancing
        self._build_model()

        # rewrite freeze_model_module to support freeze multimodal encoder, etc.
        self._freeze_model_module()

        # rewrite build_model_assets to support chat_template and processor for multimodal datasets
        self._build_model_assets()

        # rewrite build_data_transform to support multimodal transform
        self._build_data_transform()

        self.base._build_dataset()

        # rewrite build_collate_fn to support multimodal collate_fn
        self._build_collate_fn()

        self.base._build_dataloader()
        self.base._build_parallelized_model()

        # rewrite build_optimizer to support different lr param groups
        self._build_optimizer()

        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()

        self.base._build_preforward_postforward()

    def train(self):
        self.base.train()


if __name__ == "__main__":
    args = parse_args(VeOmniVLMArguments)
    trainer = VLMRLTrainer(args)
    trainer.train()
