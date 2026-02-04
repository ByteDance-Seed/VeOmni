from veomni.arguments import parse_args
from veomni.trainer.base_rl_trainer import BaseRLTrainer
from veomni.trainer.vlm_trainer import Arguments, VLMTrainer


class VLMRLTrainer(BaseRLTrainer, VLMTrainer):
    pass


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = VLMRLTrainer(args)
    trainer.fit()
