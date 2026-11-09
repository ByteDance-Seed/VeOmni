from veomni.arguments import parse_args
from veomni.arguments.decision_arguments import VeOmniDecisionArguments
from veomni.trainer.decision_trainer import DecisionTrainer


if __name__ == "__main__":
    args = parse_args(VeOmniDecisionArguments)
    trainer = DecisionTrainer(args)
    if args.decision.mode == "evaluate":
        trainer.load()
        trainer.evaluate()
        trainer.destroy_distributed()
    else:
        trainer.train()
