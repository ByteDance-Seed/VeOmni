from .base_trainer import DiTTrainerRegistry, DiTBaseTrainer

DiTTrainerRegistry.register("base", DiTBaseTrainer)
