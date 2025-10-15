from typing import Callable
from .base_trainer import DiTBaseTrainer
from .base_generator import DiTBaseGenerator

class DiTTrainerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, value: Callable):
        if name in cls._registry:
            raise ValueError(f"Trainer '{name}' is already registered")
        cls._registry[name] = value

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise KeyError(f"Trainer '{name}' is not registered, available trainers: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        trainer_cls = cls.get(name)
        return trainer_cls(*args, **kwargs)

    @classmethod
    def available_trainers(cls):
        return list(cls._registry.keys())

DiTTrainerRegistry.register("base", DiTBaseTrainer)
DiTTrainerRegistry.register("base_generator", DiTBaseGenerator)
