from typing import Callable
from transformers import AutoModel, AutoProcessor

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


class DiTBaseTrainer:
    def __init__(self, condition_model_config, dit_model_config, **kwargs):
        self.condition_model = AutoModel.from_pretrained(condition_model_config)
        self.processor = AutoProcessor.from_pretrained(condition_model_config)

        self.dit_model = AutoModel.from_pretrained(dit_model_config)
        