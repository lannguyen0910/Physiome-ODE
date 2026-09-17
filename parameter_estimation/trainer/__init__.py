from registry import Registry

from .trainer import Trainer


TRAINER_REGISTRY = Registry("TRAINER")


TRAINER_REGISTRY.register(Trainer)


__all__ = [
    "TRAINER_REGISTRY",
    "Trainer",
]
