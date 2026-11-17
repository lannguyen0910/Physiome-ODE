from registry import Registry

from .gru import GRUEncoder
from .tcn import TCNEncoder
from .transformer import TransformerEncoder
from .neural_ode import NeuralODEEncoder
from .contiformer_encoder import ContiFormerEncoder

ENCODER_REGISTRY = Registry("ENCODER")


ENCODER_REGISTRY.register(GRUEncoder)
ENCODER_REGISTRY.register(TCNEncoder)
ENCODER_REGISTRY.register(TransformerEncoder)
ENCODER_REGISTRY.register(NeuralODEEncoder)
ENCODER_REGISTRY.register(ContiFormerEncoder)


__all__ = [
    "ENCODER_REGISTRY",
    "GRUEncoder",
    "TCNEncoder",
    "TransformerEncoder",
    "NeuralODEEncoder",
    "ContiFormerEncoder"
]
