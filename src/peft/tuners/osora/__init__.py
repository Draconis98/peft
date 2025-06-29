from peft.utils import register_peft_method

from .config import OSoraConfig, OSoraRuntimeConfig
from .layer import Linear, OSoraLayer
from .model import OSoraModel


__all__ = ["OSoraConfig", "OSoraLayer", "Linear", "OSoraModel", "OSoraRuntimeConfig"]

register_peft_method(name="osora", config_cls=OSoraConfig, model_cls=OSoraModel, prefix="osora_")