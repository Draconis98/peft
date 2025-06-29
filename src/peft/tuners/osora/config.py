import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union, Literal

from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class OSoraRuntimeConfig:
    """
    This is the sub-configuration class to store the runtime configurations for the model.

    Args:
        ephemeral_gpu_offload (`bool`):
            Whether to use ephemeral GPU offloading for models partially kept in CPU memory.
    """

    ephemeral_gpu_offload: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use ephemeral GPU offloading for models partially kept in CPU memory. Ephemeral GPU offloading result in "
                "the data involved in intense operations being momentarily copied over to the GPU, and the results copied "
                "back to CPU. There is a momentary VRAM overhead, but operations are generally orders of magnitude faster "
                "compared to performing them on the CPU. This is useful when parts of the model and/or components (such "
                "as adapters) are kept in CPU memory until they are needed. Rather than perform expensive operations on "
                "small data, the data is transferred to the GPU on-demand, the operation(s) performed, and the results "
                "moved back to CPU memory. Currently only affects DoRA initialization."
            )
        },
    )

@dataclass
class OSoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`OSoraModel`].
    """
    r: int = field(default=256, metadata={"help": "The rank of the OSora layer."})

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with OSora."
                "Only linear layers are supported."
            )
        }
    )
    
    osora_dropout: float = field(default=0.0, metadata={"help": "The dropout rate of the OSora layer."})

    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"}
    )

    bias: str = field(default="none", metadata={"help": "Bias type for OSora. Can be 'none', 'all' or 'osora_only'"})

    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from OSora layers to be set as trainable and saved in the final checkpoint."
                "For example, in Sequence Classification or Token Classification tasks, the final layer"
                "`classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        }
    )
    

    init_osora_weights: Literal["default", "gaussian", "kaiming", "zeros", "in_features", "fix_O", "fix_S"] = field(
        default="default",
        metadata={
            "help": (
                "How to initialize the weights of the OSora layer."
                "Pass `'default'` (default) to use the default initialization."
                "Pass `'gaussian'` to use Gaussian initialization."
                "Pass `'kaiming'` to use Kaiming initialization."
                "Pass `'zeros'` to initialize the weights to zeros."
                "Pass `'in_features'` to initialize the weights to the in_features of the layer."
                "Pass `'fix_O'` to fix the weights of osora_O."
                "Pass `'fix_S'` to fix the weights of osora_S."
            )
        }
    )
    
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, if this argument is specified, it will apply the OSora transformations on"
                "the layer indexes that are specified inside this list. If a single integer is passed, PEFT will transform"
                "only the layer at this index."
            )
        }
    )
    
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer"
                "pattern is not in the common layers pattern."
            )
        }
    )

    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using LoRA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate LoRA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
                "The format is a list of [start, end) pairs which specify the layer ranges to stack. For example:\n"
                "   Original model has 5 layers labelled by their position in the model: `[0, 1, 2, 3, 4]`\n"
                "   layer_replication: `[[0, 4], [2, 5]]`\n"
                "   Final model will have this arrangement of original layers: `[0, 1, 2, 3, 2, 3, 4]`\n"
                "This format is based on what is used for pass-through merges in mergekit. It makes it simple to select sequential "
                "ranges of a model and stack them while reusing layers at either end of each sequence."
            )
        },
    )

    use_dora: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable <a href='https://arxiv.org/abs/2402.09353'>'Weight-Decomposed Low-Rank Adaptation' (DoRA)</a>. This technique decomposes the updates of the "
                "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
                "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
                "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger"
                "overhead than pure LoRA, so it is recommended to merge weights for inference."
            )
        }
    )
    
    # use_in_feature_ones: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "If False (default), osora_O's dimension will be out_feature. If True, osora_O's dimension will be in_feature."
    #         )
    #     }
    # )

    # no_ones: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "If True, osora_O will not be initialized to ones."
    #         )
    #     }
    # )

    # fix_O_or_S: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "If specified, osora_O or osora_S will not be updated during training."
    #         )
    #     }
    # )
    
    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.OSORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )