from __future__ import annotations

import math
import warnings
from typing import List, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .dora import DoraLinearLayer


class OSoraLayer(BaseTunerLayer):
    adapter_layer_names = ("osora_S", "osora_O") # tranable
    other_param_names = ("osora_U", "osora_V", "osora_dropout", "init_osora_weights") # fixed

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.osora_dropout = nn.ModuleDict({})
        self.use_dora: dict[str, bool] = {}
        self.osora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        
        self.osora_S = nn.ParameterDict({})
        self.osora_O = nn.ParameterDict({})
        
        self.osora_U = nn.ModuleDict({})
        self.osora_V = nn.ModuleDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features

        self.in_features = in_features
        self.out_features = out_features

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)
    
    def update_layer(
            self, 
            adapter_name, 
            r,
            osora_dropout,
            init_osora_weights,
            use_dora: bool = False,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r

        if osora_dropout > 0.0:
            osora_dropout_layer = nn.Dropout(p=osora_dropout)
        else:
            osora_dropout_layer = nn.Identity()

        self.osora_dropout.update(nn.ModuleDict({adapter_name: osora_dropout_layer}))

        if isinstance(init_osora_weights, str) and init_osora_weights.lower() == "in_features":
            self.osora_O[adapter_name] = nn.Parameter(torch.ones(self.in_features))
        elif isinstance(init_osora_weights, str) and init_osora_weights.lower() == "zeros":
            self.osora_O[adapter_name] = nn.Parameter(torch.zeros(self.out_features))
        elif isinstance(init_osora_weights, str) and init_osora_weights.lower() == "kaiming":
            self.osora_O[adapter_name] = nn.Parameter(torch.empty(self.out_features))
            nn.init.kaiming_uniform_(self.osora_O[adapter_name].unsqueeze(-1), a=math.sqrt(5))
            self.osora_O[adapter_name] = self.osora_O[adapter_name].squeeze(-1)
        elif isinstance(init_osora_weights, str) and init_osora_weights.lower() == "gaussian":
            self.osora_O[adapter_name] = nn.Parameter(torch.empty(self.out_features))
            nn.init.normal_(self.osora_O[adapter_name], std=1 / self.r[adapter_name])
        elif isinstance(init_osora_weights, str) and \
            (init_osora_weights.lower() == "fix_o" or \
                init_osora_weights.lower() == "fix_s" or \
                init_osora_weights.lower() == "default"):
            self.osora_O[adapter_name] = nn.Parameter(torch.ones(self.out_features))
        else:
            raise ValueError(f"Unknown initialization {init_osora_weights=}")
        
        self.osora_U[adapter_name] = nn.Linear(self.out_features, r, bias=False)
        self.osora_S[adapter_name] = nn.Parameter(torch.ones(r))
        self.osora_V[adapter_name] = nn.Linear(r, self.in_features, bias=False)
        
        weight = self.get_base_layer().weight
        dtype, device = weight.dtype, weight.device
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize OSoRA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)

        O = self.osora_O[adapter_name].to(device)
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        Ur = U[:, : self.r[adapter_name]]
        Sr = S[: self.r[adapter_name]]
        Vhr = Vh[: self.r[adapter_name], :]

        self.osora_U[adapter_name].weight.data = Ur.contiguous()
        self.osora_S[adapter_name].data = Sr
        self.osora_V[adapter_name].weight.data = Vhr.contiguous()

        if isinstance(init_osora_weights, str) and init_osora_weights.lower() == "in_features":
            weight = weight.data - Ur @ torch.diag(Sr) @ Vhr @ torch.diag(O)
        else:
            if not (isinstance(init_osora_weights, str) and init_osora_weights.lower() == "zeros"):
                weight = weight.data - torch.diag(O) @ Ur @ torch.diag(Sr) @ Vhr
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: str) -> None:
        if not self.osora_magnitude_vector:
            # first dora layer being added, add osora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("osora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, "fan_in_fan_out", False))
        osora_U = self.osora_U[adapter_name].weight
        osora_V = self.osora_V[adapter_name].weight
        osora_S = self.osora_S[adapter_name]
        osora_O = self.osora_O[adapter_name]

        dora_layer.update_layer(
            base_layer=self.get_base_layer(),
            osora_U=osora_U,
            osora_V=osora_V,
            osora_S=osora_S,
            osora_O=osora_O,
        )
        self.osora_magnitude_vector[adapter_name] = dora_layer

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value
    
    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.osora_S.keys():
                continue

            osora_O = self.osora_O[active_adapter]
            osora_U = self.osora_U[active_adapter].weight
            osora_S = self.osora_S[active_adapter]
            osora_V = self.osora_V[active_adapter].weight
            dropout = self.osora_dropout[active_adapter]

            sub_batch = x[sub_batch_indices_list[i]].to(osora_V.dtype)
            if not (isinstance(self.init_osora_weights, str) and self.init_osora_weights.lower() == "in_features"):
                osora_output = osora_O * F.linear(osora_S * F.linear(dropout(sub_batch), osora_V), osora_U)
            else:
                osora_output = F.linear(osora_S * F.linear(osora_O * dropout(sub_batch), osora_V), osora_U)
            result[sub_batch_indices_list[i]] += osora_output.to(torch_result_dtype)

        return result


class Linear(nn.Linear, OSoraLayer):
    # OSora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        osora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_osora_weights: str = "default",
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        OSoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.init_osora_weights = init_osora_weights
        self.update_layer(adapter_name, r, osora_dropout, init_osora_weights, use_dora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.osora_S.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.osora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out))
                            .detach()
                        )
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.osora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights += dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        weight_norm = (
                            self.osora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight.data, transpose(delta_weight, self.fan_in_fan_out)
                            )
                            .detach()
                        )
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.osora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.osora_S.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.osora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig
    
    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        osora_O = self.osora_O[adapter]
        osora_U = self.osora_U[adapter]
        osora_S = self.osora_S[adapter]
        osora_V = self.osora_V[adapter]

        device = osora_V.device
        dtype = osora_V.dtype

        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        if cast_to_fp32:
            osora_U = osora_U.float()
            osora_S = osora_S.float()
            osora_V = osora_V.float()
            osora_O = osora_O.float()

        if not (isinstance(self.init_osora_weights, str) and self.init_osora_weights.lower() == "in_features"):
            output_tensor = transpose(torch.diag(osora_O) @ osora_U @ torch.diag(osora_S) @ osora_V, self.fan_in_fan_out)
        else:
            output_tensor = transpose(osora_U @ torch.diag(osora_S) @ osora_V @ torch.diag(osora_O), self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype)

            # cast back the weights
            self.osora_U[adapter].data = osora_U.to(dtype)
            self.osora_S[adapter].data = osora_S.to(dtype)
            self.osora_V[adapter].weight.data = osora_V.to(dtype)
            self.osora_O[adapter].data = osora_O.to(dtype)
            
        return output_tensor
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.osora_S.keys():
                    continue

                osora_U = self.osora_U[active_adapter].weight
                osora_S = self.osora_S[active_adapter]
                osora_V = self.osora_V[active_adapter].weight
                osora_O = self.osora_O[active_adapter]

                dropout = self.osora_dropout[active_adapter]
                x = x.to(osora_S.dtype)
                # result = result + F.linear(dropout(x),torch.diag(osora_O) @ osora_U @ torch.diag(osora_S) @ osora_V
                if not (isinstance(self.init_osora_weights, str) and self.init_osora_weights.lower() == "in_features"):
                    result = result + osora_O * F.linear(osora_S * F.linear(dropout(x), osora_V), osora_U)
                else:
                    if not self.use_dora[active_adapter]:
                        result = result + F.linear(osora_S * F.linear(osora_O * dropout(x), osora_V), osora_U)
                    else:
                        x = dropout(x)
                        result = result + self.osora_magnitude_vector[active_adapter](
                            x,
                            osora_U=osora_U,
                            osora_V=osora_V,
                            osora_S=osora_S,
                            osora_O=osora_O,
                            base_layer=self.get_base_layer(),
                        )

        result = result.to(previous_dtype)
        return result
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "osora." + rep