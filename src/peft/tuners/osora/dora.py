# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch import nn

from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose


class DoraLinearLayer(nn.Module):
    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out

    def get_weight_norm(self, weight, osora_weight) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + osora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def update_layer(self, *, base_layer, osora_U, osora_V, osora_S, osora_O) -> None:
        with gather_params_ctx(base_layer.parameters()):
            weight = dequantize_module_weight(base_layer)
            osora_weight = torch.diag(osora_O) @ osora_U @ torch.diag(osora_S) @ osora_V

            weight_norm = self.get_weight_norm(weight.to(osora_V.device), osora_weight)

        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x, *, osora_U, osora_V, osora_S, osora_O, base_layer):
        """
        For DoRA, calculate the extra output from OSoRA with DoRA applied. 
        This should be added on top of the base layer output.
        """
        osora_result = torch.diag(osora_O)(osora_U(torch.diag(osora_S)(osora_V(x))))

        x_eye = torch.eye(osora_V.weight.shape[1], device=osora_V.weight.device, dtype=x.dtype)
        osora_weight = torch.diag(osora_O)(osora_U(torch.diag(osora_S)(osora_V(x_eye)))).T

        magnitude = self.weight
        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)
        weight_norm = self.get_weight_norm(weight, osora_weight.detach())
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        result_dora = (mag_norm_scale - 1) * (
            F.linear(x, transpose(weight, self.fan_in_fan_out))
        ) + mag_norm_scale * osora_result

        # Note: Computation could potentially be accelerated by using the code below instead of calculating X@W again.
        # This is only correct if dropout=0, otherwise results will differ:
        # https://github.com/huggingface/peft/pull/1474#issuecomment-1964682771
        # bias = self.get_base_layer().bias
        # if bias is not None:
        #     result = result - bias
        # result = mag_norm_scale * result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        # if bias is not None:
        #     result = result + bias

        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "osora.dora." + rep