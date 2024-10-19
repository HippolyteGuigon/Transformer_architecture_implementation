import torch
import torch.nn as nn
from torch import Tensor


class NormalizationLayer(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.bias = bias

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            if self.bias:
                self.beta = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.beta = None
        else:
            self.gamma = None
            self.beta = None

    def forward(self, input: Tensor) -> Tensor:
        """
        The goal of this method is to implement the
        normalization layer process to an input
        tensor

        Arguments:
            -input: Tensor: The input Tensor to
            be normalized
        Returns:
            -normalized_layer: Tensor: The input
            after it was normalized
        """

        mean = torch.mean(input, dim=-1, keepdim=True)
        variance = torch.var(input, dim=-1, keepdim=True, unbiased=False)
        standard_deviation = torch.sqrt(variance + self.eps)
        normalized_input = (input - mean) / (standard_deviation)

        if self.elementwise_affine:
            normalized_input = normalized_input * self.gamma
            if self.bias:
                normalized_input = normalized_input + self.beta

        return normalized_input
