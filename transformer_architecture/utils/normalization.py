import torch
import torch.nn as nn
from torch import Tensor


class NormalizationLayer(nn.Module):
    """
    The goal of this class is to
    implement the Layer Normalization
    process involved in the Transformer
    Architecture

    Arguments:
        -normalized_shape: int: The
        shape of the element to be normalized
        -eps: Value added to the denominator
        for numerical stability
        -elementwise_affine: bool: If the normalization
        should have learnable parameter Gamma
        learnt during the process
        -bias: bool: Only relevant if elementwise_affine
        set to True, should a bias be learnt during the
        normalization process
    Returns:
        -None
    """

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
            elementwise_affine_dim = normalized_shape
            self.gamma = nn.Parameter(
                torch.ones((elementwise_affine_dim), requires_grad=True)
            )

            if self.bias:
                self.beta = nn.Parameter(
                    torch.zeros((elementwise_affine_dim), requires_grad=True)
                )

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

        mean = input.mean(dim=-1, keepdim=True)
        standard_deviation = input.std(dim=-1, keepdim=True, unbiased=False)

        normalized_input = (input - mean) / (standard_deviation + self.eps)

        if self.elementwise_affine:
            normalized_input = normalized_input * self.gamma
            if self.bias:
                normalized_input = normalized_input + self.beta

        return normalized_input
