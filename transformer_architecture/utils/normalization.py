import torch

from typing import List


class NormalizationLayer:
    """
    The goal of this class is to
    implement the Layer Normalization
    process involved in the Transformer
    Architecture

    Arguments:
        -normalized_shape: List[int]: The
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
        normalized_shape: List[int],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        self.eps = eps

        self.elementwise_affine = elementwise_affine
        self.bias = bias

        if self.elementwise_affine:
            elementwise_affine_dim = normalized_shape[-1]
            self.gamma = torch.ones(
                (elementwise_affine_dim), requires_grad=True
            )

            if self.bias:
                self.beta = torch.zeros(
                    (elementwise_affine_dim), requires_grad=True
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The goal of this method is to implement the
        normalization layer process to an input
        tensor

        Arguments:
            -input: torch.Tensor: The input Tensor to
            be normalized
        Returns:
            -normalized_layer: torch.Tensor: The input
            after it was normalized
        """

        mean = torch.mean(input, dim=-1, keepdim=True)
        standard_deviation = torch.std(
            input, dim=-1, keepdim=True, unbiased=False
        )

        normalized_input = (input - mean) / (standard_deviation + self.eps)

        if self.elementwise_affine:
            normalized_input *= self.gamma
            if self.bias:
                normalized_input += self.beta

        return normalized_input
