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

        if elementwise_affine:
            elementwise_affine_dim = normalized_shape[-1]
            self.gamma = torch.randn(
                (elementwise_affine_dim), requires_grad=True
            )

            if bias:
                self.beta = torch.randn(
                    (elementwise_affine_dim), requires_grad=True
                )
