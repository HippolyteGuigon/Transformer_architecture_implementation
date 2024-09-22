from typing import List


class NormalizationLayer:
    """
    The goal of this class is to
    implement the Layer Normalization
    process involved in the Transformer
    Architecture

    Arguments:
        -eps: Value added to the denominator
        for numerical stability
    Returns:
        -None
    """

    def __init__(
        self, normalized_shape: List[int], eps: float = 1e-05
    ) -> None:
        self.eps = eps
