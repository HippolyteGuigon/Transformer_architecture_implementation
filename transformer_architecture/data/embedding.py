import torch
from abc import ABC, abstractmethod


class PositionalEncoding(ABC):
    """
    The goal of this class is to create an
    absctract class of Positional Encoding
    for the embedding part of the Transformer

    Arguments:
        -None
    Returns:
        -None
    """

    @abstractmethod
    def add_positional_encoding(self, data: torch.tensor) -> torch.tensor:
        """
        The goal of this abstract method is to
        compute positional encoding and add it
        to the input data

        Arguments:
            -data: torch.tensor: The input data
            to which positional encoding will
            be added
        Returns:
            -data: torch.tensor: The data once
            transformed with positional encoding
        """

        pass
