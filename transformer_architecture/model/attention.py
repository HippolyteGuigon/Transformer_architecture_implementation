import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class Attention(ABC, nn.Module):
    """
    The goal of this class is to
    create an abstraction for all
    type of attention that will then
    be implemented (self-attention,
    MultiHeadAttention)

    Arguments:
        -None
    Returns:
        -None
    """

    def __init__(self) -> None:
        super(Attention, self).__init__()

    @abstractmethod
    def forward(
        self, key: torch.tensor, 
            query: torch.tensor, 
            value: torch.tensor,
            masking: bool=False
    ) -> torch.tensor:
        """
        The goal of this method is to calculate
        the self-attention scores for a given
        input

        Arguments:
            -key: torch.tensor: The key tensor of
            the input
            -query: torch.tensor: The query tensor
            of the input
            -value: torch.tensor: The value tensor
            of the input
            -masking: bool: Wether the attention matrix 
            is masked 
        Returns:
            -attention_score: torch.tensor: The results
            for the attention values
        """

        pass
