import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from transformer_architecture.utils.activation import softmax


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
        self,
        key: torch.tensor,
        query: torch.tensor,
        value: torch.tensor,
        masking: bool = False,
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


class SelfAttention(Attention):
    """
    The goal of this class is to
    implement the self-attention
    mechanism

    Arguments:
        -None
    Returns:
        -None
    """

    def __init__(self) -> None:
        super(SelfAttention).__init__()

    def forward(
        self,
        key: torch.tensor,
        query: torch.tensor,
        value: torch.tensor,
        masking: bool = False,
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
            -attention_score: torch.tensor: The attention
            score output
        """

        d_k = key.size()[-1]

        dot_product = torch.matmul(query, key.transpose(-2, -1))
        scaled_dot_product = dot_product / torch.sqrt(d_k)
        attention_scores = softmax(scaled_dot_product, axis=-1)
        attention_scores = torch.matmul(attention_scores, value)

        return attention_scores
