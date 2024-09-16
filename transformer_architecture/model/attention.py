import math
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
        super().__init__()

    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        masking: bool = False,
    ) -> torch.Tensor:
        """
        The goal of this method is to calculate
        the self-attention scores for a given
        input

        Arguments:
            -embeddings: torch.Tensor: The embedding
            input
            -masking: bool: Wether the attention matrix
            is masked
        Returns:
            -attention_score: torch.Tensor: The results
            for the attention values
        """

        pass


class SelfAttention(Attention):
    """
    The goal of this class is to
    implement the self-attention
    mechanism

    Arguments:
        -embedding_dim: int: The
        dimension of the embedding
        input
        -d_k: int: The dimension of
        the key matrix
        -d_v: int: The dimension of
        the value matrix
    Returns:
        -None
    """

    def __init__(self, embedding_dim: int, d_k: int, d_v: int) -> None:
        super().__init__()
        self.query_layer = nn.Linear(embedding_dim, d_k)
        self.key_layer = nn.Linear(embedding_dim, d_k)
        self.value_layer = nn.Linear(embedding_dim, d_v)
        self.d_k = d_k

    def forward(
        self,
        embeddings: torch.Tensor,
        masking: bool = False,
    ) -> torch.Tensor:
        """
        The goal of this method is to calculate
        the self-attention scores for a given
        embedding input

        Arguments:
            -embeddings: torch.Tensor: The embedding
            input
            -masking: bool: Wether the attention matrix
            is masked
        Returns:
            -attention_score: torch.Tensor: The attention
            score output
        """

        query = self.query_layer(embeddings)
        key = self.key_layer(embeddings)
        value = self.value_layer(embeddings)

        dot_product = torch.matmul(query, key.transpose(-2, -1))
        scaled_dot_product = dot_product / math.sqrt(self.d_k)

        if masking:
            mask_size = key.size(-2)
            mask = torch.triu(
                torch.ones(mask_size, mask_size, device=key.device), diagonal=1
            )
            scaled_dot_product = scaled_dot_product.masked_fill(
                mask == 1, float("-inf")
            )

        attention_scores = softmax(scaled_dot_product, axis=-1)
        attention_scores = torch.matmul(attention_scores, value)

        return attention_scores


class MultiHeadAttention(SelfAttention):
    """
    The goal of this class is to
    implement the multi-head attention
    mechanism

    Arguments:
        -embedding_dim: int: The
        dimension of the embedding
        input
        -num_heads: int: The number of
        attention heads
        -d_k: int: The dimension of
        the key matrix
        -d_v: int: The dimension of
        the value matrix
    Returns:
        -None
    """

    def __init__(
        self, embedding_dim: int, num_heads: int, d_k: int, d_v: int
    ) -> None:
        super().__init__(embedding_dim, d_k, d_v)
        self.num_heads = num_heads
