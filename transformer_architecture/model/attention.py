import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple
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
        embeddings: Tensor,
        masking: bool = False,
    ) -> Tensor:
        """
        The goal of this method is to calculate
        the self-attention scores for a given
        input

        Arguments:
            -embeddings: Tensor: The embedding
            input
            -masking: bool: Wether the attention matrix
            is masked
        Returns:
            -attention_score: Tensor: The results
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
        super().__init__()

    def forward(
        self,
        key: Tensor,
        query: Tensor,
        value: Tensor,
        masking: bool = False,
    ) -> Tensor:
        """
        The goal of this method is to calculate
        the self-attention scores for a given
        embedding input

        Arguments:
            -key: Tensor: The key matrix of the
            attention head
            -query: Tensor: The query matix of the
            attention head
            -value: Tensor: The value matrix of the
            attention head
            -masking: bool: Wether the attention matrix
            is masked
        Returns:
            -attention_score: Tensor: The attention
            score output
        """

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
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.query_layer = nn.Linear(embedding_dim, d_k * num_heads)
        self.key_layer = nn.Linear(embedding_dim, d_k * num_heads)
        self.value_layer = nn.Linear(embedding_dim, d_v * num_heads)

        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        assert (
            self.embedding_dim % self.num_heads == 0
        ), "The number of heads must be divisible\
            by the dimension of the embedding"

    def _create_attention_matrices(self, embeddings: Tensor) -> None:
        """
        The goal of this method is to create
        the key, query and value matrices
        from the input embeddings with a
        projection in a lower dimension
        space

        Arguments:
            -embeddings: Tensor: The
            input embeddings
        Returns
            None
        """

        self.query = self.query_layer(embeddings).to(self.device)
        self.key = self.key_layer(embeddings).to(self.device)
        self.value = self.value_layer(embeddings).to(self.device)

    def split_heads(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        The goal of this method is to
        split the key, query, value
        tensor between the different
        attention heads

        Arguments:
            -None
        Returns:
            -Q_heads: Tensor: Splitted
            query attention head
            -K_heads: Tensor: Splitted
            key attention head
            -V_heads: Splitted value attention
            head
        """

        batch_size, seq_len, _ = self.query.size()

        Q_heads = self.query.view(
            batch_size, self.num_heads, seq_len, self.d_k
        )
        K_heads = self.key.view(batch_size, self.num_heads, seq_len, self.d_k)
        V_heads = self.value.view(
            batch_size, self.num_heads, seq_len, self.d_v
        )

        return Q_heads, K_heads, V_heads

    def forward(self, key, query, value) -> Tensor:
        """
        The goal of this function is to
        compute the self-attention score
        for all attention heads before
        concatenating the result

        Arguments:
            -key: Tensor: The key
            matrices of the attention
            heads
            -query: Tensor: The
            query matrices of the attention
            heads
            -value: Tensor: The value
            matrices of the attention heads
        Returns:
            -attention_scores: Tensor:
            The concatenated results of the
            attention score for each attention
            head
        """

        attention_scores = super().forward(key, query, value)

        batch_size, _, seq_len, _ = attention_scores.size()

        attention_scores = attention_scores.view(
            batch_size, seq_len, self.num_heads * self.d_v
        )

        return attention_scores
