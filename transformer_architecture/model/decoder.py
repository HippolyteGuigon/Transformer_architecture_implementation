import torch.nn as nn

from torch import Tensor
from typing import Callable, Optional

from transformer_architecture.model.attention import MultiHeadAttention
from transformer_architecture.utils.activation import relu
from transformer_architecture.utils.normalization import NormalizationLayer
from transformer_architecture.utils.residual_connexion import (
    ResidualConnection,
)


class TransformerDecoderLayer(MultiHeadAttention):
    """
    The goal of this class is to
    implement the Transformer decoder
    Layer

    Arguments:
        -d_model: int: The dimension of the
        input sequence (the embedding)
        -num_heads: int: The number of attention
        heads of the model
        -dim_feedforward: int: The dimension of the
        feedforward network model
        -dropout: float: The dropout value
        -activation: Callable[[Tensor], Tensor]: The
        chosen activation function for the feed forward
        neural network
        -layer_norm_eps: float: The eps value in layer
        normalization components
        -batch_first: bool: If True, then the input and
        output tensors are provided as (batch, seq, feature)
        -norm_first: bool: If True, layer norm is done prior to
        attention and feedforward operations, respectively
        -biais:  If set to False, Linear and LayerNorm layers
        will not learn an additive bias
    Returns:
        -None
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = self.d_model // self.num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_q = self.d_model // self.num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.bias = bias

        super().__init__(
            embedding_dim=self.d_model,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
        )

        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)

        self.norm1 = NormalizationLayer(normalized_shape=d_model)
        self.norm2 = NormalizationLayer(normalized_shape=d_model)

        self.residual1 = ResidualConnection(
            in_dimensions=d_model, out_dimensions=self.num_heads * self.d_v
        )
        self.residual2 = ResidualConnection(
            in_dimensions=d_model, out_dimensions=self.num_heads * self.d_v
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_casual: bool = False,
        memory_is_casual: bool = False,
    ) -> Tensor:
        """
        The goal of this method is
        to pass the input through
        the decodef layer

        Arguments:
            -tgt: Tensor: The sequence
            to the encoder layer
            -memory: Optional[Tensor]:
            The sequence from the last layer
            of the decoder
            -src_key_padding_mask: Optional[Tensor]:
            The mask for the tgt sequence
            -tgt_key_padding_mask: Optional[Tensor]: The
            mask for the tgt keys per batch
            -memory_key_padding_mask: Optional[Tensor]: The
            mask for the memory keys per batch
            -tgt_is_causal: bool: If specified, applies a causal
            mask as tgt mask.
            -memory_is_causal: bool: If specified, applies a causal
            mask as memory mask
        """

        super()._create_attention_matrices(tgt)
        Q, K, V = super().split_heads()
        attention_output = super().forward(
            key=K, query=Q, value=V, masking=True
        )

        if self.norm_first:
            attention_output = self.norm1.forward(attention_output)
            attention_output = self.residual1.forward(
                X=tgt, output=attention_output
            )
        else:
            attention_output = self.residual1.forward(
                X=tgt, output=attention_output
            )
            attention_output = self.norm1.forward(attention_output)

        cross_attention_output = super()._cross_attention(
            query=tgt, key=memory, value=memory
        )

        cross_attention_output = super()._cross_attention(
            query=tgt, key=memory, value=memory
        )

        if self.norm_first:
            cross_attention_output = self.norm2.forward(attention_output)
            cross_attention_output = self.residual2.forward(
                X=tgt, output=cross_attention_output
            )
        else:
            cross_attention_output = self.residual2.forward(
                X=tgt, output=cross_attention_output
            )
            cross_attention_output = self.norm2.forward(cross_attention_output)

        return cross_attention_output
