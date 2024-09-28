from torch import Tensor
from typing import Callable, Optional

from transformer_architecture.model.attention import MultiHeadAttention
from transformer_architecture.utils.normalization import NormalizationLayer
from transformer_architecture.utils.residual_connexion import (
    ResidualConnection,
)
from transformer_architecture.utils.activation import relu


class TransformerEncoderLayer(MultiHeadAttention):
    """
    The goal of this class is to
    implement the Transformer encoder
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

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_casual: bool = False,
    ) -> Tensor:
        """
        The goal of this method is
        to pass the input through
        the encoder layer

        Arguments:
            -src: Tensor: The sequence
            to the encoder layer
            -src_mask: Optional[Tensor]:
            The mask for the src sequence
            -src_key_padding_mask: Optional[Tensor]:
            The mask for the src keys per batch
            -is_causal: bool: If specified, applies
            a causal mask as src mask
        """

        super()._create_attention_matrices(src)
        Q, K, V = super().split_heads()
        attention_output = super().forward(key=K, query=Q, value=V)

        residual_connexion = ResidualConnection(
            in_dimensions=self.d_model,
            out_dimensions=self.num_heads * self.d_v,
        )
        attention_output = residual_connexion.forward(
            X=src, output=attention_output
        )

        input_shape = list(attention_output.size())
        normalisation = NormalizationLayer(normalized_shape=input_shape)

        attention_output = normalisation.forward(attention_output)

        return attention_output
