from transformer_architecture.model.attention import MultiHeadAttention


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
    Returns:
        -None
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_v = self.d_model // self.num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_q = self.d_model // self.num_heads

        super().__init__(
            embedding_dim=self.d_model,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
        )
