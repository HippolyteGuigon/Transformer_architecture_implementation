import unittest
import torch
import nltk

from nltk.tokenize import sent_tokenize
from nltk.corpus import webtext

from transformer_architecture.preprocessing.embedding import (
    DataPreprocessor,
    Embedding,
    SinusoidalPositionalEncoding,
)
from transformer_architecture.utils.activation import softmax, relu, sigmoid
from transformer_architecture.model.attention import MultiHeadAttention
from transformer_architecture.model.encoder import TransformerEncoderLayer

nltk.download("punkt_tab")
nltk.download("webtext")


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_embedding_dimensions(self) -> None:
        """
        The goal of this test is to check
        if the embedding class produces
        the appropriate result from a
        set of sentences

        Arguments:
            -None
        Returns:
            -None
        """

        text = webtext.raw("pirates.txt")
        sentences = sent_tokenize(text)

        embedding_dim = 512

        number_sentences = len(sentences)
        longest_sentence_word = max(len(s.split()) for s in sentences)

        preprocessor = DataPreprocessor(sentences)
        embedder = Embedding(embedding_dim=embedding_dim)
        positionnal_encoding = SinusoidalPositionalEncoding(
            max_len=longest_sentence_word, embedding_dim=embedding_dim
        )
        positionnal_encoding._init_positional_encoding()

        sentence_indices = preprocessor.get_indices()
        embeddings = embedder.embed(sentence_indices)
        embeddings = positionnal_encoding.add_positional_encoding(embeddings)

        embedded_dim = embeddings.size()

        self.assertEqual(
            number_sentences,
            embedded_dim[0],
            "Mismatch in number of sentences",
        )
        self.assertEqual(
            longest_sentence_word,
            embedded_dim[1],
            "Mismatch in the length of the longest sentence",
        )
        self.assertEqual(
            embedding_dim, embedded_dim[2], "Mismatch in embedding dimensions"
        )

    def test_activation_function(self) -> None:
        """
        The goal of this function is to
        test if the activation functions
        return appropriate results (equal 1)
        when given a neuron

        Arguments:
            -None
        Returns:
            -None
        """

        test_size = 10000

        test_neuron = torch.randint(low=-50, high=50, size=(test_size, 10))
        valid_output = torch.ones(size=(test_size,))

        softmax_results = torch.sum(softmax(x=test_neuron, axis=1), dim=1)
        relu_results = relu(test_neuron)
        sigmoid_result = sigmoid(test_neuron)

        min_relu_results = torch.min(relu_results)
        is_valid_sigmoid = torch.allclose(softmax_results, valid_output)
        is_valid_sigmoid = torch.all(sigmoid_result <= 1)

        self.assertTrue(is_valid_sigmoid)
        self.assertTrue(is_valid_sigmoid)
        self.assertGreaterEqual(min_relu_results, 0)

    def test_self_attention_mechanism(self) -> None:
        """
        The goal of this test is to check if
        the self-attention mechanism that was
        implemented returns accurate results
        when given a random output

        Arguments:
            -None
        Returns:
            -None
        """

        embedding_dim = 512
        key_dimension = 32
        value_dimension = 16
        num_heads = 8

        embedder = Embedding(embedding_dim=embedding_dim)

        multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            d_k=key_dimension,
            d_v=value_dimension,
        )

        text = webtext.raw("pirates.txt")

        sentences = sent_tokenize(text)

        number_sentences = len(sentences)
        longest_sentence_word = max(len(s.split()) for s in sentences)

        preprocessor = DataPreprocessor(sentences)
        positionnal_encoding = SinusoidalPositionalEncoding(
            max_len=longest_sentence_word, embedding_dim=embedding_dim
        )
        positionnal_encoding._init_positional_encoding()

        sentence_indices = preprocessor.get_indices()
        embeddings = embedder.embed(sentence_indices)

        embeddings = positionnal_encoding.add_positional_encoding(embeddings)
        multi_head_attention._create_attention_matrices(embeddings)

        Q, K, V = multi_head_attention.split_heads()

        attention_output = multi_head_attention.forward(
            key=K, query=Q, value=V
        )

        attention_output_size = attention_output.size()

        self.assertEqual(
            number_sentences,
            attention_output_size[0],
            "Attention value output mismatches the\
                number of sentences in its final dimensions",
        )
        self.assertEqual(
            longest_sentence_word,
            attention_output_size[1],
            "Attention value output mismatches the\
                longest sentence word in its final dimensions",
        )
        self.assertEqual(
            num_heads * value_dimension,
            attention_output_size[2],
            "Attention value output mismatches the\
                number of heads and value dimension\
                    required",
        )

    def test_encoder_layer(self) -> None:
        """
        The goal of this test is to
        make sure that the encoder layer
        returns the appropriate outputs

        Arguments:
            -None
        Returns:
            -None
        """

        embedding_dim = 256
        num_heads = 8

        embedder = Embedding(embedding_dim=embedding_dim)

        text = webtext.raw("pirates.txt")

        sentences = sent_tokenize(text)

        number_sentences = len(sentences)
        longest_sentence_word = max(len(s.split()) for s in sentences)

        preprocessor = DataPreprocessor(sentences)

        sentence_indices = preprocessor.get_indices()
        embeddings = embedder.embed(sentence_indices)

        positionnal_encoding = SinusoidalPositionalEncoding(
            max_len=longest_sentence_word, embedding_dim=embedding_dim
        )
        positionnal_encoding._init_positional_encoding()
        embeddings = positionnal_encoding.add_positional_encoding(embeddings)

        encoder = TransformerEncoderLayer(
            d_model=embedding_dim, num_heads=num_heads, norm_first=True
        )

        output = encoder.forward(src=embeddings)
        output_size = output.size()

        self.assertEqual(
            number_sentences,
            output_size[0],
            "Encoder value output mismatches the\
                number of sentences in its final dimensions",
        )
        self.assertEqual(
            longest_sentence_word,
            output_size[1],
            "Encoder value output mismatches the\
                longest sentence word in its final dimensions",
        )
        self.assertEqual(
            embedding_dim,
            output_size[2],
            "Encoder value output mismatches the\
                number of heads and value dimension\
                    required",
        )


if __name__ == "__main__":
    unittest.main()
