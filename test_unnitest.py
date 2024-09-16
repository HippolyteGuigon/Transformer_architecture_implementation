import unittest
import torch
import nltk

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
from nltk.corpus import webtext

from transformer_architecture.preprocessing.embedding import (
    DataPreprocessor,
    Embedding,
)
from transformer_architecture.utils.activation import softmax
from transformer_architecture.model.attention import MultiHeadAttention

nltk.download("punkt_tab")
nltk.download("punkt")


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

        nltk.download("webtext")

        text = webtext.raw("pirates.txt")
        sentences = sent_tokenize(text)

        embedding_dim = 512

        number_sentences = len(sentences)
        longest_sentence_word = max(len(s.split()) for s in sentences)

        preprocessor = DataPreprocessor(sentences)
        embedder = Embedding(embedding_dim=embedding_dim)

        sentence_indices = preprocessor.get_indices()
        embeddings = embedder.embed(sentence_indices)

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

        test_neuron = torch.rand(size=(test_size, 10))
        valid_output = torch.ones(size=(test_size,))

        softmax_results = torch.sum(softmax(x=test_neuron, axis=1), dim=1)

        is_valid = torch.allclose(softmax_results, valid_output)

        self.assertTrue(is_valid)

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
        key_dimension = 100
        value_dimension = 512
        num_heads = 64

        embedder = Embedding(embedding_dim=embedding_dim)

        multi_head_attention = MultiHeadAttention(
            embedding_dim, num_heads, key_dimension, value_dimension
        )

        text = gutenberg.raw("austen-emma.txt")

        sentences = sent_tokenize(text)

        number_sentences = len(sentences)
        longest_sentence_word = max(len(s.split()) for s in sentences)

        preprocessor = DataPreprocessor(sentences)

        sentence_indices = preprocessor.get_indices()
        embeddings = embedder.embed(sentence_indices)

        multi_head_attention._create_attention_matrices(embeddings)

        attention_output = multi_head_attention.forward(
            multi_head_attention.key,
            multi_head_attention.query,
            multi_head_attention.value,
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
            embedding_dim,
            attention_output_size[2],
            "Attention value output mismatches the\
                input embedding dimension in its final dimensions",
        )


if __name__ == "__main__":
    unittest.main()
