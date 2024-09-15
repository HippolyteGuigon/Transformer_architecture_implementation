import unittest
import torch

from transformer_architecture.preprocessing.embedding import (
    DataPreprocessor,
    Embedding,
)
from transformer_architecture.utils.activation import softmax


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

        sentences = [
            "I love cats",
            "I hate dogs and birds",
            "Cats are cute",
            "Dogs are loyal",
        ]

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


if __name__ == "__main__":
    unittest.main()
