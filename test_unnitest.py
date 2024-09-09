import unittest

from transformer_architecture.data.embedding import DataPreprocessor, Embedding


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


if __name__ == "__main__":
    unittest.main()
