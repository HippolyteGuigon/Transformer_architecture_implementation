import torch
from abc import ABC, abstractmethod
from typing import List, Dict


class DataPreprocessor:
    """
    The goal of this class is to preprocess
    input sequences before they are fed into
    the model by creating a word index
    reference

    Arguments:
        -sentences: List[str]: The input sentences
    Returns:
        -None
    """

    def __init__(self, sentences: List[str]) -> None:
        self.sentences = sentences
        self.word_to_index = self._create_word_index()

    def _create_word_index(self) -> Dict[str, int]:
        """
        The goal of this method is to create
        a word index dictionnary which will
        then be used for sequence embedding

        Arguments:
            -None
        Returns:
            -word_to_index: Dict[str,int]: The
            dictionnary containg all indexes for
            the vocabulary
        """

        all_words = [w for vocab in self.sentences for w in vocab.split()]
        unique_words = list(set(all_words))
        word_to_index = {w: i for i, w in enumerate(unique_words)}

        return word_to_index

    def get_indices(self) -> torch.tensor:
        """
        The goal of this method is to
        get all corresponding indices
        for each word in the input sentences

        Arguments:
            -None
        Returns:
            -sentences_indices: torch.tensor: The
            indices of the different words
            composing the input sentences
        """

        indices = [
            [self.word_to_index[word] for word in sentence.split()]
            for sentence in self.sentences
        ]
        sentences_indices = torch.tensor(indices)

        return sentences_indices


class PositionalEncoding(ABC):
    """
    The goal of this class is to create an
    absctract class of Positional Encoding
    for the embedding part of the Transformer

    Arguments:
        -None
    Returns:
        -None
    """

    @abstractmethod
    def add_positional_encoding(self, data: torch.tensor) -> torch.tensor:
        """
        The goal of this abstract method is to
        compute positional encoding and add it
        to the input data

        Arguments:
            -data: torch.tensor: The input data
            to which positional encoding will
            be added
        Returns:
            -data: torch.tensor: The data once
            transformed with positional encoding
        """

        pass
