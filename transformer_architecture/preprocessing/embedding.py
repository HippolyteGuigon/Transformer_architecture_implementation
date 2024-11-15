import math
import torch
import torch.nn as nn

import warnings

from torch import Tensor
from abc import ABC, abstractmethod
from typing import List, Dict

warnings.filterwarnings("ignore")


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
        self.max_len = max(len(sentence.split()) for sentence in sentences)

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
        word_to_index = {w: i + 1 for i, w in enumerate(unique_words)}
        word_to_index["<padding>"] = 0

        return word_to_index

    def get_indices(self) -> Tensor:
        """
        The goal of this method is to
        get all corresponding indices
        for each word in the input sentences

        Arguments:
            -None
        Returns:
            -sentences_indices: Tensor: The
            indices of the different words
            composing the input sentences
        """

        indices = [
            [self.word_to_index[word] for word in sentence.split()]
            + [self.word_to_index["<padding>"]]
            * (self.max_len - len(sentence.split()))
            for sentence in self.sentences
        ]
        sentences_indices = torch.tensor(indices, dtype=torch.long)

        return sentences_indices


class Embedding(nn.Module):
    """
    The goal of this class is to embed
    the input sequences before they are
    fed into the model

    Arguments:
        -num_embeddings: int: The total
        vocabulary size
        -embedding_dim: int: The dimension
        of the embedding
    Returns:
        -None
    """

    def __init__(
        self, num_embeddings: int = 1000000, embedding_dim: int = 512
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

    def embed(self, sentences_indices: Tensor) -> Tensor:
        """
        The goal of this method is to
        embed sequences once word index
        has been created

        Arguments:
            -sentences_indices: Tensor: The
            sentence indices which are to be embedded
        Returns:
            -embedded: Tensor: The
            embedding tensor
        """

        embedded = self.embedding(sentences_indices)

        return embedded


class PositionalEncoding(ABC):
    """
    The goal of this class is to create an
    absctract class of Positional Encoding
    for the embedding part of the Transformer

    Arguments:
        -max_len: int: The maximum size of the
        input sequence
        -embedding_dim: int: The embedding size of
        the input sequence
    Returns:
        -None
    """

    def __init__(self, max_len: int, embedding_dim: int) -> None:
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.pe = self._init_positional_encoding()

    @abstractmethod
    def _init_positional_encoding(self) -> Tensor:
        """
        The goal of this class is to
        compute the positional encoding
        before applying it to the embedded
        sequence

        Arguments:
            -None
        Returns:
            -pe: Tensor: The positional
            encoding of the tensor
        """

        pass

    @abstractmethod
    def add_positional_encoding(self, data: Tensor) -> Tensor:
        """
        The goal of this abstract method is to
        add positional encoding to the input
        sequence

        Arguments:
            -data: Tensor: The input data
            to which positional encoding will
            be added
        Returns:
            -encoded_data: Tensor: The data once
            transformed with positional encoding
        """

        pass


class RotaryPositionnalEncoding(nn.Module):
    """
    The goal of this class is to implement
    the Rotary Positionnal encoding to key
    and query matrices once they are computed

    Arguments:
        -d_model :int: Embedding dimension
        -max_len :int: Maximum length of input
        sequence
    Returns:
        -None
    """

    def __init__(self, d_model, max_len=512):
        super(RotaryPositionnalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        self.div_term = div_term

    def forward(self, seq_len, device="cpu") -> torch.Tensor:
        """
        The goal of this function is applying
        RoPE to Key and Query matrices

        Args:
            seq_len: int: The length of the sequence to be
            processed
            device: str: device on which calculus should be
            made

        Returns:
            sinusoids: torch.Tensor: Rotation matrix for
            key and queries
        """

        position = self.position[:seq_len, :]
        div_term = self.div_term.unsqueeze(0)

        sinusoids = torch.cat(
            [torch.sin(position * div_term), torch.cos(position * div_term)],
            dim=-1,
        )
        sinusoids = sinusoids.to(device)

        return sinusoids


class SinusoidalPositionalEncoding(PositionalEncoding):
    """
    The goal of this class is to implement
    the sinusoidal positionnal encoding to
    a sequence once it has been embedded

    Arguments:
        -None
    Returns:
        -None
    """

    def __init__(self, max_len: int, embedding_dim: int) -> None:
        super().__init__(max_len=max_len, embedding_dim=embedding_dim)
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def _init_positional_encoding(self) -> Tensor:
        """
        The goal of this method is to compute
        the sinusoidal positional encoding and
        add it to the data

        Arguments:
            -None
        Returns:
            -pe: Tensor: The positional
            encoding tensor
        """

        with torch.no_grad():
            position = torch.arange(0, self.max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.embedding_dim, 2)
                * (-math.log(10000) / self.embedding_dim)
            )
            pe = torch.zeros(size=(self.max_len, self.embedding_dim))
            pe[:, ::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            del position, div_term

        return pe

    def add_positional_encoding(self, data: Tensor) -> Tensor:
        """
        The goal of this abstract method is to
        add positional encoding to the input
        sequence

        Arguments:
            -data: Tensor: The input data
            to which positional encoding will
            be added
        Returns:
            -encoded_data: Tensor: The data once
            transformed with positional encoding
        """

        encoded_data = data + self.pe

        return encoded_data


class LearnablePositionnalEncoding(PositionalEncoding):
    """
    The goal of this class is to implement
    positionnal encoding that will be learnt
    during the training process through the
    backpropagation process

    Arguments:
        -None
    Returns:
        -None
    """

    def __init__(self, max_len: int, embedding_dim: int) -> None:
        super().__init__(max_len=max_len, embedding_dim=embedding_dim)
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def _init_positional_encoding(self) -> Tensor:
        """
        The goal of this method is to compute
        the learnable positional encoding and
        add it to the data

        Arguments:
            -None
        Returns:
            -pe: Tensor: The positional
            encoding tensor
        """

        pe = nn.Parameter(
            torch.normal(
                mean=0, std=1, size=(self.max_len, self.embedding_dim)
            ),
            requires_grad=True,
        )

        return pe

    def add_positional_encoding(self, data: Tensor) -> Tensor:
        """
        The goal of this abstract method is to
        add positional encoding to the input
        sequence

        Arguments:
            -data: Tensor: The input data
            to which positional encoding will
            be added
        Returns:
            -encoded_data: Tensor: The data once
            transformed with positional encoding
        """

        encoded_data = data + self.pe

        return encoded_data
