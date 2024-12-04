import torch
import torch.nn as nn
import torchtext
import gc

from typing import Optional

from transformer_architecture.model.encoder import TransformerEncoderLayer
from transformer_architecture.model.decoder import TransformerDecoderLayer

from transformer_architecture.preprocessing.embedding import (
    Embedding,
    SinusoidalPositionalEncoding,
    LearnablePositionnalEncoding,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerWithProjection(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        vocab_size_en: int,
        max_len: int,
        learnable_encoding: bool = False,
    ) -> None:
        """
        Initializes the Transformer model with embedding, encoder, decoder,
        and projection layers.

        Arguments:
            - embedding_dim: int: The size of the embedding
            vectors.
            - num_heads: int: The number of attention heads.
            - vocab_size_en: int: The vocabulary size for the
            target language.
            - max_len: int: The maximum length of input sequences.
            - learnable_encoding: bool: Whether to use learnable
            positional encoding.
        Returns:
            -None
        """

        super(TransformerWithProjection, self).__init__()

        self.embedder = Embedding(embedding_dim=embedding_dim).to(device)
        self.encoder = TransformerEncoderLayer(
            d_model=embedding_dim, num_heads=num_heads, norm_first=True
        ).to(device)

        self.decoder = TransformerDecoderLayer(
            d_model=embedding_dim, num_heads=num_heads, norm_first=True
        ).to(device)
        self.projection = nn.Linear(embedding_dim, vocab_size_en).to(device)
        self.parameters_to_optimize = list(self.parameters())

        if learnable_encoding:
            self.positionnal_encoding = LearnablePositionnalEncoding(
                max_len=max_len, embedding_dim=embedding_dim
            )
            self.parameters_to_optimize += [self.positionnal_encoding.pe]
        else:
            self.positionnal_encoding = SinusoidalPositionalEncoding(
                max_len=max_len, embedding_dim=embedding_dim
            )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Processes the input and target sequences through the Transformer model.

        Arguments:
            - src: torch.Tensor: The input sequence tensor.
            - tgt: torch.Tensor: The target sequence tensor.
            - tgt_mask: Optional[torch.Tensor]: The mask for the
            target sequence.
            - memory_mask: Optional[torch.Tensor]: The mask for the
            encoder's memory.

        Returns:
            - projected_output: torch.Tensor: The output tensor
            projected to the target vocabulary size.
        """

        src_emb = self.embedder.embed(src)
        tgt_emb = self.embedder.embed(tgt)
        src = self.positionnal_encoding.add_positional_encoding(src_emb)
        tgt = self.positionnal_encoding.add_positional_encoding(tgt_emb)
        encoder_output = self.encoder(src=src)
        del src_emb, tgt_emb, src
        torch.cuda.empty_cache()
        gc.collect()
        decoder_output = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return self.projection(decoder_output)


def print_memory_stats(stage: str) -> None:
    """
    Prints the GPU memory usage statistics at a specific stage of execution.

    Arguments:
        - stage: str: The name or description of the stage being monitored.

    Returns:
        - None
    """
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    free = torch.cuda.get_device_properties(0).total_memory / 1e9 - reserved
    print(
        f"[{stage}] GPU Memory - Allocated: {allocated:.2f}\
              GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB"
    )


def translate_sentence(
    sentence: str,
    model: TransformerWithProjection,
    vocab_fr: torchtext.vocab.Vocab,
    vocab_en: torchtext.vocab.Vocab,
    tokenizer_fr: torchtext.data.utils,
    max_len: int,
    device: torch.device = device,
) -> str:
    """
    Translates a French sentence into English using the Transformer model.

    Arguments:
        - sentence: str: The French sentence to be translated.
        - model: TransformerWithProjection: The trained
        Transformer model.
        - vocab_fr: torchtext.vocab.Vocab: The vocabulary for
        French.
        - vocab_en: torchtext.vocab.Vocab: The vocabulary for
        English.
        - tokenizer_fr: torchtext.data.utils: The tokenizer for
        French sentences.
        - max_len: int: The maximum length of the output sequence.
        - device: torch.device: The device to perform computations
        on (default: cuda or cpu).

    Returns:
        - str: The translated English sentence.
    """
    tokens = (
        [vocab_fr["<bos>"]]
        + [vocab_fr[token] for token in tokenizer_fr(sentence)]
        + [vocab_fr["<eos>"]]
    )
    input_tensor = (
        torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    )

    if input_tensor.size(1) < max_len:
        padding = torch.full(
            (1, max_len - input_tensor.size(1)),
            vocab_fr["<pad>"],
            dtype=torch.long,
            device=device,
        )
        input_tensor = torch.cat([input_tensor, padding], dim=1)
    elif input_tensor.size(1) > max_len:
        input_tensor = input_tensor[:, :max_len]

    target_tensor = torch.tensor(
        [vocab_en["<bos>"]], dtype=torch.long, device=device
    ).unsqueeze(0)

    input_tensor = input_tensor.to(device=device)
    target_tensor = target_tensor.to(device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            src_emb = model.positionnal_encoding.add_positional_encoding(
                model.embedder.embed(input_tensor.to(device=device))
            )
            tgt_emb = model.embedder.embed(target_tensor.to(device=device))
            tgt_emb = tgt_emb + model.positionnal_encoding.pe[
                : tgt_emb.size(1), :
            ].unsqueeze(0)
            encoder_output = model.encoder(src_emb)
            decoder_output = model.decoder(tgt=tgt_emb, memory=encoder_output)
            output_logits = model.projection(decoder_output)

            next_token = output_logits[:, -1, :].argmax(dim=-1).item()
            target_tensor = torch.cat(
                [target_tensor, torch.tensor([[next_token]], device=device)],
                dim=1,
            )

            if next_token == vocab_en["<eos>"]:
                break

    translated_tokens = target_tensor.squeeze().tolist()
    translated_sentence = " ".join(
        vocab_en.lookup_token(idx)
        for idx in translated_tokens
        if idx not in [vocab_en["<bos>"], vocab_en["<eos>"], vocab_en["<pad>"]]
    )
    return translated_sentence
