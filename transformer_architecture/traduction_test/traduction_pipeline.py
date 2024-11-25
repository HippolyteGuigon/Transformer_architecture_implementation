import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchtext
import pickle
import json
import logging
import subprocess
import os
import gc
from typing import List, Tuple, Optional
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import build_vocab_from_iterator
from logging.handlers import RotatingFileHandler
from transformer_architecture.model.encoder import TransformerEncoderLayer
from transformer_architecture.model.decoder import TransformerDecoderLayer
from transformer_architecture.configs.confs import load_conf, clean_params
from transformer_architecture.preprocessing.embedding import (
    Embedding,
    SinusoidalPositionalEncoding,
    LearnablePositionnalEncoding,
)
from torch.utils.data.dataset import Subset
from torch.utils.checkpoint import checkpoint
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ[
    "PYTORCH_CUDA_ALLOC_CONF"
] = "garbage_collection_threshold:0.6,max_split_size_mb:32"
main_params = load_conf(include=True)
main_params = clean_params(main_params)

learning_rate = main_params["learning_rate"]
num_epochs = main_params["num_epochs"]
embedding_dim = main_params["embedding_dim"]
num_heads = main_params["num_heads"]
batch_size = main_params["batch_size"]
train_size = main_params["train_size"]
nrows = main_params["nrows"]
max_len = main_params["max_len"]

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(stream_format)

file_handler = RotatingFileHandler(
    "warnings.log", mode="a", maxBytes=5 * 1024 * 1024, backupCount=2
)
file_handler.setLevel(logging.WARNING)
file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_format)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

subprocess.run(
    ["python", "-m", "spacy", "download", "fr_core_news_sm"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
subprocess.run(
    ["python", "-m", "spacy", "download", "en_core_web_sm"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

tokenizer_fr = get_tokenizer("spacy", language="fr_core_news_sm")
tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")

logging.info("Loading dataset...")
df = pd.read_csv("data/en-fr.csv", nrows=nrows)
df.dropna(subset=["en", "fr"], inplace=True)
logging.info(f"Dataset loaded with {df.shape[0]}!")
logging.info("Filtering dataset...")
df["sentence_length"] = df["fr"].apply(lambda x: len(x.split(" ")))
df = df[df["sentence_length"] < max_len]
df.drop("sentence_length", axis=1, inplace=True)
logging.info(f"Succesfully filtered ! New shape of {df.shape[0]}")


def print_memory_stats(stage):
    allocated = torch.cuda.memory_allocated() / 1e9  # Convertir en GB
    reserved = torch.cuda.memory_reserved() / 1e9  # Convertir en GB
    free = torch.cuda.get_device_properties(0).total_memory / 1e9 - reserved
    print(
        f"[{stage}] GPU Memory - Allocated: {allocated:.2f}\
              GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB"
    )


def get_corpus_max_len(
    data_sample: List[Tuple[torch.Tensor, torch.Tensor]]
) -> int:
    """
    Computes the maximum length of sequences in the dataset.

    Arguments:
        - data_sample: List[Tuple[torch.Tensor, torch.Tensor]]:
        A list of tokenized sentence pairs.

    Returns:
        - max_length: int: The maximum length of
        sequences found in the dataset.
    """

    fr_max_len = max(len(s[0]) for s in data_sample)
    en_max_len = max(len(s[1]) for s in data_sample)
    return max(fr_max_len, en_max_len)


def build_vocab(df: pd.DataFrame) -> torchtext.vocab.Vocab:
    """
    Builds a vocabulary object from tokenized sentence pairs.

    Arguments:
        - df: pd.DataFrame: The dataset containing
        sentence pairs in two languages.

    Returns:
        - vocab: Vocab: A vocabulary object with
        special tokens initialized.
    """

    def tokenize_pair(data):
        for _, row in data.iterrows():
            if isinstance(row["fr"], str) and isinstance(row["en"], str):
                yield tokenizer_fr(row["fr"])
                yield tokenizer_en(row["en"])

    vocab = build_vocab_from_iterator(
        tokenize_pair(df),
        specials=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def pad_sentences(
    fr_batch: List[torch.Tensor],
    en_batch: List[torch.Tensor],
    max_len: int = max_len,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads sentences in French and English batches to the same maximum length.

    Arguments:
        - fr_batch: List[torch.Tensor]: The list of French
        sentences represented as tensors.
        - en_batch: List[torch.Tensor]: The list of English
        sentences represented as tensors.
        - max_len: int: The maximum sequence length to
        pad sentences.

    Returns:
        - Tuple[torch.Tensor, torch.Tensor]: A tuple containing
        padded French and English sentences as tensors.
    """
    fr_batch_padded = [
        torch.cat(
            [
                sentence[:max_len],  # Truncate if longer than max_len
                torch.full((max_len - len(sentence),), vocab_fr["<pad>"]),
            ]
        )
        if len(sentence) < max_len
        else sentence[:max_len]
        for sentence in fr_batch
    ]

    en_batch_padded = [
        torch.cat(
            [
                sentence[:max_len],  # Truncate if longer than max_len
                torch.full((max_len - len(sentence),), vocab_en["<pad>"]),
            ]
        )
        if len(sentence) < max_len
        else sentence[:max_len]
        for sentence in en_batch
    ]

    return torch.stack(fr_batch_padded).to(device), torch.stack(
        en_batch_padded
    ).to(device)


if os.path.exists("data/vocab_fr.pkl") and os.path.exists("data/vocab_en.pkl"):
    with open("data/vocab_fr.pkl", "rb") as f:
        vocab_fr = pickle.load(f)
    with open("data/vocab_en.pkl", "rb") as f:
        vocab_en = pickle.load(f)
else:
    logging.info("Building vocab...")

    vocab_fr = build_vocab(df)
    vocab_en = build_vocab(df)

    with open("data/vocab_fr.pkl", "wb") as f:
        pickle.dump(vocab_fr, f)
    with open("data/vocab_en.pkl", "wb") as f:
        pickle.dump(vocab_en, f)

    logging.info("Vocab succesfully built !")


def tokenize_sentence_pair(
    item: pd.Series,
    vocab_fr: torchtext.vocab.Vocab,
    vocab_en: torchtext.vocab.Vocab,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes and converts sentence pairs into tensors of vocabulary indices.

    Arguments:
        - item: pd.Series: A row from the dataset containing
        French and English sentences.
        - vocab_fr: Vocab: The vocabulary object for the
        French language.
        - vocab_en: Vocab: The vocabulary object for the
        English language.

    Returns:
        - fr_tokens: torch.Tensor: A tensor of vocabulary indices
        for the French sentence.
        - en_tokens: torch.Tensor: A tensor of vocabulary indices
        for the English sentence.
    """
    fr_tokens = (
        [vocab_fr["<bos>"]]
        + [vocab_fr[token] for token in tokenizer_fr(item["fr"])]
        + [vocab_fr["<eos>"]]
    )
    en_tokens = (
        [vocab_en["<bos>"]]
        + [vocab_en[token] for token in tokenizer_en(item["en"])]
        + [vocab_en["<eos>"]]
    )
    return torch.tensor(fr_tokens), torch.tensor(en_tokens)


def preprocess_data(
    df: pd.DataFrame,
    vocab_fr: torchtext.vocab.Vocab,
    vocab_en: torchtext.vocab.Vocab,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenizes and processes a dataset of sentence pairs into tensor pairs.

    Arguments:
        - df: pd.DataFrame: The dataset containing French
        and English sentences.
        - vocab_fr: Vocab: The vocabulary object for the
        French language.
        - vocab_en: Vocab: The vocabulary object for the
        English language.

    Returns:
        - tokenized_data: List[Tuple[torch.Tensor, torch.Tensor]]: A
        list of tokenized sentence pairs as tensors.
    """
    tokenized_data = []
    for _, item in df.iterrows():
        if isinstance(item["fr"], str) and isinstance(item["en"], str):
            fr_tokens, en_tokens = tokenize_sentence_pair(
                item, vocab_fr, vocab_en
            )
            tokenized_data.append((fr_tokens, en_tokens))
    return tokenized_data


if not os.path.exists("data/train_data_sample.pkl"):
    logging.info("Preprocessing dataset...")
    data_sample = preprocess_data(df, vocab_fr, vocab_en)
    train_size = int(train_size * len(data_sample))
    valid_size = len(data_sample) - train_size
    train_data_sample, valid_data_sample = random_split(
        data_sample, [train_size, valid_size]
    )

    with open("data/train_data_sample.pkl", "wb") as f:
        pickle.dump(train_data_sample, f)

    with open("data/valid_data_sample.pkl", "wb") as f:
        pickle.dump(valid_data_sample, f)
    del data_sample

else:
    with open("data/train_data_sample.pkl", "rb") as f:
        train_data_sample = pickle.load(f)
    with open("data/valid_data_sample.pkl", "rb") as f:
        valid_data_sample = pickle.load(f)

del df
gc.collect()

logging.info("Dataset preprocessing is over")


class TransformerWithProjection(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        vocab_size_en: int,
        max_len: int = max_len,
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


def translate_sentence(
    sentence: str,
    model: TransformerWithProjection,
    vocab_fr: torchtext.vocab.Vocab,
    vocab_en: torchtext.vocab.Vocab,
    tokenizer_fr: torchtext.data.utils,
    max_len: int,
    device: str = "cpu",
) -> str:
    """
    Translates a French sentence into English using the
    trained Transformer model.

    Arguments:
        - sentence: str: The French sentence to translate.
        - model: TransformerWithProjection: The trained Transformer model.
        - vocab_fr: Vocab: The vocabulary object for French.
        - vocab_en: Vocab: The vocabulary object for English.
        - tokenizer_fr: Tokenizer: The tokenizer for French.
        - max_len: int: The maximum length of input sequences.
        - device: str: The device to use for computation ('cpu' or 'cuda').

    Returns:
        - translated_sentence: str: The translated English sentence.
    """
    # Tokenize and encode the French sentence
    tokens = (
        [vocab_fr["<bos>"]]
        + [vocab_fr[token] for token in tokenizer_fr(sentence)]
        + [vocab_fr["<eos>"]]
    )
    input_tensor = torch.tensor(
        tokens, dtype=torch.long, device=device
    ).unsqueeze(0)

    # Ensure input_tensor matches max_len
    if input_tensor.size(1) < max_len:
        padding = torch.full(
            (1, max_len - input_tensor.size(1)),
            vocab_fr["<pad>"],
            device=device,
            dtype=torch.long,
        )
        input_tensor = torch.cat([input_tensor, padding], dim=1)
    elif input_tensor.size(1) > max_len:
        input_tensor = input_tensor[:, :max_len]

    # Prepare the target tensor with just the <bos> token
    target_tensor = torch.tensor(
        [vocab_en["<bos>"]], dtype=torch.long, device=device
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            # Add positional encoding to input tensor
            input_tensor = input_tensor.to(device=device)
            src_emb = model.positionnal_encoding.add_positional_encoding(
                model.embedder.embed(input_tensor)
            )

            # Add positional encoding to target tensor dynamically
            target_tensor = target_tensor.to(device=device)
            tgt_emb = model.embedder.embed(target_tensor)
            tgt_emb = tgt_emb + model.positionnal_encoding.pe[
                : tgt_emb.size(1), :
            ].unsqueeze(0)

            # Pass through the Transformer model
            encoder_output = model.encoder(src_emb)
            decoder_output = model.decoder(tgt=tgt_emb, memory=encoder_output)
            output_logits = model.projection(decoder_output)

            # Get the next token prediction
            next_token = output_logits[:, -1, :].argmax(dim=-1).item()

            # Append the predicted token to the target sequence
            target_tensor = torch.cat(
                [target_tensor, torch.tensor([[next_token]], device=device)],
                dim=1,
            )

            # Stop if <eos> token is generated
            if next_token == vocab_en["<eos>"]:
                break

    # Decode the tokens into a sentence
    translated_tokens = target_tensor.squeeze().tolist()
    translated_sentence = " ".join(
        vocab_en.lookup_token(idx)
        for idx in translated_tokens
        if idx not in [vocab_en["<bos>"], vocab_en["<eos>"], vocab_en["<pad>"]]
    )

    return translated_sentence


embedding_dim = embedding_dim
num_heads = num_heads
vocab_size_fr = len(vocab_fr)
vocab_size_en = len(vocab_en)

model = TransformerWithProjection(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    vocab_size_en=vocab_size_en,
    max_len=max_len,
    learnable_encoding=False,
).to(device)

optimizer = optim.Adam(model.parameters_to_optimize, lr=learning_rate)

criterion = nn.CrossEntropyLoss(ignore_index=vocab_en["<pad>"])

scorer_rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

scaler = torch.cuda.amp.GradScaler()

logging.info("Model training has begun")

overall_metrics = {}

# train_loader = DataLoader(
#    train_data_sample,
#    batch_size=batch_size,
#    collate_fn=lambda batch: pad_sentences(*zip(*batch), max_len),
# )

valid_loader = DataLoader(
    valid_data_sample,
    batch_size=batch_size,
    collate_fn=lambda batch: pad_sentences(*zip(*batch), max_len=max_len),
)

indices = train_data_sample.indices
chunk_size = 50000

for i in range(0, len(indices), chunk_size):
    chunk_indices = indices[i : i + chunk_size]
    chunk_subset = Subset(train_data_sample.dataset, chunk_indices)
    train_loader = DataLoader(
        chunk_subset,
        batch_size=batch_size,
        collate_fn=lambda batch: pad_sentences(*zip(*batch), max_len=20),
    )

    del chunk_subset
    gc.collect()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for fr_batch, en_batch in train_loader:
            torch.cuda.empty_cache()
            gc.collect()

            optimizer.zero_grad()
            tgt_mask = torch.triu(
                torch.ones(en_batch.size(1), en_batch.size(1))
            ).to(fr_batch.device)
            memory_mask = (
                (fr_batch == vocab_fr["<pad>"])
                .transpose(0, 1)
                .to(fr_batch.device)
            )

            output = model(
                src=fr_batch,
                tgt=en_batch,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )

            output = output.view(-1, vocab_size_en)
            loss = criterion(output, en_batch.view(-1))

            logging.info(f"Training Loss: {loss:.4f}")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logging.warning(f"Epoch {epoch}, Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        total_bleu = 0
        total_rouge_1 = 0
        total_rouge_l = 0
        num_samples = 0

        with torch.no_grad():
            for fr_batch, en_batch in valid_loader:
                tgt_mask = (
                    torch.triu(torch.ones(en_batch.size(1), en_batch.size(1)))
                    .bool()
                    .to(fr_batch.device)
                )
                memory_mask = (
                    (fr_batch == vocab_fr["<pad>"])
                    .transpose(0, 1)
                    .bool()
                    .to(fr_batch.device)
                )

                output = model(
                    src=fr_batch,
                    tgt=en_batch,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                )

                output = output.view(-1, vocab_size_en)
                loss = criterion(output, en_batch.view(-1))
                val_loss += loss.item()

                output_tokens = (
                    output.argmax(dim=-1).view(en_batch.size()).tolist()
                )
                target_tokens = en_batch.tolist()

                for output_seq, target_seq in zip(
                    output_tokens, target_tokens
                ):
                    output_seq = [
                        vocab_en.lookup_token(idx)
                        for idx in output_seq
                        if idx
                        not in [
                            vocab_en["<bos>"],
                            vocab_en["<eos>"],
                            vocab_en["<pad>"],
                        ]
                    ]
                    target_seq = [
                        vocab_en.lookup_token(idx)
                        for idx in target_seq
                        if idx
                        not in [
                            vocab_en["<bos>"],
                            vocab_en["<eos>"],
                            vocab_en["<pad>"],
                        ]
                    ]

                    bleu_score = sentence_bleu([target_seq], output_seq)
                    total_bleu += bleu_score

                    rouge_scores = scorer_rouge.score(
                        " ".join(target_seq), " ".join(output_seq)
                    )
                    total_rouge_1 += rouge_scores["rouge1"].fmeasure
                    total_rouge_l += rouge_scores["rougeL"].fmeasure

                    num_samples += 1

        avg_val_loss = val_loss / len(valid_loader)
        avg_bleu = total_bleu / num_samples
        avg_rouge_1 = total_rouge_1 / num_samples
        avg_rouge_l = total_rouge_l / num_samples

        metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "bleu_score": avg_bleu,
            "rouge1_score": avg_rouge_1,
            "rougeL_score": avg_rouge_l,
        }

        epoch_key = f"epoch_{epoch}"
        overall_metrics[epoch_key] = metrics

        logging.warning(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
        logging.warning(f"Epoch {epoch}, Validation BLEU Score: {avg_bleu}")
        logging.warning(
            f"Epoch {epoch}, Validation ROUGE-1 Score: {avg_rouge_1}"
        )
        logging.warning(
            f"Epoch {epoch}, Validation ROUGE-L Score: {avg_rouge_l}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            "models/checkpoint_last_epoch.pth",
        )

        del train_loader
        gc.collect()

os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics_epochs.json", "w") as f:
    json.dump(overall_metrics, f)


# Example of model utilisation
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

french_sentence = "Bonjour, comment allez-vous ?"
translated_sentence = translate_sentence(
    sentence=french_sentence,
    model=model,
    vocab_fr=vocab_fr,
    vocab_en=vocab_en,
    tokenizer_fr=tokenizer_fr,
    max_len=max_len,
    device="cpu",
)

print("French:", french_sentence)
print("English:", translated_sentence)
