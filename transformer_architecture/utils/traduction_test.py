import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import logging
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import build_vocab_from_iterator
from logging.handlers import RotatingFileHandler
from transformer_architecture.model.encoder import TransformerEncoderLayer
from transformer_architecture.model.decoder import TransformerDecoderLayer
from transformer_architecture.preprocessing.embedding import (
    Embedding,
    SinusoidalPositionalEncoding,
)
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

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

tokenizer_fr = get_tokenizer("spacy", language="fr_core_news_sm")
tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")

df = pd.read_csv("en-fr.csv", nrows=2000)
df.dropna(subset=["en", "fr"], inplace=True)


def get_corpus_max_len(data_sample):
    fr_max_len = max(len(s[0]) for s in data_sample)
    en_max_len = max(len(s[1]) for s in data_sample)
    return max(fr_max_len, en_max_len)


def pad_sentences(fr_batch, en_batch, max_len):
    fr_batch_padded = [
        torch.cat(
            [
                sentence,
                torch.full((max_len - len(sentence),), vocab_fr["<pad>"]),
            ]
        )
        if len(sentence) < max_len
        else sentence
        for sentence in fr_batch
    ]
    en_batch_padded = [
        torch.cat(
            [
                sentence,
                torch.full((max_len - len(sentence),), vocab_en["<pad>"]),
            ]
        )
        if len(sentence) < max_len
        else sentence
        for sentence in en_batch
    ]
    return torch.stack(fr_batch_padded), torch.stack(en_batch_padded)


try:
    with open("vocab_fr.pkl", "rb") as f:
        vocab_fr = pickle.load(f)
    with open("vocab_en.pkl", "rb") as f:
        vocab_en = pickle.load(f)
except FileNotFoundError:

    def build_vocab(df):
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

    vocab_fr = build_vocab(df)
    vocab_en = build_vocab(df)

    with open("vocab_fr.pkl", "wb") as f:
        pickle.dump(vocab_fr, f)
    with open("vocab_en.pkl", "wb") as f:
        pickle.dump(vocab_en, f)


def tokenize_sentence_pair(item, vocab_fr, vocab_en):
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


def preprocess_data(df, vocab_fr, vocab_en):
    tokenized_data = []
    for _, item in df.iterrows():
        if isinstance(item["fr"], str) and isinstance(item["en"], str):
            fr_tokens, en_tokens = tokenize_sentence_pair(
                item, vocab_fr, vocab_en
            )
            tokenized_data.append((fr_tokens, en_tokens))
    return tokenized_data


data_sample = preprocess_data(df, vocab_fr, vocab_en)
train_size = int(0.8 * len(data_sample))
valid_size = len(data_sample) - train_size
train_data_sample, valid_data_sample = random_split(
    data_sample, [train_size, valid_size]
)

max_len = get_corpus_max_len(train_data_sample)

train_loader = DataLoader(
    train_data_sample,
    batch_size=128,
    collate_fn=lambda batch: pad_sentences(*zip(*batch), max_len),
)
valid_loader = DataLoader(
    valid_data_sample,
    batch_size=128,
    collate_fn=lambda batch: pad_sentences(*zip(*batch), max_len),
)


class TransformerWithProjection(nn.Module):
    def __init__(
        self, embedding_dim, num_heads, vocab_size_fr, vocab_size_en, max_len
    ):
        super(TransformerWithProjection, self).__init__()
        self.embedder = Embedding(embedding_dim=embedding_dim)
        self.positionnal_encoding = SinusoidalPositionalEncoding(
            max_len=max_len, embedding_dim=embedding_dim
        )
        self.encoder = TransformerEncoderLayer(
            d_model=embedding_dim, num_heads=num_heads, norm_first=True
        )
        self.decoder = TransformerDecoderLayer(
            d_model=embedding_dim, num_heads=num_heads, norm_first=True
        )
        self.projection = nn.Linear(embedding_dim, vocab_size_en)

    def forward(self, src, tgt, tgt_mask=None, memory_mask=None):
        src_emb = self.embedder.embed(src)
        tgt_emb = self.embedder.embed(tgt)
        src = self.positionnal_encoding.add_positional_encoding(src_emb)
        tgt = self.positionnal_encoding.add_positional_encoding(tgt_emb)
        encoder_output = self.encoder(src=src)
        decoder_output = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return self.projection(decoder_output)


embedding_dim = 8
num_heads = 4
vocab_size_fr = len(vocab_fr)
vocab_size_en = len(vocab_en)

model = TransformerWithProjection(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    vocab_size_fr=vocab_size_fr,
    vocab_size_en=vocab_size_en,
    max_len=max_len,
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab_en["<pad>"])

scorer_rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

for epoch in range(10):
    model.train()
    total_loss = 0

    for fr_batch, en_batch in train_loader:
        optimizer.zero_grad()
        tgt_mask = torch.triu(
            torch.ones(en_batch.size(1), en_batch.size(1))
        ).to(fr_batch.device)
        memory_mask = (
            (fr_batch == vocab_fr["<pad>"]).transpose(0, 1).to(fr_batch.device)
        )

        output = model(
            src=fr_batch,
            tgt=en_batch,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        output = output.view(-1, vocab_size_en)
        loss = criterion(output, en_batch.view(-1))
        logging.warning(f"Training Loss: {loss:.4f}")
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch}, Training Loss: {avg_train_loss}")

    model.eval()
    val_loss = 0
    total_bleu = 0
    total_rouge_1 = 0
    total_rouge_l = 0
    num_samples = 0

    with torch.no_grad():
        for fr_batch, en_batch in valid_loader:
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
            val_loss += loss.item()

            output_tokens = (
                output.argmax(dim=-1).view(en_batch.size()).tolist()
            )
            target_tokens = en_batch.tolist()

            for output_seq, target_seq in zip(output_tokens, target_tokens):
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

    logging.warning(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
    logging.warning(f"Epoch {epoch}, Validation BLEU Score: {avg_bleu}")
    logging.warning(f"Epoch {epoch}, Validation ROUGE-1 Score: {avg_rouge_1}")
    logging.warning(f"Epoch {epoch}, Validation ROUGE-L Score: {avg_rouge_l}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        },
        "checkpoint_last_epoch.pth",
    )
