import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from transformer_architecture.model.encoder import TransformerEncoderLayer
from transformer_architecture.model.decoder import TransformerDecoderLayer
from transformer_architecture.preprocessing.embedding import (
    Embedding,
    SinusoidalPositionalEncoding,
)

tokenizer_fr = get_tokenizer("spacy", language="fr_core_news_sm")
tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")

dataset = load_dataset("opus100", "en-fr", cache_dir="./cache")
train_data = dataset["train"]

try:
    with open("vocab_fr.pkl", "rb") as f:
        vocab_fr = pickle.load(f)
    with open("vocab_en.pkl", "rb") as f:
        vocab_en = pickle.load(f)
except FileNotFoundError:

    def build_vocab(dataset):
        def tokenize_pair(data):
            for item in data:
                yield tokenizer_fr(item["translation"]["fr"])
                yield tokenizer_en(item["translation"]["en"])

        vocab = build_vocab_from_iterator(
            tokenize_pair(dataset),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    vocab_fr = build_vocab(train_data)
    vocab_en = build_vocab(train_data)

    with open("vocab_fr.pkl", "wb") as f:
        pickle.dump(vocab_fr, f)
    with open("vocab_en.pkl", "wb") as f:
        pickle.dump(vocab_en, f)

try:
    with open("train_data_sample.pkl", "rb") as f:
        train_data_sample = pickle.load(f)
except FileNotFoundError:

    def tokenize_sentence_pair(item, vocab_fr, vocab_en):
        fr_tokens = (
            [vocab_fr["<bos>"]]
            + [
                vocab_fr.get(token, vocab_fr["<unk>"])
                for token in tokenizer_fr(item["translation"]["fr"])
            ]
            + [vocab_fr["<eos>"]]
        )
        en_tokens = (
            [vocab_en["<bos>"]]
            + [
                vocab_en.get(token, vocab_en["<unk>"])
                for token in tokenizer_en(item["translation"]["en"])
            ]
            + [vocab_en["<eos>"]]
        )
        return torch.tensor(fr_tokens), torch.tensor(en_tokens)

    def preprocess_data(data, vocab_fr, vocab_en):
        tokenized_data = []
        for item in data:
            fr_tokens, en_tokens = tokenize_sentence_pair(
                item, vocab_fr, vocab_en
            )
            tokenized_data.append((fr_tokens, en_tokens))
        return tokenized_data

    train_data_sample = preprocess_data(train_data, vocab_fr, vocab_en)

    with open("train_data_sample.pkl", "wb") as f:
        pickle.dump(train_data_sample, f)


def get_corpus_max_len(data_sample):
    fr_max_len = max(len(s[0]) for s in data_sample)
    en_max_len = max(len(s[1]) for s in data_sample)
    return max(fr_max_len, en_max_len)


max_len = get_corpus_max_len(train_data_sample)


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


def collate_fn(batch):
    fr_batch, en_batch = zip(*batch)
    return pad_sentences(fr_batch, en_batch, max_len)


train_loader = DataLoader(
    train_data_sample, batch_size=32, collate_fn=collate_fn
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


embedding_dim = 16
num_heads = 2
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
        print(f"loss: {loss}")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")
