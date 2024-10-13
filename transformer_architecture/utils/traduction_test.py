import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from transformer_architecture.model.encoder import TransformerEncoderLayer
from transformer_architecture.model.decoder import TransformerDecoderLayer

warnings.filterwarnings("ignore")

tokenizer_fr = get_tokenizer("spacy", language="fr_core_news_sm")
tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")

dataset = load_dataset("opus100", "en-fr")

train_data = dataset["train"]


def tokenize_sentence_pair(item, vocab_fr, vocab_en, max_len=None):
    fr_tokens = (
        [vocab_fr["<bos>"]]
        + [
            vocab_fr[token]
            for token in tokenizer_fr(item["translation"]["fr"])
        ]
        + [vocab_fr["<eos>"]]
    )
    en_tokens = (
        [vocab_en["<bos>"]]
        + [
            vocab_en[token]
            for token in tokenizer_en(item["translation"]["en"])
        ]
        + [vocab_en["<eos>"]]
    )

    if max_len:
        fr_tokens = fr_tokens[:max_len]
        en_tokens = en_tokens[:max_len]

    return torch.tensor(fr_tokens), torch.tensor(en_tokens)


def build_vocab(dataset):
    def tokenize_pair(data):
        for item in data:
            yield tokenizer_fr(item["translation"]["fr"])
            yield tokenizer_en(item["translation"]["en"])

    vocab = build_vocab_from_iterator(
        tokenize_pair(dataset), specials=["<unk>", "<pad>", "<bos>", "<eos>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


vocab_fr = build_vocab(train_data)
vocab_en = build_vocab(train_data)


def preprocess_data(data, vocab_fr, vocab_en, max_len=None):
    tokenized_data = []
    for item in data:
        fr_tokens, en_tokens = tokenize_sentence_pair(
            item, vocab_fr, vocab_en, max_len
        )
        tokenized_data.append((fr_tokens, en_tokens))
    return tokenized_data


train_data_sample = preprocess_data(
    train_data.select(range(100)), vocab_fr, vocab_en
)


def get_max_len(fr_batch, en_batch):
    fr_max_len = max(len(sentence) for sentence in fr_batch)
    en_max_len = max(len(sentence) for sentence in en_batch)
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

    fr_batch_padded = torch.stack(fr_batch_padded)
    en_batch_padded = torch.stack(en_batch_padded)

    return fr_batch_padded, en_batch_padded


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_padding_mask(seq, pad_idx):
    return (seq == pad_idx).transpose(0, 1)


class TransformerWithProjection(nn.Module):
    def __init__(self, embedding_dim, num_heads, vocab_size_fr, vocab_size_en):
        super(TransformerWithProjection, self).__init__()

        self.embedding_fr = nn.Embedding(vocab_size_fr, embedding_dim)
        self.embedding_en = nn.Embedding(vocab_size_en, embedding_dim)

        self.encoder = TransformerEncoderLayer(
            d_model=embedding_dim, num_heads=num_heads, norm_first=True
        )
        self.decoder = TransformerDecoderLayer(
            d_model=embedding_dim, num_heads=num_heads, norm_first=True
        )

        self.projection = nn.Linear(embedding_dim, vocab_size_en)

    def forward(self, src, tgt, tgt_mask=None, memory_mask=None):
        src = self.embedding_fr(src)
        tgt = self.embedding_en(tgt)

        encoder_output = self.encoder(src=src)

        decoder_output = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        vocab_output = self.projection(decoder_output)

        return vocab_output


def collate_fn(batch):
    fr_batch, en_batch = zip(*batch)

    max_len = get_max_len(fr_batch, en_batch)

    fr_batch_padded, en_batch_padded = pad_sentences(
        fr_batch, en_batch, max_len
    )

    return fr_batch_padded, en_batch_padded


train_loader = DataLoader(
    train_data_sample, batch_size=32, collate_fn=collate_fn
)

embedding_dim = 16
num_heads = 2
vocab_size_fr = len(vocab_fr)
vocab_size_en = len(vocab_en)

model = TransformerWithProjection(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    vocab_size_fr=vocab_size_fr,
    vocab_size_en=vocab_size_en,
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab_en["<pad>"])

for epoch in range(10):
    model.train()
    total_loss = 0
    for fr_batch, en_batch in train_loader:
        optimizer.zero_grad()

        tgt_mask = generate_square_subsequent_mask(en_batch.size(1)).to(
            fr_batch.device
        )
        memory_mask = create_padding_mask(fr_batch, vocab_fr["<pad>"]).to(
            fr_batch.device
        )

        tgt_input = en_batch

        output = model(
            src=fr_batch,
            tgt=tgt_input,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        output = output.view(-1, vocab_size_en)
        tgt_output = en_batch.view(-1)
        loss = criterion(output, tgt_output)
        total_loss += loss.item()

        print("loss", loss)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")
