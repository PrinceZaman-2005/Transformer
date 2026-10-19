
import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def build_vocabulary(reviews, max_vocab_size=10000):
    """Build a vocabulary from training reviews."""
    word_counts = Counter()

    for review in reviews:
        word_counts.update(tokenize(review))

    # Reserve IDs 0 and 1 for special tokens.
    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    available_words = max_vocab_size - len(vocabulary)

    for word, _ in word_counts.most_common(available_words):
        vocabulary[word] = len(vocabulary)

    return vocabulary


def encode_review(review, vocabulary, max_length=200):
    """Convert a review into a fixed-length sequence of token IDs."""
    tokens = tokenize(review)

    token_ids = [
        vocabulary.get(token, vocabulary[UNK_TOKEN])
        for token in tokens[:max_length]
    ]

    # Pad shorter reviews.
    token_ids.extend(
        [vocabulary[PAD_TOKEN]] * (max_length - len(token_ids))
    )

    return token_ids


def prepare_split(split, vocabulary, max_length=200):
    """Convert a Hugging Face dataset split into tensors."""
    reviews = split["text"]
    labels = split["label"]

    encoded_reviews = [
        encode_review(review, vocabulary, max_length)
        for review in reviews
    ]

    inputs = torch.tensor(encoded_reviews, dtype=torch.long)
    targets = torch.tensor(labels, dtype=torch.float32)

    return TensorDataset(inputs, targets)


def create_dataloaders(
    batch_size=32,
    max_vocab_size=10000,
    max_length=200,
    seed=42,
):
    """Load IMDb and return training/testing DataLoaders."""

    print("Loading IMDb dataset...")

    dataset = load_dataset("stanfordnlp/imdb")

    train_split = dataset["train"]
    test_split = dataset["test"]

    print("Building vocabulary from training reviews...")

    vocabulary = build_vocabulary(
        train_split["text"],
        max_vocab_size=max_vocab_size,
    )

    print(f"Vocabulary size: {len(vocabulary)}")

    print("Encoding training reviews...")
    train_dataset = prepare_split(
        train_split,
        vocabulary,
        max_length,
    )

    print("Encoding testing reviews...")
    test_dataset = prepare_split(
        test_split,
        vocabulary,
        max_length,
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, vocabulary
