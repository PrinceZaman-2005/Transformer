import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.
    """

    def __init__(self, embedding_dim, max_length=5000):
        super().__init__()

        positions = torch.arange(max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-math.log(10000.0) / embedding_dim)
        )

        encoding = torch.zeros(max_length, embedding_dim)

        encoding[:, 0::2] = torch.sin(positions * div_term)

        # Handle odd embedding dimensions.
        odd_dimensions = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(
            positions * div_term[:odd_dimensions]
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, embedding_dim)
        """
        seq_len = x.size(1)

        return x + self.encoding[:, :seq_len, :]


class TransformerEncoderBlock(nn.Module):
    """
    One Transformer encoder block containing:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections
    - Layer normalization
    """

    def __init__(
        self,
        embedding_dim=64,
        num_heads=4,
        feedforward_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embedding_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # Self-attention
        attention_output, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        # Residual connection and normalization
        x = self.norm1(x + self.dropout(attention_output))

        # Feed-forward network
        feedforward_output = self.feedforward(x)

        # Residual connection and normalization
        x = self.norm2(x + self.dropout(feedforward_output))

        return x


class Transformer(nn.Module):
    """
    Transformer encoder for binary sentiment classification.

    Input:
        x: (batch_size, seq_len)

    Output:
        logits: (batch_size,)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        num_heads=4,
        feedforward_dim=128,
        num_layers=2,
        output_size=1,
        max_length=80,
        dropout=0.1,
        padding_idx=0,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.position_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_length,
        )

        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(
            embedding_dim,
            output_size,
        )

    def forward(self, x):
        """
        x: Token IDs, shape (batch_size, seq_len).
        Padding tokens must use padding_idx (default: 0).
        """

        # True marks padding positions for MultiheadAttention.
        padding_mask = x.eq(0)

        # Convert token IDs into embeddings.
        x = self.embedding(x)

        # Add positional information.
        x = self.position_encoding(x)

        x = self.dropout(x)

        # Pass through Transformer encoder blocks.
        for layer in self.encoder_layers:
            x = layer(
                x,
                padding_mask=padding_mask,
            )

        # Mask padding positions before mean pooling.
        valid_tokens = (~padding_mask).unsqueeze(-1).to(x.dtype)

        summed_embeddings = (x * valid_tokens).sum(dim=1)

        token_counts = valid_tokens.sum(dim=1).clamp(min=1)

        pooled_output = summed_embeddings / token_counts

        # Produce one raw logit per review.
        logits = self.classifier(pooled_output)

        return logits.squeeze(-1)
