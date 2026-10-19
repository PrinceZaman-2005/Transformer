# ✨ Transformer Notes

## 1. What Is a Transformer?

A **Transformer** is a neural network architecture that uses attention mechanisms to learn relationships between tokens in a sequence.

Unlike RNNs and LSTMs, Transformers can process tokens in parallel during training and use self-attention to connect information across a sequence.

In this project, we use a Transformer encoder for IMDb movie review sentiment classification.

## 2. Core Components

### 🔹 Token Embedding

Converts token IDs into dense numerical vectors.

Each token receives a learnable representation of dimension `embedding_dim`.

### 🔹 Positional Encoding

Self-attention does not inherently understand token order. Positional encoding adds position-dependent information to token embeddings.

This implementation uses sinusoidal positional encoding:

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

Where:

* `pos` = token position
* `i` = embedding dimension index
* `d` = embedding dimension

### 🔹 Multi-Head Self-Attention

Self-attention allows each token to gather information from other tokens in the sequence.

It uses three learned projections:

* **Query (Q):** What information is the token looking for?
* **Key (K):** What information does the token represent?
* **Value (V):** What information can the token contribute?

The attention operation is:

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Multiple attention heads learn different relationships between tokens.

### 🔹 Feed-Forward Network

After attention, each token representation passes through a small neural network:

```text
Linear → ReLU → Linear
```

It transforms the features learned by the attention mechanism.

### 🔹 Residual Connections and Layer Normalization

Residual connections preserve the original representation while adding the sublayer's output.

Layer normalization helps stabilize the activations.

This implementation uses:

```text
Attention → Residual + LayerNorm
Feed-Forward → Residual + LayerNorm
```

### 🔹 Padding Mask

Reviews have different lengths. Padding makes their sequences uniform, but padding tokens should not contribute to attention.

The padding mask prevents attention from using padded positions as keys or values.

### 🔹 Masked Mean Pooling

The model averages the final representations of valid tokens while excluding padding positions.

The pooled representation is passed to a linear classifier.

## 3. Architecture

```text
      Input Token IDs
           │
           ▼
      Token Embedding
           │
           ▼
    Positional Encoding
           │
           ▼
 ┌──────────────────────┐
 │ Transformer Encoder  │
 │                      │
 │ Multi-Head Attention │
 │          │           │
 │ Residual + LayerNorm │
 │          │           │
 │ Feed-Forward Network │
 │          │           │
 │ Residual + LayerNorm │
 └──────────────────────┘
           │
           ▼
    Masked Mean Pooling
           │
           ▼
     Linear Classifier
           │
           ▼
      Sentiment Logit
```

## 4. Model Configuration

| Parameter               | Value |
| ----------------------- | ----: |
| Embedding dimension     |    64 |
| Attention heads         |     4 |
| Encoder layers          |     2 |
| Feed-forward dimension  |   128 |
| Maximum sequence length |    80 |
| Dropout                 |   0.1 |
| Output units            |     1 |

## 5. Input and Output

**Input shape:**

```text
(batch_size, sequence_length)
```

Example: `(32, 80)`

**Output shape:**

```text
(batch_size,)
```

Each output is a raw logit for binary sentiment classification.

The model uses `BCEWithLogitsLoss` during training. Apply sigmoid to convert logits into probabilities for prediction.

## 6. Key Takeaways

* Transformers use self-attention to learn relationships between tokens.
* Positional encoding provides information about token order.
* Multi-head attention can learn different types of relationships.
* Residual connections and layer normalization support stable learning.
* Padding masks prevent padded tokens from influencing attention.
* Unlike recurrent models, Transformers can process sequence positions in parallel during training.

**Important:** This project implements a Transformer encoder for classification, not a GPT-style language-generation model.
