# ⚙️ Transformer

A Transformer encoder built with PyTorch to explore self-attention, positional encoding, and attention-based learning.

This project demonstrates how Transformers model relationships between tokens without recurrent computation. The model is trained to classify IMDb movie reviews as positive or negative.

## Introduction

A **Transformer** is a neural network architecture that uses attention mechanisms to learn relationships between elements in a sequence.

Unlike RNNs and LSTMs, Transformers do not rely on recurrent hidden states to process tokens sequentially. Instead, self-attention allows each token to gather information from other tokens, while positional encoding provides information about token order.

Transformers are widely used in natural language processing, machine translation, text generation, and modern large language models.

This project implements a **Transformer encoder for binary sentiment classification**.

### Architecture

```text
             IMDb Review
                  |
                  v
               Token IDs
                  |
                  v
            Token Embedding
                  |
                  v
          Positional Encoding
                  |
                  v
       Transformer Encoder × 2
       /                    \
Multi-Head Attention    Feed-Forward
       |                    |
       v                    v
Residual + LayerNorm   Residual + LayerNorm
       \                    /
        \__________________/
                  |
                  v
          Masked Mean Pooling
                  |
                  v
            Linear Layer
                  |
                  v
            Sentiment Logit
                  |
                  v
          Positive / Negative
```

Each Transformer encoder block contains multi-head self-attention, a feed-forward network, residual connections, and layer normalization.

The model stacks two encoder blocks to refine token representations. Masked mean pooling then combines the representations of valid tokens into a single vector, which is passed to a linear classifier.

### How Does It Work?

1. **Input sequence:** The model receives tokenized movie reviews.
2. **Embedding:** Each token ID is converted into a trainable vector representation.
3. **Positional encoding:** Sinusoidal positional information is added to the embeddings.
4. **Query, Key, and Value:** Learned projections allow attention to determine which token information is relevant.
5. **Multi-head self-attention:** Four attention heads learn relationships between tokens in parallel.
6. **Residual connections and layer normalization:** Help preserve representations and stabilize the transformations.
7. **Feed-forward network:** Further transforms each token representation independently.
8. **Encoder stacking:** Two Transformer blocks progressively refine the representations.
9. **Masked mean pooling:** Combines valid token representations while excluding padding positions.
10. **Classification:** A linear layer produces a sentiment logit.
11. **Learning:** Backpropagation calculates gradients, allowing the optimizer to update the model parameters.

The objective is to learn useful contextual representations for sentiment classification.

> Self-attention allows tokens to exchange information directly, but it does not guarantee that the model will learn every meaningful relationship or generalize well to unseen data.

## 📊 Dataset

This project uses the **IMDb Large Movie Review Dataset**.

**Dataset:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (Hugging Face)

| Property                |                      Value |
| ----------------------- | -------------------------: |
| Total reviews           |                     50,000 |
| Training reviews        |                     25,000 |
| Testing reviews         |                     25,000 |
| Classes                 | Positive (1), Negative (0) |
| Maximum vocabulary size |                     10,000 |
| Maximum review length   |                  80 tokens |

### Data Preprocessing

* Convert reviews to lowercase.
* Tokenize text into words.
* Build a vocabulary using training reviews only.
* Convert words into integer token IDs.
* Replace unknown words with `<UNK>`.
* Pad shorter reviews and truncate longer ones to 80 tokens.
* Use a padding mask to prevent padded positions from contributing as keys or values in self-attention.
* Apply masked mean pooling to exclude padding positions from the final sequence representation.

**Special tokens:**

* `<PAD>` → `0`
* `<UNK>` → `1`

## 🏗️ Model Configuration

| Parameter                 |               Value |
| ------------------------- | ------------------: |
| Embedding dimension       |                  64 |
| Number of attention heads |                   4 |
| Feed-forward dimension    |                  64 |
| Number of encoder layers  |                   2 |
| Maximum sequence length   |                  80 |
| Batch size                |                  32 |
| Dropout                   |                 0.1 |
| Optimizer                 |                Adam |
| Learning rate             |               0.001 |
| Loss function             | `BCEWithLogitsLoss` |
| Training epochs           |                  10 |

The model uses an embedding layer, sinusoidal positional encoding, two Transformer encoder blocks, masked mean pooling, and a linear output layer.

The output layer produces one logit per review. During inference, the sigmoid function can convert this logit into a probability.

* Probability ≥ 0.5: Positive
* Probability < 0.5: Negative

During training, raw logits are passed directly to `BCEWithLogitsLoss`.

### Attention Complexity

Self-attention computes relationships between pairs of tokens. For a sequence of length \(n\), its attention computation and score matrix scale approximately as \(O(n^2)\) with respect to sequence length.

This implementation uses a maximum sequence length of 80 tokens to keep the experiment manageable on a CPU.

## 📈 Experimental Results

The model was trained for 10 epochs on a CPU.

| Metric                  |     Result |
| ----------------------- | ---------: |
| Best test accuracy      | **78.98%** |
| Best epoch              |          6 |
| Lowest test loss        |     0.4626 |
| Lowest-loss epoch       |          2 |
| Final training accuracy |     90.81% |
| Final test accuracy     |     77.94% |

### Training Observations

Training loss decreased consistently from 0.5786 to 0.2256, while training accuracy increased from 68.21% to 90.81%.

Test accuracy improved during the early epochs and reached its highest observed value of **78.98% at Epoch 6**.

After the early epochs, training performance continued improving while test loss generally increased.

This suggests that the model was increasingly fitting the training data without achieving corresponding improvements on unseen reviews.

### Accuracy vs. Loss

The lowest test loss occurred at Epoch 2, while the highest test accuracy occurred at Epoch 6.

These metrics measure different aspects of model performance. Accuracy measures the proportion of correct predictions, while binary cross-entropy also accounts for the confidence of predictions.

Consequently, test loss can increase even when test accuracy remains relatively stable.

### Best Checkpoint

The best observed metrics occurred at different epochs:

* **Highest test accuracy:** 78.98% at Epoch 6.
* **Lowest test loss:** 0.4626 at Epoch 2.

The training script saves the model after the final epoch, so the saved checkpoint corresponds to the final training state rather than necessarily the best-performing epoch.

For rigorous model selection, a separate validation set should be used to select checkpoints before evaluating the final model on the test set.

*Results represent one training run and may vary with initialization, data order, and other experimental conditions.*

## ⚖️ Trade-Off: Self-Attention vs. Recurrent Computation

Transformers use self-attention to model relationships between tokens without relying on recurrent processing.

### Advantages

* **Direct token interactions:** Each token can attend to other tokens within the sequence.
* **Parallel computation:** Token positions can be processed in parallel during training.
* **Multi-head attention:** Different heads can learn different relationships between tokens.
* **Contextual representations:** Token representations are refined using information from the surrounding sequence.
* **Flexible architecture:** Transformer blocks can be stacked to build deeper models.
* **Broad applicability:** The architecture underlies many modern language models and other AI systems.

### Trade-Offs

* **Quadratic attention complexity:** Standard self-attention computation and attention-score storage scale quadratically with sequence length.
* **Memory requirements:** Longer sequences and larger models can require substantial memory.
* **Positional information:** Self-attention needs positional information to distinguish token order.
* **Overfitting:** A model can fit its training data without achieving corresponding improvements on unseen data.
* **Limited context:** Information beyond the configured maximum sequence length is discarded during preprocessing.
* **Training cost:** Attention-based models can be computationally demanding, even though token positions can be processed in parallel.

Transformers provide an alternative to recurrent computation, but their effectiveness depends on the task, dataset, architecture, and training configuration.

## What This Project Explores

* Transformer encoder architecture
* Token embeddings and text representation
* Sinusoidal positional encoding
* Query, Key, and Value projections
* Multi-head self-attention
* Feed-forward networks
* Residual connections and layer normalization
* Padding masks and masked mean pooling
* Binary sentiment classification
* Training loss and evaluation metrics
* Overfitting and generalization
* Attention complexity and computational trade-offs

## 🛠️ Technologies

* Python
* PyTorch
* Hugging Face Datasets

The project is designed to run on a CPU and does not require a GPU.

## 🎯 Learning Objective

This project concludes the series by moving from recurrent neural networks to attention-based architectures.

The focus is on understanding how Transformers use self-attention to model relationships between tokens, how positional information contributes to sequence processing, and how encoder representations can be used for classification.

The aim is to understand both sides:

* **How attention-based architectures process sequence information without recurrent computation.**
* **Why a more sophisticated architecture does not automatically guarantee better generalization or higher accuracy.**

## 📚 About the Series

**10 Days of Modeling AI for Beginners**

A hands-on series exploring AI concepts through small, understandable projects.

The objective is to learn by building, experimenting, and examining the mechanisms behind AI—not simply by importing a library and accepting its output.

### 🎉 Series Completed!

**Day 10 — Transformer**

The final project explores attention-based learning and demonstrates how a Transformer encoder can classify movie reviews using self-attention, positional encoding, and feed-forward networks.

Ten projects. Ten days of modeling AI concepts. One learning journey from basic neural networks to Transformers.
