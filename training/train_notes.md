# Training Notes

## 1. Training Configuration

The model is trained on the IMDb dataset for binary sentiment classification.

| Parameter                 |             Value |
| ------------------------- | ----------------: |
| Dataset                   |              IMDb |
| Training samples          |            25,000 |
| Testing samples           |            25,000 |
| Maximum vocabulary size   |            10,000 |
| Maximum sequence length   |         80 tokens |
| Embedding dimension       |                64 |
| Number of attention heads |                 4 |
| Feed-forward dimension    |                64 |
| Number of encoder layers  |                 2 |
| Batch size                |                32 |
| Epochs                    |                10 |
| Learning rate             |             0.001 |
| Optimizer                 |              Adam |
| Loss function             | BCEWithLogitsLoss |
| Device                    |               CPU |

## 2. Training Pipeline

The training process follows these steps:

1. Load the IMDb dataset and build a vocabulary from the training reviews.
2. Convert reviews into token IDs and prepare batches.
3. Pass token IDs through the embedding layer.
4. Add positional encoding to provide information about token positions.
5. Process token representations through Transformer encoder blocks using multi-head self-attention and feed-forward networks.
6. Apply masked mean pooling to combine valid token representations.
7. Generate sentiment logits using the linear classifier.
8. Calculate the loss against the actual labels.
9. Perform backpropagation and update parameters using Adam.
10. Evaluate the model on the test dataset after each epoch.

### Loss Function

We use `BCEWithLogitsLoss`, which combines sigmoid activation and binary cross-entropy in a numerically stable implementation.

The model returns raw logits during training.

### Evaluation Metrics

* **Training loss:** Measures prediction error on training data.
* **Training accuracy:** Measures the proportion of correct training predictions.
* **Test loss:** Measures prediction error on test data.
* **Test accuracy:** Measures the proportion of correct test predictions.

The test set is used for evaluation after each epoch. Selecting a checkpoint based on test performance means the test set influences model selection; a separate validation set would provide a cleaner final evaluation.

## 3. Observations

### Learning Progress

The model was trained for 10 epochs on a CPU.

| Metric                  |     Result |
| ----------------------- | ---------: |
| Best test accuracy      | **78.98%** |
| Best epoch              |          6 |
| Final training accuracy |     90.81% |
| Final test accuracy     |     77.94% |

### Learning Progress

Training loss decreased consistently from **0.5786 to 0.2256**, while training accuracy increased from **68.21% to 90.81%**.

Test accuracy improved rapidly during the first few epochs, reaching its highest value of **78.98% at Epoch 6**.

### Overfitting

After the early epochs, training performance continued improving while test loss generally increased.

This suggests that the model increasingly fitted the training data without achieving corresponding improvements on unseen reviews.

### Accuracy vs. Loss

The lowest test loss occurred at Epoch 2, while the highest test accuracy occurred at Epoch 6.

Test accuracy measures the proportion of correct predictions, whereas test loss also accounts for prediction confidence. A model can therefore achieve higher accuracy while having a higher loss.

### Best Checkpoint

The best epoch depends on the evaluation metric:

* **Highest test accuracy:** Epoch 6, at 78.98%.
* **Lowest test loss:** Epoch 2, at 0.4626.

These results do not identify a single best checkpoint for every objective. A separate validation set should ideally be used to select checkpoints before final test evaluation.

## 4. Key Takeaways

* Self-attention allows tokens to learn relationships without recurrent processing.
* Positional encoding provides information about token order.
* Multiple attention heads can learn different relationships between tokens.
* Training accuracy alone is insufficient to assess model quality.
* More epochs do not necessarily improve generalization.
* Test loss can increase even when test accuracy remains relatively stable.
* Transformer performance depends on the dataset, preprocessing, architecture, and training configuration.

**Result:** The Transformer achieved a best observed test accuracy of **78.98%** during this 10-epoch experiment.
