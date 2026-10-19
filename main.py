import torch

from core.transformer import Transformer
from training.dataset import create_dataloaders
from training.train import Trainer


# Configuration
BATCH_SIZE = 32
MAX_VOCAB_SIZE = 10000
MAX_LENGTH = 80

EMBEDDING_DIM = 64
NUM_HEADS = 4
FEEDFORWARD_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1

EPOCHS = 10
LEARNING_RATE = 0.001


def main():
    # Load the same dataset configuration as Day 8
    train_loader, test_loader, vocabulary = create_dataloaders(
        batch_size=BATCH_SIZE,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_length=MAX_LENGTH,
    )

    # Initialize the Transformer
    model = Transformer(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        feedforward_dim=FEEDFORWARD_DIM,
        num_layers=NUM_LAYERS,
        max_length=MAX_LENGTH,
        dropout=DROPOUT,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        learning_rate=LEARNING_RATE,
    )

    history = trainer.fit(
        train_loader,
        test_loader,
        epochs=EPOCHS,
    )

    # Save the trained model and its configuration
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "config": {
                "max_vocab_size": MAX_VOCAB_SIZE,
                "max_length": MAX_LENGTH,
                "embedding_dim": EMBEDDING_DIM,
                "num_heads": NUM_HEADS,
                "feedforward_dim": FEEDFORWARD_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            },
        },
        "transformer_imdb.pth",
    )

    print("\nTraining completed!")
    print("Model saved to transformer_imdb.pth")


if __name__ == "__main__":
    main()
