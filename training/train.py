import torch
import torch.nn as nn
from torch.optim import Adam


class Trainer:
    def __init__(
        self,
        model,
        learning_rate=0.001,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        # Binary classification loss.
        self.criterion = nn.BCEWithLogitsLoss()

        # Update all trainable parameters.
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.history = {
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }

    def train_epoch(self, train_loader):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Clear gradients from the previous batch.
            self.optimizer.zero_grad()

            # Forward pass.
            logits = self.model(inputs)

            # Calculate loss.
            loss = self.criterion(logits, labels)

            # Backpropagation through time.
            loss.backward()

            # Update model parameters.
            self.optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = (logits >= 0).long()
            correct += (predictions == labels.long()).sum().item()
            total += batch_size

        average_loss = total_loss / total
        accuracy = correct / total

        return average_loss, accuracy

    def evaluate(self, test_loader):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)

                loss = self.criterion(logits, labels)

                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size

                predictions = (logits >= 0).long()
                correct += (
                    predictions == labels.long()
                ).sum().item()

                total += batch_size

        average_loss = total_loss / total
        accuracy = correct / total

        return average_loss, accuracy

    def fit(
        self,
        train_loader,
        test_loader,
        epochs=5,
    ):
        print(f"Training on: {self.device}")
        print(f"Epochs: {epochs}\n")

        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(
                train_loader
            )

            test_loss, test_accuracy = self.evaluate(
                test_loader
            )

            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
            self.history["train_accuracy"].append(
                train_accuracy
            )
            self.history["test_accuracy"].append(
                test_accuracy
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2%} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Accuracy: {test_accuracy:.2%}"
            )

        return self.history
