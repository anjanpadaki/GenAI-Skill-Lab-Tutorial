"""
GENAI SKILL LAB - QUESTION 11
Time series forecasting using an RNN in PyTorch.
  i.   Generate synthetic time series data (sine + noise) and visualize it.
  ii.  Prepare input-output pairs using a fixed time-step window (sequence length).
  iii. Define an RNN model: recurrent layer (nn.RNN) + fully connected layer (nn.Linear).
  iv.  Train with MSE loss and the Adam optimizer; predict on the time series.
  v.   Print the loss during training to monitor progress.
  vi.  Visualize the original data with the predicted values.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(data, seq_length):
    """Each input is `seq_length` consecutive values; the target is the next value."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class RNNForecaster(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)            # output: (batch, seq_len, hidden)
        return self.fc(output[:, -1, :])   # use the last time step to predict the next value


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # i. Generate synthetic sine-wave data with noise and visualize it
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Generating synthetic time series (sine + noise)")
    print("=" * 60)
    t = np.linspace(0, 40 * np.pi, 1200)
    clean = np.sin(t)
    data = clean + 0.08 * np.random.randn(len(t))
    print(f"Number of time steps : {len(data)}")
    print(f"Noise std deviation  : 0.08")

    plt.figure(figsize=(12, 4))
    plt.plot(data, label="Sine + noise (generated data)", linewidth=1)
    plt.plot(clean, label="Underlying sine wave", linestyle="--", alpha=0.7)
    plt.title("Synthetic Time Series Data")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rnn_data.png", dpi=150)
    print("Saved plot: rnn_data.png")
    plt.show()

    # ------------------------------------------------------------------
    # ii. Create fixed-length input-output sequences
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Preparing input-output pairs")
    print("=" * 60)
    sequence_length = 30
    X, y = create_sequences(data, sequence_length)
    split = int(0.8 * len(X))

    X_train = torch.from_numpy(X[:split]).unsqueeze(-1)   # (N, 30, 1)
    y_train = torch.from_numpy(y[:split]).unsqueeze(-1)   # (N, 1)
    X_test = torch.from_numpy(X[split:]).unsqueeze(-1)
    y_test = torch.from_numpy(y[split:]).unsqueeze(-1)

    print(f"Sequence length (window) : {sequence_length}")
    print(f"Training pairs           : {len(X_train)}")
    print(f"Testing pairs            : {len(X_test)}")
    print(f"Input shape              : {tuple(X_train.shape)}  (samples, time steps, features)")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # ------------------------------------------------------------------
    # iii. Define the RNN model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: RNN model")
    print("=" * 60)
    model = RNNForecaster(hidden_size=32)
    print(model)

    # ------------------------------------------------------------------
    # iv + v. Train with MSE loss and Adam, printing the loss
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Training (MSE loss, Adam optimizer)")
    print("=" * 60)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(X_train)

        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test), y_test).item()
            print(f"Epoch {epoch:03d}/{epochs} - Train Loss: {train_loss:.6f} - Test Loss: {test_loss:.6f}")

    # Predict on the whole series (train + test) with the trained model
    model.eval()
    with torch.no_grad():
        train_prediction = model(X_train).squeeze().numpy()
        test_prediction = model(X_test).squeeze().numpy()
        test_mse = criterion(model(X_test), y_test).item()

    noise_mse = 0.08 ** 2
    print(f"\nFinal Test MSE            : {test_mse:.6f}")
    print(f"Noise variance (best case): {noise_mse:.6f}  (the noise itself cannot be predicted)")

    # ------------------------------------------------------------------
    # vi. Visualize original data with predicted values
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Visualizing predictions")
    print("=" * 60)
    train_idx = np.arange(sequence_length, sequence_length + split)
    test_idx = np.arange(sequence_length + split, sequence_length + split + len(test_prediction))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(data, label="Original time series", linewidth=1, alpha=0.7)
    axes[0].plot(train_idx, train_prediction, label="Prediction (train)", linewidth=1.2)
    axes[0].plot(test_idx, test_prediction, label="Prediction (test)", linewidth=1.2)
    axes[0].axvline(test_idx[0], color="black", linestyle="--", label="Test start")
    axes[0].set_title("PyTorch RNN Time-Series Forecasting - Full Series")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="upper right")

    axes[1].plot(test_idx, data[test_idx], label="Original (test)", linewidth=1)
    axes[1].plot(test_idx, test_prediction, label="RNN prediction", linewidth=1.5)
    axes[1].set_title(f"Zoom on Test Region (Test MSE = {test_mse:.4f})")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Value")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("rnn_forecast.png", dpi=150)
    print("Saved plot: rnn_forecast.png")
    plt.show()


if __name__ == "__main__":
    main()
