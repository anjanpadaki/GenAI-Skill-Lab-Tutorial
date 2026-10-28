"""
================================================================================
  GENAI SKILL LAB - QUESTION 12
================================================================================
  LSTM-Based Text Generation Model
  Objective:
    Build and train an LSTM neural network to predict the next character
    from text sequences using a classic Shakespeare dataset.
    Generate new text from seed phrases and experiment with hyperparameters
    (sequence length, LSTM hidden units, and training epochs).
================================================================================
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Logger to write output simultaneously to stdout and 'Q12-output'
class Logger:
    def __init__(self, filename="Q12-output"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


sys.stdout = Logger("Q12-output")


# ==============================================================================
# STEP 1: DATASET DEFINITION & PREPROCESSING
# ==============================================================================
shakespeare_text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die - to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream - ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause - there's the respect
That makes calamity of so long life.
"""

text = shakespeare_text.lower().strip()
chars = sorted(list(set(text)))
vocab_size = len(chars)

char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}


class TextDataset(Dataset):
    def __init__(self, text, char2idx, seq_length):
        self.seq_length = seq_length
        self.inputs = []
        self.targets = []
        for i in range(len(text) - seq_length):
            seq_in = text[i:i + seq_length]
            seq_out = text[i + seq_length]
            self.inputs.append([char2idx[ch] for ch in seq_in])
            self.targets.append(char2idx[seq_out])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


# ==============================================================================
# STEP 2: LSTM MODEL ARCHITECTURE
# ==============================================================================
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        out = self.embedding(x)
        out, hidden = self.lstm(out, hidden)
        out = self.fc(out[:, -1, :])  # Take output of last time step
        return out, hidden


# ==============================================================================
# STEP 3: TEXT GENERATION FUNCTION
# ==============================================================================
def generate_text(model, seed_phrase, gen_length, seq_length, char2idx, idx2char, temperature=0.7):
    model.eval()
    seed = seed_phrase.lower()
    
    # Pad or truncate seed to match seq_length
    if len(seed) < seq_length:
        seed = " " * (seq_length - len(seed)) + seed
    else:
        seed = seed[-seq_length:]

    generated = seed_phrase
    curr_seq = [char2idx.get(ch, 0) for ch in seed]

    with torch.no_grad():
        for _ in range(gen_length):
            x = torch.tensor([curr_seq], dtype=torch.long)
            output, _ = model(x)
            
            # Apply temperature scaling
            logits = output.squeeze() / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            
            next_char = idx2char[next_idx]
            generated += next_char
            curr_seq = curr_seq[1:] + [next_idx]

    return generated


# ==============================================================================
# STEP 4: MODEL TRAINING & HYPER-PARAMETER EXPERIMENTATION
# ==============================================================================
def train_and_evaluate(config_name, seq_len, hidden_dim, epochs, lr=0.005, batch_size=32):
    print("\n" + "=" * 80)
    print(f"  EXPERIMENT: {config_name}")
    print("=" * 80)
    print(f"Hyperparameters:")
    print(f"  - Sequence Length (Window): {seq_len}")
    print(f"  - LSTM Hidden Units       : {hidden_dim}")
    print(f"  - Training Epochs         : {epochs}")
    print(f"  - Learning Rate           : {lr}")
    print(f"  - Batch Size              : {batch_size}")

    dataset = TextDataset(text, char2idx, seq_length=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CharLSTM(vocab_size=vocab_size, embed_dim=32, hidden_dim=hidden_dim, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    print("\nTraining Progress:")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            out, _ = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"   Epoch {epoch:2d}/{epochs:2d} | Cross-Entropy Loss: {avg_loss:.4f}")

    print(f"\nFinal Loss for {config_name}: {avg_loss:.4f}")
    return model, seq_len, avg_loss


def main():
    print("=" * 80)
    print("  LSTM TEXT GENERATION MODEL (SHAKESPEARE DATASET)")
    print("=" * 80)
    print("Dataset Summary:")
    print(f"  - Corpus Character Count : {len(text)}")
    print(f"  - Unique Vocabulary Size : {vocab_size} characters: {repr(''.join(chars))}\n")

    # Experiment 1: Baseline Configuration (Shorter window, fewer units, fewer epochs)
    model_exp1, seq_len1, loss1 = train_and_evaluate(
        config_name="Configuration 1 (Baseline: seq_len=10, hidden=64, epochs=15)",
        seq_len=10,
        hidden_dim=64,
        epochs=15
    )

    # Experiment 2: Tuned Configuration (Longer context, more LSTM units, more epochs)
    model_exp2, seq_len2, loss2 = train_and_evaluate(
        config_name="Configuration 2 (Enhanced: seq_len=20, hidden=128, epochs=30)",
        seq_len=20,
        hidden_dim=128,
        epochs=30
    )

    # ==============================================================================
    # STEP 5: COMPARATIVE ANALYSIS & TEXT GENERATION FROM SEED PHRASES
    # ==============================================================================
    print("\n" + "=" * 80)
    print("  HYPERPARAMETER EXPERIMENTATION COMPARISON")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'Seq Len':<10} | {'Hidden Dim':<12} | {'Epochs':<8} | {'Final Loss':<10}")
    print("-" * 80)
    print(f"{'Config 1 (Baseline)':<35} | {10:<10} | {64:<12} | {15:<8} | {loss1:.4f}")
    print(f"{'Config 2 (Enhanced)':<35} | {20:<10} | {128:<12} | {30:<8} | {loss2:.4f}")
    loss_improvement = ((loss1 - loss2) / loss1) * 100.0
    print(f"\nObservation: Config 2 achieved {loss_improvement:.1f}% loss reduction by increasing sequence context and LSTM capacity.")

    print("\n" + "=" * 80)
    print("  TEXT GENERATION FROM STARTING SEED PHRASES")
    print("=" * 80)

    seeds = [
        "to be, or not to be",
        "to die, to sleep",
        "the slings and arrows"
    ]

    for seed in seeds:
        print(f"\n>>> Seed Phrase: '{seed}'")
        generated_sample = generate_text(
            model=model_exp2,
            seed_phrase=seed,
            gen_length=80,
            seq_length=seq_len2,
            char2idx=char2idx,
            idx2char=idx2char,
            temperature=0.6
        )
        print(f"Generated Continuation:\n\"{generated_sample}\"\n")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("  EXECUTION & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
