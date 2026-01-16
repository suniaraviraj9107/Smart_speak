import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================= CONFIG =================
DATA_DIR = r"C:\Users\risha\Downloads\asl_landmarks"

CLASSES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z"
]

SEQ_LEN = 16
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ================= DATASET =================
class ASLDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []

        for idx, letter in enumerate(CLASSES):
            folder = os.path.join(DATA_DIR, letter)

            if not os.path.exists(folder):
                raise ValueError(f"❌ Missing folder: {folder}")

            files = os.listdir(folder)
            if len(files) == 0:
                raise ValueError(f"❌ No samples in folder: {folder}")

            for file in files:
                path = os.path.join(folder, file)

                seq = np.load(path)              # (16,21,3)
                seq = seq.reshape(SEQ_LEN, -1)   # (16,63)

                self.data.append(seq)
                self.labels.append(idx)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        print("Total samples:", len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = ASLDataset()
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# ================= MODEL =================
class ASLLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=63,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)        # (B, T, 256)
        out = out[:, -1, :]         # last timestep
        out = self.fc(out)          # (B, 26)
        return out


model = ASLLSTM(len(CLASSES)).to(DEVICE)
print(model)

# ================= TRAINING =================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)

    acc = 100 * correct / total

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss:.4f} | Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "asl_lstm_A_Z.pth")
        print("✅ Best model saved")

print("\n🏁 Training complete")
print("🏆 Best accuracy:", best_acc)
print("📦 Model file saved as: asl_lstm_A_Z.pth")
