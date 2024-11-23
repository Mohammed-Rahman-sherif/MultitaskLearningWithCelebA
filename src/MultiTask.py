# %%
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import Normalize, Compose, ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import Architecture

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Accelerator: {device}")

# %%
torch.manual_seed(33)
model = Architecture.MTArchitecture().to(device=device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Reduce LR every 5 epochs
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# %%
transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = CelebA(
    root="data", split="train", transform=transform, download=True)
test_data = CelebA(
    root="data", split="test", transform=transform, download=True)

# %%
batch_size = 256
train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

# %%
epochs = 10  # Increased epochs for better training and evaluation
train_losses = []
test_losses = []
best_test_wts = None
best_test_loss = float("inf")

for itr in range(epochs):
    batch_loss = 0.0
    model.train()
    for batch, (X_train, y_train) in enumerate(train_loader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_train = y_train[:, :40].float()
        optimizer.zero_grad()
        y_pred = model(X_train)

        head_loss = 0.0
        for idx, head_output in enumerate(y_pred):
            target = y_train[:, idx].view(-1, 1)
            train_loss = criterion(head_output, target)
            head_loss += train_loss

        # Normalize head_loss by the number of heads
        head_loss /= len(y_pred)

        head_loss.backward()
        optimizer.step()

        batch_loss += head_loss.item()

        if batch % 10 == 0:
            print(
                f"Epoch {itr+1}/{epochs}, Batch: [{batch}/{len(train_loader)}], Train Loss: {head_loss.item()}"
            )

        # if batch == 50: break
    train_losses.append(batch_loss / len(train_loader))

    model.eval()
    test_batch_loss = 0.0
    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_test = y_test[:, :40].float()
            y_val = model(X_test)
            head_val_loss = 0.0
            for idx, head_val_output in enumerate(y_val):
                target_val = y_test[:, idx].view(-1, 1)
                test_loss = criterion(head_val_output, target_val)
                head_val_loss += test_loss.item()
            # Normalize the validation loss by the number of heads
            head_val_loss /= len(y_val)
            test_batch_loss += head_val_loss

    avg_test_loss = test_batch_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_test_wts = model.state_dict()

    scheduler.step()  # Step the scheduler at the end of each epoch

    print(
        f"Epoch {itr+1}/{epochs}, Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}"
    )

if best_test_wts is not None:
    torch.save(best_test_wts, f"models/best_CelebA_{best_test_loss}.pt")

# %%
plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
plt.title("Training and Testing Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# %%
torch.save(model.state_dict(), "models/last_CelebA_fully_trained.pt")