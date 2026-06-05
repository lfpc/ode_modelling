import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from ode import NeuralODE, NeuralCorrection
from data_utils import get_simulator_data

data_file = "../cuda_muons_generative/data/training_data_easy.bin"

def train(n_epochs=200, lr=1e-3, batch_size=256, device=None):
    if device is None:
        device = torch.device("cuda")

    transport = NeuralCorrection(hidden=64, layers=3).to(device)
    model = NeuralODE(charge=-1.0, field_fn=None, transport=transport).to(device)
    optimizer = torch.optim.Adam(transport.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)

    # p0: (N, 3), steps: (N,), p1: (N, 3)
    p0, steps, p1 = get_simulator_data(data_file)

    n_val = max(1, int(0.1 * len(p0)))
    train_ds = TensorDataset(p0[n_val:], steps[n_val:], p1[n_val:])
    val_ds   = TensorDataset(p0[:n_val], steps[:n_val], p1[:n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    best_val, best_state = float("inf"), None
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for s0, ds, s1 in train_loader:
            s0, ds, s1 = s0.to(device), ds.to(device), s1.to(device)
            optimizer.zero_grad()
            pred = model._rk4_momentum(s0, ds.unsqueeze(-1))
            p_mag = s0.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            loss = nn.functional.mse_loss((pred - s0) / p_mag, (s1 - s0) / p_mag)
            loss.backward()
            nn.utils.clip_grad_norm_(transport.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            avg_val = sum(
                nn.functional.mse_loss(
                    (model._rk4_momentum(s0.to(device), ds.to(device).unsqueeze(-1)) - s0.to(device))
                    / s0.to(device).norm(dim=-1, keepdim=True).clamp(min=1e-10),
                    (s1.to(device) - s0.to(device))
                    / s0.to(device).norm(dim=-1, keepdim=True).clamp(min=1e-10)
                ).item()
                for s0, ds, s1 in val_loader
            ) / len(val_loader)

        scheduler.step(avg_val)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in transport.state_dict().items()}

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:4d}  train {avg_train:.6f}  val {avg_val:.6f}  lr {lr_now:.2e}")

    transport.load_state_dict(best_state)
    torch.save(transport.state_dict(), "outputs/transport.pt")
    plot_loss(train_losses, val_losses)
    return model


def plot_loss(train_losses, val_losses, path="outputs/plots/loss.png"):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="train")
    ax.plot(val_losses,   label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"loss curve saved to {path}")


if __name__ == "__main__":
    epochs = 10
    train(n_epochs=epochs)
