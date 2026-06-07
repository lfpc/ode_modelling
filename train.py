import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from ode import NeuralODE, NeuralCorrection, NeuralSDE
from data_utils import get_simulator_data

data_file = "data/training_data_easy.bin"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rotate_z_pair(p0, p1):
    """Rotate p0 and p1 by the same random azimuth, so the SDE sees the isotropic
    scattering distribution instead of the fixed-x convention used in the data."""
    phi = torch.rand(p0.shape[0], device=p0.device) * 2 * torch.pi
    c, s = phi.cos(), phi.sin()

    def rot(p):
        px = c * p[:, 0] - s * p[:, 1]
        py = s * p[:, 0] + c * p[:, 1]
        return torch.stack([px, py, p[:, 2]], dim=-1)

    return rot(p0), rot(p1)


def train(n_epochs=10, lr=1e-3, batch_size=1024, device=None, deterministic=False):
    if device is None:
        device = pick_device()
    mode = ("deterministic ODE (RK4 + MSE)" if deterministic
            else "stochastic SDE (Euler-Maruyama + Gaussian NLL)")
    print(f"Using device: {device}")
    print(f"Mode: {mode}")
    os.makedirs("outputs/plots", exist_ok=True)

    if deterministic:
        transport = NeuralCorrection(hidden=64, layers=3).to(device)
        model = NeuralODE(transport).to(device)
        trainable = transport
    else:
        model = NeuralSDE(hidden=64, layers=3).to(device)
        trainable = model

    optimizer = torch.optim.Adam(trainable.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, min_lr=1e-6)

    # p0: (N, 3), steps: (N,), p1: (N, 3)
    p0, steps, p1 = get_simulator_data(data_file)

    n_val = max(1, int(0.1 * len(p0)))
    train_ds = TensorDataset(p0[n_val:], steps[n_val:], p1[n_val:])
    val_ds   = TensorDataset(p0[:n_val], steps[:n_val], p1[:n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    def batch_loss(s0, ds, s1):
        s0, ds, s1 = s0.to(device), ds.to(device), s1.to(device)
        ds = ds.unsqueeze(-1)
        if deterministic:
            pred = model._rk4_momentum(s0, ds)
            p_mag = s0.norm(dim=-1, keepdim=True)
            return nn.functional.mse_loss((pred - s0) / p_mag, (s1 - s0) / p_mag)
        s0, s1 = _rotate_z_pair(s0, s1)
        return model.nll(s0, ds, s1).mean()

    loss_name = "MSE" if deterministic else "NLL"

    @torch.no_grad()
    def eval_loss(loader):
        model.eval()
        return sum(batch_loss(s0, ds, s1).item() for s0, ds, s1 in loader) / len(loader)

    # epoch-0 baseline: loss of the untrained model
    init_train, init_val = eval_loss(train_loader), eval_loss(val_loader)
    train_losses, val_losses = [init_train], [init_val]
    best_val = init_val
    best_state = {k: v.cpu().clone() for k, v in trainable.state_dict().items()}
    print(f"epoch  init  train {init_train:.6f}  val {init_val:.6f}  [{loss_name}]")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for s0, ds, s1 in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(s0, ds, s1)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)

        avg_val = eval_loss(val_loader)

        scheduler.step(avg_val)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in trainable.state_dict().items()}

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:4d}  train {avg_train:.6f}  val {avg_val:.6f}  "
              f"lr {lr_now:.2e}  [{loss_name}]")

    trainable.load_state_dict(best_state)
    out_path = "outputs/transport.pt" if deterministic else "outputs/sde.pt"
    torch.save(trainable.state_dict(), out_path)
    print(f"saved model to {out_path}")
    plot_loss(train_losses, val_losses, loss_name,
              path=f"outputs/plots/loss_{'ode' if deterministic else 'sde'}.png")
    return model


def plot_loss(train_losses, val_losses, loss_name="loss", path="outputs/plots/loss.png"):
    fig, ax = plt.subplots()
    epochs = range(len(train_losses))   # 0 = before training
    ax.plot(epochs, train_losses, marker="o", ms=3, label="train")
    ax.plot(epochs, val_losses,   marker="o", ms=3, label="val")
    ax.axvline(0, color="gray", ls=":", lw=1, label="before training")
    ax.set_xlabel("epoch (0 = before training)")
    ax.set_ylabel(f"{loss_name} loss")
    if min(min(train_losses), min(val_losses)) > 0:   # NLL can be negative
        ax.set_yscale("log")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"loss curve saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the particle-transport model.")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use the deterministic NeuralODE (RK4 + MSE). "
             "Omit this flag (default) to train the stochastic SDE "
             "(Euler-Maruyama + Gaussian NLL).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    train(n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
          deterministic=args.deterministic)
