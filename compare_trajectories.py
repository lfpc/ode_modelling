import numpy as np
import torch
import matplotlib.pyplot as plt
from ode import NeuralODE, NeuralCorrection


# --- data helpers ---

def run_geant4_ensemble(initial_state, n_runs, mag_field):
    """
    Run Geant4 n_runs times from the same initial_state.
    initial_state: [x, y, z, px, py, pz, charge]
    Returns list of arrays (n_steps+1, 6): [x,y,z,px,py,pz] per run.
    """
    ic = np.array([initial_state] * n_runs)
    muon_data = run_g4(ic, mag_field)
    trajs = []
    for m in muon_data:
        traj = np.stack([m['x'], m['y'], m['z'], m['px'], m['py'], m['pz']], axis=-1)
        trajs.append(traj)
    return trajs  # list of (n_steps+1, 6)


def step_sizes_from_positions(traj):
    """Compute arc-length step sizes from consecutive positions."""
    dr = np.diff(traj[:, :3], axis=0)
    return np.linalg.norm(dr, axis=-1)  # (n_steps,)


def load_model(weights_path, device):
    transport = NeuralCorrection(hidden=64, layers=3).to(device)
    transport.load_state_dict(torch.load(weights_path, map_location=device))
    model = NeuralODE(charge=-1.0, field_fn=None, transport=transport).to(device)
    model.eval()
    return model


# --- prediction ---

def predict_neural_ode(model, initial_state, step_sizes, device, stochastic=False):
    """
    Predict momentum trajectory with NeuralODE.
    initial_state: (6,) array [x,y,z,px,py,pz]  (position ignored)
    step_sizes:    (n_steps,) array
    Returns (n_steps+1, 3) numpy array of momenta.
    """
    p0 = torch.tensor(initial_state[3:], dtype=torch.float32, device=device).unsqueeze(0)
    ds = torch.tensor(step_sizes,        dtype=torch.float32, device=device)
    with torch.no_grad():
        traj = model.momentum_trajectory(p0, ds, stochastic=stochastic)
    return traj[0].cpu().numpy()  # (n_steps+1, 3)


# --- plots ---

def plot_comparison(g4_trajs, neural_traj, neural_trajs, save_dir="outputs/plots"):
    # G4: (n_runs, n_steps+1, 6) — columns 3,4,5 are px,py,pz
    g4      = np.stack(g4_trajs)
    g4_mom  = g4[:, :, 3:]                            # (n_runs, n_steps+1, 3)
    g4_mean = g4_mom.mean(axis=0)
    g4_std  = g4_mom.std(axis=0)
    # NeuralODE stochastic ensemble: (n_runs, n_steps+1, 3)
    neu     = np.stack(neural_trajs)
    steps   = np.arange(g4_mom.shape[1])

    # --- 1. momentum components vs step ---
    labels = ["px", "py", "pz"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        for run in g4_mom:
            ax.plot(steps, run[:, i], color="steelblue", alpha=0.1, lw=0.7)
        for run in neu:
            ax.plot(steps, run[:, i], color="tomato", alpha=0.1, lw=0.7)
        ax.plot(steps, g4_mean[:, i],        color="steelblue", lw=2, ls="--", label="G4 mean")
        ax.plot(steps, neural_traj[:, i],    color="tomato",    lw=2,           label="NODE det.")
        ax.fill_between(steps,
                        g4_mean[:, i] - g4_std[:, i],
                        g4_mean[:, i] + g4_std[:, i],
                        color="steelblue", alpha=0.2, label="G4 ±1σ")
        ax.set_xlabel("step"); ax.set_ylabel(lbl); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/momentum_vs_step.png", dpi=150)
    plt.close(fig)

    # --- 2. final-state distribution: pT vs pz scatter ---
    g4_pT   = np.sqrt(g4_mom[:, -1, 0]**2 + g4_mom[:, -1, 1]**2)
    g4_pz   = g4_mom[:, -1, 2]
    neu_pT  = np.sqrt(neu[:, -1, 0]**2    + neu[:, -1, 1]**2)
    neu_pz  = neu[:, -1, 2]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(g4_pz,  g4_pT,  s=10, alpha=0.4, color="steelblue", label="G4")
    ax.scatter(neu_pz, neu_pT, s=10, alpha=0.4, color="tomato",    label="NeuralODE")
    ax.set_xlabel("pz final (GeV)"); ax.set_ylabel("pT final (GeV)"); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_dir}/final_state_distribution.png", dpi=150)
    plt.close(fig)

    # --- 3. pz and |pT| mean ± std: G4 vs NeuralODE ---
    neu_mean = neu.mean(axis=0)
    neu_std  = neu.std(axis=0)
    g4_pT_traj  = np.sqrt(g4_mom[:, :, 0]**2 + g4_mom[:, :, 1]**2)
    neu_pT_traj = np.sqrt(neu[:, :, 0]**2     + neu[:, :, 1]**2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (g4_v, neu_v, lbl) in zip(axes, [
        (g4_pT_traj, neu_pT_traj, "|pT| (GeV)"),
        (g4_mom[:, :, 2], neu[:, :, 2], "pz (GeV)"),
    ]):
        ax.fill_between(steps, g4_v.mean(0)-g4_v.std(0), g4_v.mean(0)+g4_v.std(0),
                        alpha=0.25, color="steelblue")
        ax.fill_between(steps, neu_v.mean(0)-neu_v.std(0), neu_v.mean(0)+neu_v.std(0),
                        alpha=0.25, color="tomato")
        ax.plot(steps, g4_v.mean(0),  color="steelblue", lw=2, label="G4 mean")
        ax.plot(steps, neu_v.mean(0), color="tomato",    lw=2, label="NODE mean")
        ax.set_xlabel("step"); ax.set_ylabel(lbl); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/pt_pz_bands.png", dpi=150)
    plt.close(fig)

    print("Plots saved to", save_dir)


# --- main ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0",        type=float, default=10.0)
    parser.add_argument("--n_runs",    type=int,   default=200)
    parser.add_argument("--n_steps",   type=int,   default=50)
    parser.add_argument("--mag_field", type=str,   default="toy")
    parser.add_argument("--weights",   type=str,   default="outputs/transport.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial_state = np.array([0., 0., 0., 0., 0., args.p0, -1.])  # [x,y,z,px,py,pz,charge]

    print(f"Running {args.n_runs} Geant4 simulations...")
    g4_trajs = run_geant4_ensemble(initial_state, args.n_runs, args.mag_field)

    # Use step sizes from first G4 run as reference for NeuralODE
    ref_steps = step_sizes_from_positions(g4_trajs[0])

    print("Loading NeuralODE and predicting trajectory...")
    model = load_model(args.weights, device)
    # One deterministic + N stochastic NeuralODE trajectories to match G4 ensemble
    neural_traj  = predict_neural_ode(model, initial_state[:6], ref_steps, device, stochastic=False)
    neural_trajs = [predict_neural_ode(model, initial_state[:6], ref_steps, device, stochastic=True)
                    for _ in range(args.n_runs)]

    plot_comparison(g4_trajs, neural_traj, neural_trajs)
