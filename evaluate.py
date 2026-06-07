"""Compare the learned models against textbook physics: energy loss vs.
Bethe-Bloch, scattering vs. Highland, energy-loss straggling vs. Bohr/Vavilov.

Units (the .bin carries none): momentum in GeV, step in metres, dPz in GeV.
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from ode import NeuralODE, NeuralCorrection, NeuralSDE

# muon / iron constants
K, Z_FE, A_FE, RHO_FE = 0.307075, 26.0, 55.845, 7.874   # K in MeV cm^2/mol
I_FE, ME, MMU, X0_FE = 286e-6, 0.510999, 105.6584, 1.757  # MeV, MeV, MeV, cm
_DE = dict(C=4.2911, a=0.14680, m=2.9632, x0=-0.0012, x1=3.1531, d0=0.12)  # Sternheimer, iron


def _density_effect(bg):
    x, ln10 = np.log10(bg), np.log(10.0)
    d = np.where(x >= _DE['x1'], 2 * ln10 * x - _DE['C'],
                 2 * ln10 * x - _DE['C'] + _DE['a'] * (_DE['x1'] - x) ** _DE['m'])
    d = np.where(x < _DE['x0'], _DE['d0'] * 10.0 ** (2 * (x - _DE['x0'])), d)
    return d


def bethe_bloch_dEdx(p_GeV):
    """Mean stopping power of a muon in iron [MeV/cm]."""
    p = np.asarray(p_GeV, dtype=float) * 1000.0
    E = np.sqrt(p ** 2 + MMU ** 2)
    gamma, beta2, bg = E / MMU, (p / E) ** 2, p / MMU
    Wmax = 2 * ME * bg ** 2 / (1 + 2 * gamma * ME / MMU + (ME / MMU) ** 2)
    bracket = 0.5 * np.log(2 * ME * bg ** 2 * Wmax / I_FE ** 2) - beta2 - _density_effect(bg) / 2
    return K * (Z_FE / A_FE) / beta2 * bracket * RHO_FE


def bethe_bloch_dPz(p_GeV, step_m):
    return bethe_bloch_dEdx(p_GeV) * (step_m * 100.0) / 1000.0   # MeV/cm * cm -> GeV


def highland_pT_rms(step_m, p_GeV):
    """Highland RMS of the total transverse momentum |pT| over one step [MeV]."""
    p = np.asarray(p_GeV, dtype=float) * 1000.0
    beta = p / np.sqrt(p ** 2 + MMU ** 2)
    xr = (np.asarray(step_m, dtype=float) * 100.0) / X0_FE
    theta0 = (13.6 / (beta * p)) * np.sqrt(xr) * (1 + 0.038 * np.log(xr / beta ** 2))
    return np.sqrt(2.0) * p * theta0


def vavilov_sigma_E(step_m, p_GeV):
    """Bohr/Vavilov straggling: RMS of the energy loss over one step [MeV]."""
    p = np.asarray(p_GeV, dtype=float) * 1000.0
    E = np.sqrt(p ** 2 + MMU ** 2)
    gamma, beta2, bg = E / MMU, (p / E) ** 2, p / MMU
    Wmax = 2 * ME * bg ** 2 / (1 + 2 * gamma * ME / MMU + (ME / MMU) ** 2)
    x_gcm2 = (np.asarray(step_m, dtype=float) * 100.0) * RHO_FE
    return np.sqrt((K / 2) * (Z_FE / A_FE) / beta2 * x_gcm2 * Wmax * (1 - beta2 / 2))


def load_data(data_file):
    n = os.path.getsize(data_file) // 16
    d = np.asarray(np.memmap(data_file, dtype=np.float32, mode="r", shape=(n, 4))).astype(np.float64)
    p0, step = np.exp(d[:, 0]), np.exp(d[:, 1])
    dPz, dPt = np.exp(d[:, 3]) * p0, np.exp(d[:, 2]) * p0
    return p0, step, dPz, dPt


def _to_input(p0, step):
    s0 = torch.zeros(len(p0), 3)
    s0[:, 2] = torch.tensor(p0, dtype=torch.float32)
    return s0, torch.tensor(step, dtype=torch.float32).unsqueeze(-1)


def model_dPz(weights, p0, step):
    transport = NeuralCorrection()
    transport.load_state_dict(torch.load(weights, map_location="cpu"))
    model = NeuralODE(transport).eval()
    with torch.no_grad():
        s0, ds = _to_input(p0, step)
        return (s0[:, 2] - model._rk4_momentum(s0, ds)[:, 2]).numpy()   # |dPz|, loss positive


def model_diffusion(weights, p0, step):
    model = NeuralSDE()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        s0, ds = _to_input(p0, step)
        sig = model.drift_diffusion(s0, ds)[1].exp().numpy()
    return sig[:, 0], sig[:, 1], sig[:, 2]


def momentum_dEdx_largestep(p0, step, dPz_data, dPz_model, step_min, step_max):
    """dE/dx vs momentum in the self-averaging window where data -> Bethe-Bloch."""
    sel = (step >= step_min) & (step <= step_max)
    p_bins = np.linspace(p0.min(), p0.max(), 15)
    p_cent = 0.5 * (p_bins[:-1] + p_bins[1:])
    binmean = lambda y, by: binned_statistic(by, y, "mean", bins=p_bins)[0]

    dedx_data = binmean((dPz_data[sel] / step[sel]) * 10.0, p0[sel])
    dedx_model = binmean((dPz_model[sel] / step[sel]) * 10.0, p0[sel])

    print(f"\nLarge-step window [{step_min*100:.2f}, {step_max*100:.1f}] cm ({sel.sum():,} samples):")
    print(f"  data  <dE/dx> = {((dPz_data[sel]/step[sel])*10).mean():.2f} MeV/cm")
    print(f"  model <dE/dx> = {((dPz_model[sel]/step[sel])*10).mean():.2f} MeV/cm")
    print(f"  Bethe-Bloch   = {bethe_bloch_dEdx(p0[sel].mean()):.2f} MeV/cm")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(p_cent, dedx_data, "o-", ms=4, label="data (Geant4)")
    ax.plot(p_cent, dedx_model, "s-", ms=4, label="NeuralODE")
    ax.plot(p_cent, bethe_bloch_dEdx(p_cent), "k--", lw=1.5, label="Bethe-Bloch")
    ax.set(xlabel="p0 [GeV]", ylabel="dE/dx [MeV/cm]",
           title=f"stopping power vs momentum ({step_min*100:.2g}-{step_max*100:.2g} cm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/plots/bethe_bloch_vs_momentum_largestep.png", dpi=150)
    plt.close(fig)


def evaluate(weights="outputs/transport.pt", data_file="data/training_data_easy.bin",
             step_min=2e-3, step_max=5e-2):
    os.makedirs("outputs/plots", exist_ok=True)
    p0, step, dPz_data, _ = load_data(data_file)
    dPz_model = model_dPz(weights, p0, step)
    dPz_bb = bethe_bloch_dPz(p0, step)

    step_bins = np.logspace(np.log10(step.min()), np.log10(step.max()), 40)
    centers = np.sqrt(step_bins[:-1] * step_bins[1:])
    binmean = lambda y, by=step, bins=step_bins: binned_statistic(by, y, "mean", bins=bins)[0]

    clean = (step > 3e-3) & (step < 5e-2)
    bb_ref = bethe_bloch_dEdx(p0.mean())
    print(f"Bethe-Bloch (muon in iron) at p={p0.mean():.1f} GeV : {bb_ref:.2f} MeV/cm")
    for label, y in [("data", dPz_data), ("model", dPz_model)]:
        print(f"  {label:5s} <dPz>/step (clean window) : {(y[clean]/step[clean]).mean()*10:.2f} MeV/cm")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].loglog(centers, binmean(dPz_data), "o-", ms=3, label="data (Geant4)")
    ax[0].loglog(centers, binmean(dPz_model), "s-", ms=3, label="NeuralODE")
    ax[0].loglog(centers, binmean(dPz_bb), "k--", lw=1.5, label="Bethe-Bloch")
    ax[0].set(xlabel="step [m]", ylabel=r"$\langle|dP_z|\rangle$ [GeV]", title="mean energy loss vs step")
    ax[0].legend()

    ax[1].semilogx(centers, binmean(dPz_data) / centers * 10.0, "o-", ms=3, label="data")
    ax[1].semilogx(centers, binmean(dPz_model) / centers * 10.0, "s-", ms=3, label="NeuralODE")
    ax[1].axhline(bb_ref, color="k", ls="--", lw=1.5, label=f"Bethe-Bloch ({bb_ref:.1f} MeV/cm)")
    ax[1].axvspan(3e-3, 5e-2, color="green", alpha=0.08, label="clean window")
    ax[1].set(xlabel="step [m]", ylabel=r"$\langle|dP_z|\rangle/$step [MeV/cm]",
              title="stopping power vs step", ylim=(0, 4 * bb_ref))
    ax[1].legend(fontsize=8)

    p_bins = np.linspace(p0.min(), p0.max(), 15)
    p_cent = 0.5 * (p_bins[:-1] + p_bins[1:])
    ax[2].plot(p_cent, binmean((dPz_data / step)[clean] * 10.0, p0[clean], p_bins), "o-", ms=3, label="data")
    ax[2].plot(p_cent, binmean((dPz_model / step)[clean] * 10.0, p0[clean], p_bins), "s-", ms=3, label="NeuralODE")
    ax[2].plot(p_cent, bethe_bloch_dEdx(p_cent), "k--", lw=1.5, label="Bethe-Bloch")
    ax[2].set(xlabel="p0 [GeV]", ylabel="dE/dx [MeV/cm]", title="stopping power vs momentum (clean window)")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig("outputs/plots/bethe_bloch_comparison.png", dpi=150)
    plt.close(fig)
    print("saved outputs/plots/bethe_bloch_comparison.png")

    momentum_dEdx_largestep(p0, step, dPz_data, dPz_model, step_min, step_max)


def evaluate_sde_diffusion(weights="outputs/sde.pt", data_file="data/training_data_easy.bin"):
    os.makedirs("outputs/plots", exist_ok=True)
    p0, step, dPz, dPt = load_data(data_file)
    sx, sy, sz = model_diffusion(weights, p0, step)

    step_bins = np.logspace(np.log10(step.min()), np.log10(step.max()), 40)
    centers = np.sqrt(step_bins[:-1] * step_bins[1:])
    bm = lambda y, stat="mean": binned_statistic(step, y, stat, bins=step_bins)[0]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(centers, np.sqrt(bm(dPt ** 2)) * 1000, "o-", ms=3, label="data  RMS$|p_T|$")
    ax[0].loglog(centers, bm(np.sqrt(sx ** 2 + sy ** 2) * np.sqrt(step)) * 1000, "s-", ms=3,
                 label=r"SDE  $\sqrt{\sigma_x^2+\sigma_y^2}\sqrt{ds}$")
    ax[0].loglog(centers, bm(highland_pT_rms(step, p0)), "k--", lw=1.5, label="Highland scattering")
    ax[0].set(xlabel="step [m]", ylabel=r"RMS $|p_T|$ [MeV]", title="transverse diffusion vs multiple scattering")
    ax[0].legend()

    ax[1].loglog(centers, bm(dPz, "std") * 1000, "o-", ms=3, label=r"data  std($dP_z$)")
    ax[1].loglog(centers, bm(sz * np.sqrt(step)) * 1000, "s-", ms=3, label=r"SDE  $\sigma_z\sqrt{ds}$")
    ax[1].loglog(centers, bm(vavilov_sigma_E(step, p0)), "k--", lw=1.5, label="Bohr/Vavilov straggling")
    ax[1].set(xlabel="step [m]", ylabel=r"width of $dP_z$ [MeV]", title="longitudinal diffusion vs straggling")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig("outputs/plots/sde_diffusion_comparison.png", dpi=150)
    plt.close(fig)

    clean = (step > 1e-2) & (step < 5e-2)
    print("\nSDE diffusion (self-averaging window 1-5 cm):")
    print(f"  transverse  RMS|pT|: data {np.sqrt((dPt[clean]**2).mean())*1000:.2f}  "
          f"model {(np.sqrt(sx[clean]**2+sy[clean]**2)*np.sqrt(step[clean])).mean()*1000:.2f}  "
          f"Highland {highland_pT_rms(step[clean], p0[clean]).mean():.2f}  MeV")
    print(f"  longitudinal std(dPz): data {dPz[clean].std()*1000:.2f}  "
          f"model {(sz[clean]*np.sqrt(step[clean])).mean()*1000:.2f}  "
          f"Bohr/Vavilov {vavilov_sigma_E(step[clean], p0[clean]).mean():.2f}  MeV")
    print("saved outputs/plots/sde_diffusion_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare learned models to textbook physics.")
    parser.add_argument("--weights", default="outputs/transport.pt")
    parser.add_argument("--data", default="data/training_data_easy.bin")
    parser.add_argument("--step-min", type=float, default=2e-3)
    parser.add_argument("--step-max", type=float, default=5e-2)
    parser.add_argument("--sde-weights", default="outputs/sde.pt")
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.step_min, args.step_max)
    if os.path.exists(args.sde_weights):
        evaluate_sde_diffusion(args.sde_weights, args.data)
