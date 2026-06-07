import os
import numpy as np
import torch


def get_simulator_data(data_file):
    """Load the Geant4 table and return (p0, step, p1) tensors.

    p0 = [0, 0, p0],  p1 = [dPt, 0, p0 + dPz]. The transverse kick is placed
    along x; its azimuth is random and is restored at inference.
    """
    n = os.path.getsize(data_file) // 16
    d = np.memmap(data_file, dtype=np.float32, mode="r", shape=(n, 4))
    p0   = np.exp(d[:, 0])
    step = np.exp(d[:, 1])
    dPt  = np.exp(d[:, 2]) * p0
    dPz  = -np.exp(d[:, 3]) * p0

    p0_vec = np.stack([np.zeros_like(p0), np.zeros_like(p0), p0], axis=-1)
    p1_vec = np.stack([dPt, np.zeros_like(dPt), p0 + dPz], axis=-1)
    return (torch.tensor(p0_vec, dtype=torch.float32),
            torch.tensor(step,   dtype=torch.float32),
            torch.tensor(p1_vec, dtype=torch.float32))


if __name__ == "__main__":
    # Exploratory look at how the energy loss (dPz) and scattering (dPt) depend
    # on the step size.
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.stats import binned_statistic

    out_dir = "outputs/plots"
    os.makedirs(out_dir, exist_ok=True)

    # decode raw log-columns in float64 (avoids the float32 cancellation that
    # would kill the tiny dPz if reconstructed as p1 - p0)
    n = os.path.getsize("data/training_data_easy.bin") // 16
    d = np.asarray(np.memmap("data/training_data_easy.bin", dtype=np.float32,
                             mode="r", shape=(n, 4))).astype(np.float64)
    p0, step = np.exp(d[:, 0]), np.exp(d[:, 1])
    rel_dPt, rel_dPz = np.exp(d[:, 2]), np.exp(d[:, 3])
    dPt, dPz = rel_dPt * p0, -rel_dPz * p0

    # power-law slopes: ~1 for Bethe-Bloch, ~0.5 for diffusive scattering
    a_z = np.polyfit(np.log(step), np.log(rel_dPz), 1)[0]
    a_t = np.polyfit(np.log(step), np.log(rel_dPt), 1)[0]

    step_bins = np.logspace(np.log10(step.min()), np.log10(step.max()), 40)
    centers = np.sqrt(step_bins[:-1] * step_bins[1:])

    def prof(y, stat):
        return binned_statistic(step, y, statistic=stat, bins=step_bins)[0]

    counts = np.histogram(step, bins=step_bins)[0]
    imax = int(counts.argmax())
    lo, hi = step_bins[imax], step_bins[imax + 1]
    mask = (step >= lo) & (step < hi)

    print(f"loaded {len(step):,} samples")
    print(f"  step     : {step.min():.2e} .. {step.max():.2e}  (median {np.median(step):.2e})")
    print(f"  p0       : {p0.min():.1f} .. {p0.max():.1f} GeV")
    print(f"  |dPz|/p0 ~ step^{a_z:.3f}   (expect ~1.0)")
    print(f"  dPt /p0  ~ step^{a_t:.3f}   (expect ~0.5)")

    # Figure 1: distributions, scaling, and the two "rates"
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    ax[0, 0].hist(step, bins=step_bins)
    ax[0, 0].axvspan(lo, hi, color="orange", alpha=0.35, label="most-frequent bin")
    ax[0, 0].set(xscale="log", yscale="log", xlabel="step [m]", ylabel="counts",
                 title="step-size distribution")
    ax[0, 0].legend()

    ax[0, 1].hist(p0, bins=50)
    ax[0, 1].set(xlabel="p0 [GeV]", ylabel="counts", title="initial momentum p0")

    mz, mt = prof(rel_dPz, "mean"), prof(rel_dPt, "mean")
    ax[0, 2].loglog(centers, mz, "o-", ms=3, label=r"$\langle|dP_z|/p_0\rangle$")
    ax[0, 2].loglog(centers, mt, "s-", ms=3, label=r"$\langle dP_t/p_0\rangle$")
    ax[0, 2].loglog(centers, mz[imax] * (centers / centers[imax]) ** 1.0, "k--", lw=1,
                    label=r"$\propto$ step$^{1}$")
    ax[0, 2].loglog(centers, mt[imax] * (centers / centers[imax]) ** 0.5, "k:", lw=1,
                    label=r"$\propto$ step$^{1/2}$")
    ax[0, 2].set(xlabel="step [m]", ylabel="mean relative change", title="scaling with step")
    ax[0, 2].legend(fontsize=8)

    ax[1, 0].semilogx(centers, prof(rel_dPz, "mean") / centers, "o-", ms=3, label="mean")
    ax[1, 0].semilogx(centers, prof(rel_dPz, "median") / centers, "x--", ms=3, label="median")
    ax[1, 0].set(xlabel="step [m]", ylabel=r"$\langle|dP_z|/p_0\rangle\,/\,$step",
                 title="energy-loss rate (flat if $dP_z\\propto$ step)")
    ax[1, 0].legend()

    ax[1, 1].semilogx(centers, prof(rel_dPt, "mean") / np.sqrt(centers), "o-", ms=3, label="mean")
    ax[1, 1].semilogx(centers, prof(rel_dPt, "median") / np.sqrt(centers), "x--", ms=3, label="median")
    ax[1, 1].set(xlabel="step [m]", ylabel=r"$\langle dP_t/p_0\rangle\,/\,\sqrt{\mathrm{step}}$",
                 title="scattering rate (flat if $dP_t\\propto\\sqrt{\\mathrm{step}}$)")
    ax[1, 1].legend()

    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 0.5,
                  f"samples : {len(step):,}\n"
                  f"step    : {step.min():.1e} - {step.max():.1e} m\n"
                  f"p0      : {p0.min():.0f} - {p0.max():.0f} GeV\n\n"
                  f"|dPz|/p0 ~ step^{a_z:.2f}  (expect 1.0)\n"
                  f"dPt /p0  ~ step^{a_t:.2f}  (expect 0.5)",
                  fontsize=11, family="monospace", va="center")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/data_overview.png", dpi=150)

    # Figure 2: joint distributions, both axes log
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
    for a, y, name in [(ax2[0], rel_dPz, "|dP_z|/p_0"), (ax2[1], rel_dPt, "dP_t/p_0")]:
        h = a.hist2d(np.log10(step), np.log10(y), bins=150, norm=LogNorm(), cmap="viridis")
        fig2.colorbar(h[3], ax=a, label="counts")
        a.set(xlabel=r"$\log_{10}$ step", ylabel=rf"$\log_{{10}}\ {name}$", title=f"{name} vs step")
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/data_2d_step_vs_change.png", dpi=150)

    # Figure 3: conditional distributions at the most-frequent step
    fig3, ax3 = plt.subplots(1, 2, figsize=(14, 5))
    for a, y, name in [(ax3[0], rel_dPz[mask], "|dP_z|/p_0"), (ax3[1], rel_dPt[mask], "dP_t/p_0")]:
        bins = np.logspace(np.log10(y.min()), np.log10(y.max()), 80)
        a.hist(y, bins=bins)
        a.axvline(y.mean(), color="r", ls="--", label=f"mean={y.mean():.2e}")
        a.axvline(np.median(y), color="k", ls=":", label=f"median={np.median(y):.2e}")
        a.set(xscale="log", yscale="log", xlabel=rf"${name}$", ylabel="counts",
              title=f"{name} at step $\\approx${centers[imax]:.1e} m")
        a.legend()
    fig3.tight_layout()
    fig3.savefig(f"{out_dir}/data_fixed_step.png", dpi=150)

    print(f"saved 3 figures to {out_dir}/")
