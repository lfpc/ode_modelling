import torch
import numpy as np
import h5py
import os
from tqdm import tqdm

def get_simulator_data(data_file):
    """
    Returns (p0_vec, steps, p1_vec):
      p0_vec : (N, 3) float32 — initial momentum [0, 0, p0]
      steps  : (N,)   float32 — arc-length step size per sample
      p1_vec : (N, 3) float32 — final momentum [px, py, pz] after one step
    """
    def to_original(data):
        p0   = np.exp(data[:, 0])
        step = np.exp(data[:, 1])
        dPt  = np.exp(data[:, 2]) * p0
        dPz  = -np.exp(data[:, 3]) * p0
        return p0, dPt, dPz, step

    n_rows  = os.path.getsize(data_file) // (4 * 4)
    data_mm = np.memmap(data_file, dtype=np.float32, mode="r", shape=(n_rows, 4))
    p0, dPt, dPz, step = to_original(data_mm)

    # Always align the transverse kick along x — the azimuthal direction is
    # physically random (uniform in phi) and cannot be learned by a deterministic
    # network. We fix phi=0 here and apply a random rotation at inference time.
    pz = p0 + dPz

    p0_vec = np.stack([np.zeros_like(p0), np.zeros_like(p0), p0], axis=-1)
    p1_vec = np.stack([dPt, np.zeros_like(dPt), pz], axis=-1)

    return (
        torch.tensor(p0_vec, dtype=torch.float32),
        torch.tensor(step,   dtype=torch.float32),
        torch.tensor(p1_vec, dtype=torch.float32),
    )
