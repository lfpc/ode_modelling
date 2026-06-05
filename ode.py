import torch
from torch._prims_common import corresponding_complex_dtype, dtype_or_default
import torch.nn as nn



def LorentzForce(state, charge, field_fn):
    pos, mom = state[..., :3], state[..., 3:]
    p_mag = mom.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    p_hat = mom / p_mag
    B = torch.from_numpy(field_fn(pos)).to(state.device, dtype=state.dtype)
    dp = charge * (torch.cross(p_hat, B, dim=-1))
    return dp


class NeuralCorrection(nn.Module):
    """MLP that predicts dp/ds. Takes [p, ds] as input (4D) so it can
    capture both the ODE part (dPz ~ ds) and the diffusion part (dPt ~ sqrt(ds))."""

    def __init__(self, hidden=64, layers=3):
        super().__init__()
        net = [nn.Linear(4, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*net)

    def forward(self, p, ds):
        # ds: (batch, 1) or scalar broadcast; cat along last dim
        ds_in = ds.expand(*p.shape[:-1], 1)
        return self.net(torch.cat([p, ds_in], dim=-1))


class NeuralODE(nn.Module):
    """
    Lorentz force + neural transport, integrated over arc length s.
    State: [x, y, z, px, py, pz].
    """

    def __init__(self, charge, field_fn = None, transport: NeuralCorrection = None):
        super().__init__()
        self.charge = charge
        self.field_fn = field_fn
        self.lorentz = LorentzForce
        self.transport = transport

    def forward(self, state):
        #dx/ds = p_hat
        p = state[..., 3:]
        p_hat = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        #dp/ds = Lorentz + transport
        lorentz = self.lorentz(state, self.charge, self.field_fn) if self.field_fn is not None else torch.zeros_like(p)
        # ds unknown in the full-state forward pass — pass zeros as placeholder
        particle_matter = self.transport(state[..., 3:], torch.zeros(*p.shape[:-1], 1, device=p.device)) if self.transport is not None else torch.zeros_like(p)
        dp = lorentz + particle_matter
        return torch.cat([p_hat, dp], dim=-1) 

    def _rk4(self, state, ds):
        k1 = self(state)
        k2 = self(state + ds / 2 * k1)
        k3 = self(state + ds / 2 * k2)
        k4 = self(state + ds * k3)
        return state + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _rk4_momentum(self, p, ds):
        # ds: (batch, 1) — passed to transport so it can use it directly
        k1 = self.transport(p,                ds)
        k2 = self.transport(p + ds / 2 * k1, ds)
        k3 = self.transport(p + ds / 2 * k2, ds)
        k4 = self.transport(p + ds * k3,     ds)
        return p + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def _random_rotate_z(p):
        """Apply a random azimuthal rotation around z per sample in the batch."""
        theta = torch.rand(p.shape[0], device=p.device) * 2 * torch.pi
        c, s = theta.cos(), theta.sin()
        px = c * p[:, 0] - s * p[:, 1]
        py = s * p[:, 0] + c * p[:, 1]
        return torch.stack([px, py, p[:, 2]], dim=-1)

    def momentum_trajectory(self, p0, step_sizes, stochastic=False):
        """Integrate dp/ds only (no position). Returns (batch, n_steps+1, 3).
        stochastic: if True, apply a random azimuthal rotation after each step
                    to simulate multiple-scattering direction randomness.
        """
        step_sizes = torch.as_tensor(step_sizes, device=p0.device)
        momenta, p = [p0], p0
        for ds in step_sizes:
            p = self._rk4_momentum(p, ds)
            if stochastic:
                p = self._random_rotate_z(p)
            momenta.append(p)
        return torch.stack(momenta, dim=1)

    def trajectory(self, state0, step_sizes, n_steps=None):
        """
        Integrate over arc length. Returns (batch, n_steps+1, 6).

        step_sizes: scalar or 0-d tensor → uniform steps, n_steps required
                    1-D tensor (n_steps,)  → variable steps (Geant4 output)
        """
        step_sizes = torch.as_tensor(step_sizes, device=state0.device) 
        if step_sizes.ndim == 0:
            assert n_steps is not None
            step_sizes = step_sizes.expand(n_steps)
        states, state = [state0], state0
        for ds in step_sizes:
            state = self._rk4(state, ds)
            states.append(state)
        return torch.stack(states, dim=1)

if __name__ == "__main__":
    from mag_fields import UniformMagneticField
    import matplotlib.pyplot as plt
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = "outputs" 
    plots_dir = os.path.join(outputs_dir, "plots")

    charge = -1.0
    mag_field_fn = None#UniformMagneticField(By = 9.2)
    neural_transport = NeuralCorrection().to(device)
    model = NeuralODE(charge, mag_field_fn, neural_transport).to(device)

    state0 = torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 0.0]], device=device)  # (batch=1, 6)
    step_sizes = torch.linspace(0.1, 1.0, steps=10, device=device)  # variable steps
    trajectory = model.trajectory(state0, step_sizes).detach().cpu().numpy()

    plt.plot(trajectory[0, :, 0], trajectory[0, :, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Particle Trajectory in Uniform Magnetic Field")
    plt.axis("equal")
    plt.show()
    plt.savefig(os.path.join(plots_dir, "trajectory.png"))

