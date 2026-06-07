import torch
import torch.nn as nn

# input scales from the data: p0 ~ 15 GeV, log(step) ~ N(-4.69, 1.28)
P_SCALE, LOG_DS_MEAN, LOG_DS_STD = 15.0, -4.69, 1.28


def features(p, ds):
    ds = ds.expand(*p.shape[:-1], 1)
    log_ds = (ds.log() - LOG_DS_MEAN) / LOG_DS_STD
    return torch.cat([p / P_SCALE, log_ds], dim=-1)


def mlp(in_dim, out_dim, hidden=64, layers=3):
    net = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(layers - 1):
        net += [nn.Linear(hidden, hidden), nn.Tanh()]
    net.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*net)


class NeuralCorrection(nn.Module):
    """Predicts the transport rate dp/ds from [p, step]."""

    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.net = mlp(4, 3, hidden, layers)

    def forward(self, p, ds):
        return self.net(features(p, ds))


class NeuralODE(nn.Module):
    """RK4 integration of the learned transport over arc length."""

    def __init__(self, transport):
        super().__init__()
        self.transport = transport

    def _rk4_momentum(self, p, ds):
        k1 = self.transport(p, ds)
        k2 = self.transport(p + ds / 2 * k1, ds)
        k3 = self.transport(p + ds / 2 * k2, ds)
        k4 = self.transport(p + ds * k3, ds)
        return p + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def _rotate_z(p):
        theta = torch.rand(p.shape[0], device=p.device) * 2 * torch.pi
        c, s = theta.cos(), theta.sin()
        return torch.stack([c * p[:, 0] - s * p[:, 1],
                            s * p[:, 0] + c * p[:, 1],
                            p[:, 2]], dim=-1)

    @torch.no_grad()
    def momentum_trajectory(self, p0, step_sizes, stochastic=False):
        step_sizes = torch.as_tensor(step_sizes, device=p0.device, dtype=p0.dtype)
        out, p = [p0], p0
        for ds in step_sizes:
            p = self._rk4_momentum(p, ds)
            if stochastic:
                p = self._rotate_z(p)
            out.append(p)
        return torch.stack(out, dim=1)


class NeuralSDE(nn.Module):
    """Learns the drift and diffusion of  dp = mu ds + sigma dW.

    Energy loss is the drift (mean ~ ds); scattering is the diffusion, whose
    sqrt(ds) growth comes from the noise term so the network only needs a smooth
    sigma. Trained by the Gaussian NLL of one Euler-Maruyama step.
    """

    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.net = mlp(4, 6, hidden, layers)

    def drift_diffusion(self, p, ds):
        mu, log_sigma = self.net(features(p, ds)).split(3, dim=-1)
        return mu, log_sigma.clamp(-12, 8)

    def nll(self, p0, ds, p1):
        mu, log_sigma = self.drift_diffusion(p0, ds)
        var = (2 * log_sigma).exp() * ds
        resid = p1 - (p0 + mu * ds)
        return (0.5 * (resid ** 2 / var + var.log())).sum(-1)

    def step(self, p, ds):
        mu, log_sigma = self.drift_diffusion(p, ds)
        return p + mu * ds + log_sigma.exp() * ds.sqrt() * torch.randn_like(p)

    @torch.no_grad()
    def momentum_trajectory(self, p0, step_sizes):
        step_sizes = torch.as_tensor(step_sizes, device=p0.device, dtype=p0.dtype)
        out, p = [p0], p0
        for ds in step_sizes:
            p = self.step(p, ds)
            out.append(p)
        return torch.stack(out, dim=1)
