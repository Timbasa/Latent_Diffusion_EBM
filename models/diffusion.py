import torch
from utils import _extract


# Get z_t given z_0 and t
def q_sample(z_start, t, a_s_cum, sigmas_cum, noise=None):
    if noise == None:
        noise = torch.randn_like(z_start)
    z_t = _extract(a_s_cum, t, z_start.shape) * z_start + \
          _extract(sigmas_cum, t, z_start.shape) * noise
    return z_t


# Get z_t and z_{t+1}
def q_sample_pairs(z_start, t, a_s, sigmas, a_s_cum, sigmas_cum):
    noise = torch.randn_like(z_start)
    z_t = q_sample(z_start, t, a_s_cum, sigmas_cum)
    z_t_plus_one = _extract(a_s, t + 1, z_start.shape) * z_t + \
                   _extract(sigmas, t + 1, z_start.shape) * noise
    return z_t, z_t_plus_one


# Get the sequence of {z_i} for i = {0,1,...t}
def q_sample_full(z_start, num_timesteps, a_s_cum, sigmas_cum):
    z_pred = []
    for t in range(num_timesteps + 1):
        t_now = torch.ones(z_start.shape[0], dtype=torch.long) * t
        z = q_sample(z_start, t_now, a_s_cum, sigmas_cum)
        z_pred.append(z)
    z_preds = torch.stack(z_pred, dim=0)
    return z_preds