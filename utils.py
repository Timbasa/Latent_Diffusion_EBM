import torch
import numpy as np


def get_sigma_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end, 1000, dtype=np.float64)
    betas = np.append(betas, 1.)
    assert isinstance(betas, np.ndarray)
    betas = betas.astype(np.float64)
    assert (betas > 0).all() and (betas <= 1).all()
    sqrt_alphas = np.sqrt(1. - betas)
    idx = torch.IntTensor(
        np.concatenate([np.arange(num_diffusion_timesteps) * (1000 // ((num_diffusion_timesteps - 1) * 2)), [999]]))
    a_s = np.concatenate(
        [[np.prod(sqrt_alphas[: idx[0] + 1])],
         np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])
    sigmas = np.sqrt(1 - a_s ** 2)
    return sigmas, a_s


def _extract(a, t, x_shape):
    if isinstance(t, int) or len(t.shape) == 0:
        t = torch.ones(x_shape[0], dtype=torch.long) * t
    bs, = t.shape
    out = a[t]
    return out.reshape([bs] + ((len(x_shape) - 1) * [1]))


def reshape(img):
    # transpose numpy array to the PIL format, i.e., Channels x W x H
    out = np.transpose(img, (1, 2, 0))
    return ((out + 1.0) * 255.0 / 2.0).astype(np.uint8)
