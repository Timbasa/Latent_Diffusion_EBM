import torch
from utils import _extract
from models.diffusion import q_sample_full


# Get the true data
def sample_p_data(data, batch_size):
    return data[torch.LongTensor(batch_size).random_(0, data.size(0))].detach()


# Sample from the Gaussian distribution in shape (b,c,1,1)
def sample_p_0(batch_size, nz, device):
    return torch.randn(*[batch_size, nz, 1, 1]).to(device)


# log p(z|tilde_z) = -f(z) - 0.5/sigma^2 * ||tilde_z - z||^2
def log_prob(y, tilde_z, step_size_square, sigma, E):
    return -E(y)/step_size_square - torch.sum((tilde_z - y) ** 2 / 2 / sigma ** 2, dim=(1)).unsqueeze(1)


# Compute the gradient of log p(z|tilde_z)
def grad_f(y, tilde_z, step_size_square, sigma, E):
    y = y.clone().detach().requires_grad_(True)
    log_p_y = log_prob(y, tilde_z, step_size_square, sigma, E)
    grad_y = torch.autograd.grad(log_p_y.sum(), y)[0]
    return grad_y


# MCMC sampling for the latent ebm prior model.
def sample_langevin_prior(z, E, t, sigmas, sigmas_cum, a_s_prev, mcmc_step_size_b_square, K_0):
    sigma = _extract(sigmas, t + 1, z.shape)
    sigma_cum = _extract(sigmas_cum, t, z.shape)
    a_s = _extract(a_s_prev, t + 1, z.shape)
    c_t_square = sigma_cum / sigmas_cum[0]
    step_size_square = c_t_square * sigma ** 2 * mcmc_step_size_b_square
    y = z.clone().detach().requires_grad_(True)

    for i in range(K_0):
        y_grad = grad_f(y, z, step_size_square, sigma, E)
        y.data = y.data + 0.5 * step_size_square * y_grad \
                 + torch.sqrt(step_size_square) * torch.randn_like(z).data
    return y.detach() / a_s


# MCMC sampling for Generator model p(z|x) = p(z) * p(x|z) / p(x)
# grad(log p(x|z)) = sum_t(grad_z_t(log p(z_{t}|z_{t-1}))) - gaussia  //EBM part
#                       + grad_z(G(z))
def sample_langevin_posterior(z, x, G, E, t, K_1, llhd_sigma, mse, num_timesteps, sigmas, a_1, a_s_cum, sigmas_cum):
    y = z.clone().detach().requires_grad_(True)
    for i in range(K_1):
        x_hat = G(y)
        g_log_lkhd = - 1.0 / (2.0 * llhd_sigma * llhd_sigma) * mse(x_hat, x)
        grad_g = torch.autograd.grad(g_log_lkhd, y)[0]

        # EBM likelihood = sum_t(grad(E(z_t), z_t))
        z_ts = q_sample_full(y, num_timesteps, a_s_cum, sigmas_cum)
        grad_e = - torch.randn_like(y)
        for t, (z_t, z_t_plus_one) in enumerate(zip(z_ts,z_ts[1:])):
            sigma = _extract(sigmas, t + 1, z.shape)
            step_size_square = 1
            grad_e += grad_f(z_t, z_t_plus_one, step_size_square, sigma, E)

        y.data = y.data + 0.5 * a_1 * a_1 * (grad_e + grad_g) + a_1 * torch.randn_like(y).data
    return y.detach()