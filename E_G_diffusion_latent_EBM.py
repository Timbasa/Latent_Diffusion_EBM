import os
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfm
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from models.EBM import _E, _G
from models.diffusion import q_sample_pairs
from models.sampling import sample_p_0, sample_p_data, sample_langevin_prior, sample_langevin_posterior
from utils import get_sigma_schedule, _extract, reshape

img_size = 32
batch_size = 100
nz, nc, ndf, ngf = 128, 3, 200, 128
K_0, a_0, K_1, a_1 = 60, 0.4, 40, 0.1
llhd_sigma = 0.3
n_iter = 70000
num_timesteps = 6
mcmc_step_size_b_square = 2e-4
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

(sigmas, a_s) = get_sigma_schedule(beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=num_timesteps)
a_s_cum = np.cumprod(a_s)
sigmas_cum = np.sqrt(1 - a_s_cum ** 2)
sigmas = torch.Tensor(sigmas).to(device)
a_s = torch.Tensor(a_s).to(device)
sigmas_cum = torch.Tensor(sigmas_cum).to(device)
a_s_cum = torch.Tensor(a_s_cum).to(device)
a_s_prev = a_s.clone()
a_s_prev[-1] = 1

transform = tfm.Compose([tfm.Resize(img_size), tfm.ToTensor(), tfm.Normalize(([0.5] * 3), ([0.5] * 3)), ])
data = torch.stack([x[0] for x in tv.datasets.CIFAR10(root='data', train=True,
                                                  download=True, transform=transform)]).to(device)
G, E = _G(nz=nz, nc=nc, ngf=ngf).to(device), _E(nz=nz, ndf=ndf).to(device)
mse = nn.MSELoss(reduction='sum')
optE = torch.optim.Adam(E.parameters(), lr=2e-5, betas=(0.5, 0.999))
optG = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))


def train():
    for i in range(n_iter):
        x = sample_p_data(data, batch_size)
        z_g_0 = sample_p_0(batch_size, nz, device)
        E.train()

        B = z_g_0.shape[0]
        t = torch.randint(high=num_timesteps, size=(B,)).to(device)
        z_g_k = sample_langevin_posterior(z_g_0, x, G, E, t, K_1, llhd_sigma, mse, num_timesteps, sigmas, a_1)

        # Add the timesteps to diffuse z_e_0 to z_e_t.
        z_pos, z_neg = q_sample_pairs(z_g_k, t, a_s, sigmas, a_s_cum, sigmas_cum)
        z_neg = sample_langevin_prior(z_neg, E, t, sigmas, sigmas_cum, a_s_prev, mcmc_step_size_b_square, K_0)

        optG.zero_grad()
        x_hat = G(z_g_k.detach())
        loss_g = mse(x_hat, x) / batch_size
        loss_g.backward()
        optG.step()

        optE.zero_grad()
        a_s = _extract(a_s_prev, t + 1, z_pos.shape)
        y_pos = a_s * z_pos
        y_neg = a_s * z_neg
        en_pos, en_neg = E(y_pos.detach()).mean(), E(y_neg.detach()).mean()
        loss_e = en_pos - en_neg
        loss_e.backward()
        optE.step()

        if i % 100 == 0:
            print('Epoch {}: generator = {}, en_pos = {}, en_neg = {}, loss = {}.'
                  .format(i, loss_g.data, en_pos.data, en_neg.data, loss_e.data))

        if i % 2000 == 1999:
            z_0 = sample_p_0(batch_size=16, nz=nz, device=device)
            z_k = sample_langevin_prior(Variable(z_0), E, num_timesteps-1, sigmas, sigmas_cum, a_s_prev, mcmc_step_size_b_square, K_0)
            #
            imgs = G(z_k)
            #
            grid = make_grid(imgs, nrow=imgs.size(0), padding=8)
            npimg = grid.detach().cpu().numpy()  # to numpy array
            #
            fig, ax = plt.subplots(figsize=tuple((100, 4)))
            ax.axis("off")
            npimg = reshape(npimg)
            ax.imshow(npimg)
            #
            fig.savefig("outputs/Latent_Diffusion_EBM_epoch={}.pdf".format(i), bbox_inches='tight')
            plt.close(fig)

            torch.save(G.state_dict(), os.path.abspath('outputs/G_Latent_D_epoch={}.pth.tar'.format(i)))
            torch.save(E.state_dict(), os.path.abspath('outputs/E_Latent_D_epoch={}.pth.tar'.format(i)))


def test():
    E.load_state_dict(torch.load(os.path.abspath('outputs/E_Latent_D_epoch=69999.pth.tar')))
    G.load_state_dict(torch.load(os.path.abspath('outputs/G_Latent_D_epoch=69999.pth.tar')))
    z_0 = sample_p_0(8, nz, device)
    z_k = sample_langevin_prior(Variable(z_0), E, num_timesteps - 1, sigmas, sigmas_cum, a_s_prev, mcmc_step_size_b_square, K_0)
    imgs =  G(z_k)
    for _ in range(7):
        z_0 = sample_p_0(8, nz, device)
        z_k = sample_langevin_prior(Variable(z_0), E, num_timesteps - 1, sigmas, sigmas_cum, a_s_prev, mcmc_step_size_b_square, K_0)
        imgs = torch.cat((imgs, G(z_k)))

    grid = make_grid(imgs, nrow=8, padding=8)
    npimg = grid.detach().cpu().numpy()  # to numpy array
    fig, ax = plt.subplots(figsize=tuple((100, 4)))
    ax.axis("off")
    npimg = reshape(npimg)
    ax.imshow(npimg)
    fig.savefig("outputs/Latent_Diffusion_EBM_Generation.pdf", bbox_inches='tight', dpi=1200)
    plt.close(fig)

train()
test()