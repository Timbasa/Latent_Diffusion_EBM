import os
import pickle

import PIL
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfm
from torchvision import datasets
from VAE import VAE
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

img_size = 64
batch_size = 100
nz, nc, ndf, ngf = 128, 3, 200, 128
K_0, a_0, K_1, a_1 = 60, 0.4, 40, 0.1
llhd_sigma = 0.3
n_iter = 1000
num_timesteps = 6
mcmc_step_size_b_square = 2e-4

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')


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


class SingleImagesFolderMTDataset(torch.utils.data.Dataset):
    def __init__(self, root, cache, num_images, transform=None, workers=32, protocol=None):
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            path_imgs = os.listdir(root)[:num_images]
            n_splits = len(path_imgs) // 1000
            # n_splits = num_images
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.decompress(self.images[item])

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output


sigmas, a_s = get_sigma_schedule(beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=num_timesteps)
a_s_cum = np.cumprod(a_s)
sigmas_cum = np.sqrt(1 - a_s_cum ** 2)

sigmas = torch.Tensor(sigmas).to(device)
a_s = torch.Tensor(a_s).to(device)
sigmas_cum = torch.Tensor(sigmas_cum).to(device)
a_s_cum = torch.Tensor(a_s_cum).to(device)

a_s_prev = a_s.clone()
a_s_prev[-1] = 1


def _extract(a, t, x_shape):
    if isinstance(t, int) or len(t.shape) == 0:
        t = torch.ones(x_shape[0], dtype=torch.long) * t
    bs, = t.shape
    out = a[t]
    return out.reshape([bs] + ((len(x_shape) - 1) * [1]))


def q_sample(z_start, t, noise=None):
    if noise == None:
        noise = torch.randn_like(z_start)
    z_t = _extract(a_s_cum, t, z_start.shape) * z_start + \
          _extract(sigmas_cum, t, z_start.shape) * noise
    return z_t


def q_sample_pairs(z_start, t):
    noise = torch.randn_like(z_start)
    z_t = q_sample(z_start, t)
    z_t_plus_one = _extract(a_s, t + 1, z_start.shape) * z_t + \
                   _extract(sigmas, t + 1, z_start.shape) * noise
    return z_t, z_t_plus_one


def q_sample_full(z_start):
    z_pred = []
    for t in range(num_timesteps + 1):
        t_now = torch.ones(z_start.shape[0], dtype=torch.long) * t
        z = q_sample(z_start, t_now)
        z_pred.append(z)
    z_preds = torch.stack(z_pred, dim=0)
    return z_preds

# For CelebA
class _G(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.gen(z)


class _E(nn.Module):
    def __init__(self):
        super().__init__()
        self.ebm = nn.Sequential(nn.Linear(nz, ndf), nn.LeakyReLU(0.2),
                                 nn.Linear(ndf, ndf), nn.LeakyReLU(0.2),
                                 nn.Linear(ndf, 1))

    def forward(self, z):
        return self.ebm(z.squeeze()).view(-1, 1, 1, 1)


transform = tfm.Compose([PIL.Image.fromarray,
                         tfm.Resize(img_size),
                         tfm.CenterCrop(img_size),
                         tfm.ToTensor(),
                         tfm.Normalize(([0.5] * 3), ([0.5] * 3)), ])

ds_train = SingleImagesFolderMTDataset(root='data/celeba/img_align_celeba', cache=None, num_images=40000, transform=transform)

dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)

G, E = _G().to(device), _E().to(device)
mse = nn.MSELoss(reduction='sum')

optE = torch.optim.Adam(E.parameters(), lr=2e-5, betas=(0.5, 0.999))
optG = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))


def sample_p_0(n=batch_size):
    return torch.randn(*[n, nz, 1, 1]).to(device)


def log_prob(y, tilde_z, step_size_square, sigma, E):
    return -E(y)/step_size_square - torch.sum((tilde_z - y) ** 2 / 2 / sigma ** 2, dim=(1)).unsqueeze(1)


def grad_f(y, tilde_z, step_size_square, sigma, E):
    y = y.clone().detach().requires_grad_(True)
    log_p_y = log_prob(y, tilde_z, step_size_square, sigma, E)
    grad_y = torch.autograd.grad(log_p_y.sum(), y)[0]
    return grad_y


def sample_langevin_prior(z, E, t):
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


def sample_langevin_posterior(z, x, G, E, t):
    y = z.clone().detach().requires_grad_(True)
    for i in range(K_1):
        x_hat = G(y)
        g_log_lkhd = - 1.0 / (2.0 * llhd_sigma * llhd_sigma) * mse(x_hat, x)
        grad_g = torch.autograd.grad(g_log_lkhd, y)[0]

        # EBM likelihood = sum_t(grad(E(z_t), z_t))
        z_ts = q_sample_full(y)
        grad_e = - torch.randn_like(y)
        for t, (z_t, z_t_plus_one) in enumerate(zip(z_ts,z_ts[1:])):
            sigma = _extract(sigmas, t + 1, z.shape)
            step_size_square = 1
            grad_e += grad_f(z_t, z_t_plus_one, step_size_square, sigma, E)

        y.data = y.data + 0.5 * a_1 * a_1 * (grad_e + grad_g) + a_1 * torch.randn_like(y).data
    return y.detach()


def reshape(img):
    # transpose numpy array to the PIL format, i.e., Channels x W x H
    out = np.transpose(img, (1, 2, 0))
    return ((out + 1.0) * 255.0 / 2.0).astype(np.uint8)


def train():
    for i in range(n_iter):
        for x in dataloader_train:
            x = x.to(device)
            z_g_0 = sample_p_0()
            E.train()
            G.train()

            B = z_g_0.shape[0]
            t = torch.randint(high=num_timesteps, size=(B,)).to(device)
            z_g_k = sample_langevin_posterior(z_g_0, x, G, E, t)

            # Add the timesteps to diffuse z_e_0 to z_e_t.
            z_pos, z_neg = q_sample_pairs(z_g_k, t)
            z_neg = sample_langevin_prior(z_neg, E, t)

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

        # if i % 100 == 0:
        print('Epoch {}: generator = {}, en_pos = {}, en_neg = {}, loss = {}.'
              .format(i, loss_g.data, en_pos.data, en_neg.data, loss_e.data))

        if i % 50 == 49:
            z_0 = sample_p_0(n=16)
            z_k = sample_langevin_prior(Variable(z_0), E, num_timesteps-1)
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
            fig.savefig("outputs/4-26-2022/Latent_Diffusion_EBM_epoch={}_celeba.pdf".format(i), bbox_inches='tight')
            plt.close(fig)

            torch.save(G.state_dict(), os.path.abspath('outputs/4-26-2022/G_Latent_D.pth_epoch={}_celeba.tar'.format(i)))
            torch.save(E.state_dict(), os.path.abspath('outputs/4-26-2022/E_Latent_D.pth_epoch={}_celeba.tar'.format(i)))


def test():
    E.load_state_dict(torch.load(os.path.abspath('outputs/4-26-2022/E_Latent_D.pth_epoch=69999.tar')))
    G.load_state_dict(torch.load(os.path.abspath('outputs/4-26-2022/G_Latent_D.pth_epoch=69999.tar')))
    E.eval()
    G.eval()
    z_0 = sample_p_0(n=8)
    z_k = sample_langevin_prior(Variable(z_0), E, num_timesteps - 1)
    imgs =  G(z_k)
    for _ in range(7):
        z_0 = sample_p_0(n=8)
        z_k = sample_langevin_prior(Variable(z_0), E, num_timesteps - 1)
        imgs = torch.cat((imgs, G(z_k)))

    grid = make_grid(imgs, nrow=8, padding=8)
    npimg = grid.detach().cpu().numpy()  # to numpy array
    fig, ax = plt.subplots(figsize=tuple((100, 4)))
    ax.axis("off")
    npimg = reshape(npimg)
    ax.imshow(npimg)
    #
    fig.savefig("outputs/4-26-2022/Latent_Diffusion_EBM_Generation_celeba.pdf", bbox_inches='tight', dpi=1200)
    plt.close(fig)

train()
test()