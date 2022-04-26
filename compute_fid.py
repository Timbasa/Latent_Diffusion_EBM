import os
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as tfm
from torch.autograd import Variable
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from pytorch_fid.inception import InceptionV3
from models.sampling import sample_langevin_prior, sample_p_0
from models.EBM import _G, _E
from utils import get_sigma_schedule


n_fid_samples = 40000
img_size = 32
batch_size = 100
nz, nc, ndf, ngf = 128, 3, 200, 128
K_0, a_0, K_1, a_1 = 60, 0.4, 40, 0.1
llhd_sigma = 0.3
n_iter = 70000
num_timesteps = 6
mcmc_step_size_b_square = 2e-4
dims = 2048
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
E.load_state_dict(torch.load(os.path.abspath('outputs/E_Latent_D.pth.tar')))
G.load_state_dict(torch.load(os.path.abspath('outputs/G_Latent_D.pth.tar')))


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def get_activations(imgs, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    model.eval()

    if batch_size > len(imgs):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(imgs)

    dataloader = torch.utils.data.DataLoader(imgs,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(imgs), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr



def compute_fid(device, x_data, x_samples):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    act = get_activations(x_data, model, batch_size, dims=2048, device=device, num_workers=1)
    mu1 = np.mean(act, axis=0)
    sigma1 = np.cov(act, rowvar=False)

    act = get_activations(x_samples, model, batch_size, dims=2048, device=device, num_workers=1)
    mu2 = np.mean(act, axis=0)
    sigma2 = np.cov(act, rowvar=False)

    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def get_fid_ours(n):
    to_range_0_1 = lambda x: (x + 1.) / 2.
    ds_fid = torch.stack([to_range_0_1(data[i].clone()) for i in range(n)]).cpu()
    assert n <= ds_fid.shape[0]

    print('computing fid with {} samples'.format(n))
    def sample_x():
        z_0 = sample_p_0(batch_size, nz, device)
        z_k = sample_langevin_prior(Variable(z_0), E, num_timesteps - 1, sigmas, sigmas_cum, a_s_prev, mcmc_step_size_b_square, K_0)
        x_samples = to_range_0_1(G(z_k)).clamp(min=0., max=1.).detach().cpu()
        return x_samples

    x_samples = torch.cat([sample_x() for _ in range(int(n /batch_size))])
    fid = compute_fid(device, ds_fid[:n], x_samples)

    return fid

print(get_fid_ours(n_fid_samples))