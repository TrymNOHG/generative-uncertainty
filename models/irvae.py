
# This code was heavily inspired by https://github.com/Gabe-YHLee/IRVAE-public
import torch
import torch.nn as nn
from models.vae import VAE_Encoder, VAE_Decoder
import numpy as np

from models.base import BaseModel



# In the code, the developers utilize an isotropic gaussian where the variance is learnable.
class IsotropicGaussian(nn.Module):
    """Isotripic Gaussian density function paramerized by a neural net.
    standard deviation is a free scalar parameter"""

    def __init__(self, net, sigma=1):
        super().__init__()
        self.net = net
        sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
        self.register_parameter("sigma", sigma)

    def log_likelihood(self, x: torch.Tensor, z: torch.Tensor):
        decoder_out = self.net(z).reshape_as(x)
        D = torch.prod(torch.tensor(x.shape[1:]))
        sig = self.sigma
        const = -D * 0.5 * torch.log(
            2 * torch.tensor(np.pi, dtype=torch.float32)
        ) - D * torch.log(sig)
        loglik = const - 0.5 * ((x - decoder_out) ** 2).view((x.shape[0], -1)).sum(
            dim=1
        ) / (sig ** 2)
        return loglik

    def forward(self, z):
        return self.net(z)

    def sample(self, z):
        x_hat = self.net(z)
        return x_hat + torch.randn_like(x_hat) * self.sigma

class IRVAE(BaseModel):
    def __init__(self, in_channels, out_channels, iso_reg=1.0, metric='identity', hidden_dim: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.iso_reg = iso_reg
        self.metric = metric

        self.encoder = VAE_Encoder(in_channels, hidden_dim)
        self.decoder = IsotropicGaussian(VAE_Decoder(out_channels, hidden_dim))

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)

        return x_hat, z, mean, log_var
    
    def gaussian_sample_in_euclid_space(self, num_samples: int = 1):
        z = torch.randn(num_samples, self.hidden_dim)
        return self.decoder(z)
    
    def predict(self, x):
        return self.forward(x)[0]

    def reconstruct(self, z):
        return self.decoder(z)[0] # 0 correct here?
    
    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(mean)
        var = torch.exp(0.5 * log_var) * eps
        return mean + var
    
    # In the paper, they use negative likelihood loss instead of MSE for the reconstruction loss.

    # The loss function introduced in this paper is essentially composed of three parts:
    # 1. The reconstruction loss, where NLL is used.
    # 2. The KL-divergence
    # 3. The scaled isometric regularization term
    
    def loss(self, x: torch.Tensor, y: torch.Tensor):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        nll = - self.decoder.log_likelihood(x, z)
        kl_loss = - 0.5 * torch.mean(torch.sum(1 + (log_var) - (torch.exp(log_var)) - (mean**2), dim=1), dim=0) 
        iso_loss = relaxed_distortion_measure(self.decoder, z, eta=0.2, metric=self.metric)
            
        loss = (nll + kl_loss).mean() + self.iso_reg * iso_loss
        return loss, nll.mean(), kl_loss, iso_loss
    
    def loss_legend(self,):
        return ["Loss", "Reconstruction Loss", "KL Loss", "Isometric Regularization"]
    
    def short_name(self,):
        return "irvae"

    


# def eval_step(self, dl, **kwargs):
#     device = kwargs["device"]
#     score = []
#     for x, _ in dl:
#         z = self.encode(x.to(device))
#         G = get_pullbacked_Riemannian_metric(self.decode, z)
#         score.append(get_flattening_scores(G, mode='condition_number'))
#     mean_condition_number = torch.cat(score).mean()
#     return {
#         "MCN_": mean_condition_number.item()
#     }

def relaxed_distortion_measure(func, z, eta=0.2, metric='identity', create_graph=True):
    if metric == 'identity':
        bs = len(z)
        z_perm = z[torch.randperm(bs)]
        if eta is not None:
            alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha*z + (1-alpha)*z_perm
        else:
            z_augmented = z
        v = torch.randn(z.size()).to(z)
        Jv = torch.autograd.functional.jvp(func, z_augmented, v=v, create_graph=create_graph)[1]
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
        JTJv = (torch.autograd.functional.vjp(func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
        TrG2 = torch.sum(JTJv**2, dim=1).mean()
        return TrG2/TrG**2
    else:
        raise NotImplementedError

def get_flattening_scores(G, mode='condition_number'):
    if mode == 'condition_number':
        S = torch.svd(G).S
        scores = S.max(1).values/S.min(1).values
    elif mode == 'variance':
        G_mean = torch.mean(G, dim=0, keepdim=True)
        A = torch.inverse(G_mean)@G
        scores = torch.sum(torch.log(torch.svd(A).S)**2, dim=1)
    else:
        pass
    return scores

def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    batch_size, z_dim = inputs.size()
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    jac = (
        torch.autograd.functional.jvp(
            func, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return jac

def get_pullbacked_Riemannian_metric(func, z):
    J = jacobian_decoder_jvp_parallel(func, z, v=None)
    G = torch.einsum('nij,nik->njk', J, J)
    return G