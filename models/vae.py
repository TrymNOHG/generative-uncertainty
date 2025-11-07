import torch
import torch.nn as nn
from models.base import BaseModel

from stochman.manifold import EmbeddedManifold


class VAE_Encoder(nn.Module):
    """
        With layer norm.
    """
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            *[nn.Linear(in_channels[i], in_channels[i+1]) for i in range(len(in_channels) -1)]
        )
        # self.non_linearity = nn.LeakyReLU() 
        # self.non_linearity = nn.SiLU() 
        # self.non_linearity = nn.GELU() 
        self.non_linearity = nn.GELU() 
        self.bn = nn.Sequential(
            *[nn.LayerNorm(in_channels[i+1]) for i in range(len(in_channels) -1)]
        )
        self.fc_mean = nn.Linear(in_channels[-1], hidden_dim)          
        self.fc_log_var = nn.Linear(in_channels[-1], hidden_dim)       


    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        z = x
        for i, layer in enumerate(self.fc):
            z = layer(z)
            z = self.bn[i](z)
            z = self.non_linearity(z)
        z = self.non_linearity(z)
        mean = self.fc_mean(z)
        log_var = self.fc_log_var(z)

        return mean, log_var


class VAE_Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_channels[0]),
            *[nn.Linear(out_channels[i], out_channels[i+1]) for i in range(len(out_channels) - 1)]
        )
        # self.non_linearity = nn.LeakyReLU()
        self.non_linearity = nn.SiLU()
        # self.non_linearity = nn.GELU()

    def forward(self, z):
        for layer in self.fc:
            z = layer(z)
            z = self.non_linearity(z)
        x_hat = nn.Sigmoid()(z)

        return x_hat
    
class Uncertain_VAE_Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_channels[0]),
            *[nn.Linear(out_channels[i], out_channels[i+1]) for i in range(len(out_channels) - 2)]
        )
        self.mu_head = nn.Linear(in_features=out_channels[-2], out_features=out_channels[-1])
        self.logvar_head = nn.Linear(in_features=out_channels[-2], out_features=out_channels[-1])
        # self.non_linearity = nn.LeakyReLU()
        self.non_linearity = nn.SiLU()
        # self.non_linearity = nn.GELU()

    def forward(self, z):
        for layer in self.fc:
            z = layer(z)
            z = self.non_linearity(z)
        
        mu = self.mu_head(z)
        logvar = self.logvar_head(z)
        mu = nn.Sigmoid()(mu)

        return mu, logvar


class VariationalAutoencoder(BaseModel, EmbeddedManifold):
    def __init__(self, in_channels, out_channels, hidden_dim: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = VAE_Encoder(in_channels, hidden_dim)
        self.decoder = VAE_Decoder(out_channels, hidden_dim)

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
    
    def loss(self, x: torch.Tensor, y: torch.Tensor):
        x_hat, _, mean, log_var = self.forward(x)
        reconstruction_loss = torch.nn.BCELoss(reduction="sum")(x_hat.reshape_as(x), x)
        # reconstruction_loss = torch.nn.MSELoss()(x_hat, x)
        log_var = torch.clamp(log_var, min=-10, max=10) # I do this to make the logvar more numerically stable.
        kl_loss = - 0.5 * torch.mean(torch.sum(1 + (log_var) - (torch.exp(log_var)) - (mean**2), dim=1), dim=0) 
        kl_loss *= 1e-2
        return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss
    
    def embed(self, c, jacobian = False):
        return self.decoder(c)
    
    def loss_legend(self,):
        return ["Loss", "Reconstruction Loss", "KL Loss"]
    
    def short_name(self,):
        return "vae"
    


# TODO: alter this to be a one-layer hierarchical VAE (where I can retrieve mean and variance of the decoder and pass them through a softmax)
class GaussianVariationalAutoencoder(BaseModel):
    """
    In this VAE, the decoder will produce a mean and variance output.
    """
    def __init__(self, in_channels, out_channels, beta: float = 1e-2, hidden_dim: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = VAE_Encoder(in_channels, hidden_dim)
        self.decoder = Uncertain_VAE_Decoder(out_channels, hidden_dim)

        self.beta = beta

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        mu_dec, logvar_dec = self.decoder(z)

        return (mu_dec, logvar_dec), z, mean, log_var

    def gaussian_sample_in_euclid_space(self, num_samples: int = 1):
        z = torch.randn(num_samples, self.hidden_dim)
        return self.mu_decoder(z)
    
    def predict(self, x):
        return self.forward(x)[0]

    def reconstruct(self, z):
        return self.mu_decoder(z)[0] # 0 correct here?
    
    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(mean)
        var = torch.exp(0.5 * log_var) * eps
        return mean + var
    
    def loss(self, x: torch.Tensor, y: torch.Tensor):
        (x_hat, log_var_hat), _, mean, encoder_log_var = self.forward(x)

        x_hat = x_hat.reshape_as(x)
        log_var_hat = log_var_hat.reshape_as(x)
        residual = (x-x_hat).pow(2)
        
        log_var_hat = log_var_hat.clamp(min=-50, max=50)
        var = torch.exp(log_var_hat).clamp(min=1e-8, max=1e8)
        reconstruction_loss = 0.5 * torch.mean((log_var_hat + (residual) / var))

        encoder_log_var = torch.clamp(encoder_log_var, min=-10, max=10) # I do this to make the logvar more numerically stable.
        kl_loss = - 0.5 * torch.mean(torch.sum(1 + (encoder_log_var) - (torch.exp(encoder_log_var)) - (mean**2), dim=1), dim=0) 
        total_loss = reconstruction_loss + self.beta * kl_loss

        return total_loss, reconstruction_loss, kl_loss
    
    def loss_legend(self,):
        return ["Loss", "Gaussian Reconstruction Loss", "KL Loss"]
    
    def short_name(self,):
        return "gaussian-vae"
    