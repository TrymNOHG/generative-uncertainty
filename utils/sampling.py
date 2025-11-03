# Here the code for the different ways of sampling should be written.
import torch
from models.base import BaseModel

def gaussian_sampling(model: BaseModel, x, device, scale=1e2, num_samples=5):
    mean, log_var = model.encoder(x.to(device)) # Latent encoding
    latent_samples = [mean + torch.randn_like(mean) * scale * torch.exp(0.5 * log_var) for _ in range(num_samples)]
    outputs = [model.decoder(sample) for sample in latent_samples]
    return outputs

def metric_aware_sampling(model, x, device, num_samples=5):
    mean, log_var = model.encoder(x.to(device)) # Latent encoding
    latent_samples = [mean + torch.randn_like(mean) * torch.exp(0.5 * log_var) for _ in range(num_samples)]
    outputs = [model.decoder(sample) for sample in latent_samples]
    return outputs