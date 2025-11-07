# Here the code for the different ways of sampling should be written.
import torch
import random
from models.base import BaseModel
from models.rbf import RBFNet
from stochman.manifold import EmbeddedManifold
from utils.geodesic import find_geodesic, points_on_geodesic


def gaussian_sampling(model: BaseModel, x, device, scale=1e2, num_samples=5):
    mean, log_var = model.encoder(x.to(device)) # Latent encoding
    latent_samples = [mean + torch.randn_like(mean) * scale * torch.exp(0.5 * log_var) for _ in range(num_samples)]
    outputs = [model.decoder(sample) for sample in latent_samples]
    return outputs

def gaussian_sampling_uvae(model: BaseModel, x, device, scale=1e2, num_samples=5):
    mean, log_var = model.encoder(x.to(device)) # Latent encoding
    gauss_sample = lambda mean, log_var: mean + torch.randn_like(mean) * scale * torch.exp(0.5 * log_var) 
    latent_samples = [gauss_sample(mean, log_var) for _ in range(num_samples)]
    output_dists = [model.decoder(sample) for sample in latent_samples]
    outputs = [gauss_sample(mean, log_var) for mean, log_var in output_dists]
    return outputs

def gaussian_sampling_rbf(model: BaseModel, rbf: RBFNet, x, device, scale=1e2, num_samples=5):
    mean, log_var = model.encoder(x.to(device)) # Latent encoding
    gauss_sample = lambda mean, log_var: mean + torch.randn_like(mean) * scale * torch.exp(0.5 * log_var) 
    latent_samples = [gauss_sample(mean, log_var) for _ in range(num_samples)]
    output_variance = [rbf(sample).unsqueeze(0) for sample in latent_samples]
    print(output_variance)
    outputs = [model.decoder(sample).unsqueeze(0) for sample in latent_samples]
    output_samples = [gauss_sample(mean, var) for mean, var in zip(outputs, output_variance)]
    return output_samples


# Metric aware sampling

def brute_force_metric_sampling(model, x_test, device, scale=1e2, num_samples=5):
    """
    In this sampling method, a significant amount of latent points will be sampled. The distance between these points and the original test input will be found using 
    geodesics/Riemannian metric. From there, only the 5 closest will be used.
    """
    mean, log_var = model.encoder(x_test.to(device)) # Latent encoding
    gauss_sample = lambda mean, log_var: mean + torch.randn_like(mean) * scale * torch.exp(0.5 * log_var) 

    latent_samples = [gauss_sample(mean, log_var) for _ in range(10_000)]
    # Calculate distance between these and the original on the manifold using metric

def geodesic_sampling(model: EmbeddedManifold, z_train_dict: dict, x_test, prediction: int, device, num_samples=5):
    """
        In this sampling method, a geodesic is drawn between the given test point and a randomly chosen train datapoint with the same label as the predicted label for 
        the test input. This idea was inspired by Data Generation in Low Sample Size Setting Using Manifold Sampling and a Geometry-Aware VAE - Chadebec et. al 2021.
    """
    model.eval()
    with torch.no_grad():
        _, z1, _, _ = model(x_test)

        z_class = z_train_dict[prediction]
        
        samples = []
        for i in range(num_samples):
            try:
                z2 = z_class[i]
                curve = find_geodesic(model, z1, z2)
                print(f"Curve {i} has geodesic length: {curve.euclidean_length().item()}")
                points, _ = points_on_geodesic(curve, 10, 0, 0.1)
                latent_point = points.squeeze(0)[random.randint(0, 4)]
                sample = model.decoder(latent_point)
                samples.append(sample)
            except:
                break
        
    return samples

def geodesic_interpolation_x(model: EmbeddedManifold, z_train_dict: dict, x_test, class_val: int, device, num_steps=5):
    """
        In this sampling method, a geodesic is drawn between the given test point and a randomly chosen train datapoint with the same label as the predicted label for 
        the test input. This idea was inspired by Data Generation in Low Sample Size Setting Using Manifold Sampling and a Geometry-Aware VAE - Chadebec et. al 2021.
    """
    model.eval()
    with torch.no_grad():
        _, z1, _, _ = model(x_test)

        z_class = z_train_dict[class_val]
        z2 = z_class[random.randint(0, len(z_class) - 1)]
        
        samples = []
        curve = find_geodesic(model, z1, z2)
        print(f"Curve has geodesic length: {curve.euclidean_length().item()}")
        points, _ = points_on_geodesic(curve, num_steps, 0, 1.0)
        points = points.squeeze(0)
        for i in range(num_steps):
            try:
                latent_point = points[i]
                sample = model.decoder(latent_point)
                samples.append(sample)
            except:
                break
    
    return samples
    

def geodesic_interpolation_z(model: EmbeddedManifold, z1: torch.Tensor, z2: torch.Tensor, num_steps=5):
    """
        In this sampling method, a geodesic is drawn between the given test point and a randomly chosen train datapoint with the same label as the predicted label for 
        the test input. This idea was inspired by Data Generation in Low Sample Size Setting Using Manifold Sampling and a Geometry-Aware VAE - Chadebec et. al 2021.
    """
    model.eval()
    with torch.no_grad():
        samples = []
        curve = find_geodesic(model, z1, z2)
        print(f"Curve has geodesic length: {curve.euclidean_length().item()}")
        points, _ = points_on_geodesic(curve, num_steps, 0, 1.0)
        points = points.squeeze(0)
        for i in range(num_steps):
            try:
                latent_point = points[i]
                sample = model.decoder(latent_point)
                samples.append(sample)
            except:
                break
        
    return samples
    
    # TODO: now sample along these curves.

def wrapped_normal_dist_sampling(model, x_test, device, num_samples=5):
    """
    This sampling method is also taken from Data Generation in Low Sample Size Setting Using Manifold Sampling and a Geometry-Aware VAE - Chadebec et. al 2021.
    The sampling method works by constructing a distribution on the manifold utilizing the maximum entropy distribution and the geodesic distance. 
    Essentially, a eucldiean normal distribution is placed in the tangent space and moved onto the manifold.
    """

def metric_aware_sampling(model, x, device, num_samples=5):
    mean, log_var = model.encoder(x.to(device)) # Latent encoding
    latent_samples = [mean + torch.randn_like(mean) * torch.exp(0.5 * log_var) for _ in range(num_samples)]
    outputs = [model.decoder(sample) for sample in latent_samples]
    return outputs