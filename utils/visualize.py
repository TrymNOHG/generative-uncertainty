# Visualization functions here
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.riemann import log_riemann_volume, log_random_riemann_volume

def visualize_latent_grid(model: torch.nn.Module, device: torch.device, n_steps:int = 32):
    """
        This code was taken and slightly modified from the auto-encoder exercise by Oskar Jørgensen.
    """
    # Create a grid of points between 0.01 and 0.99
    x = torch.linspace(0.01, 0.99, n_steps)
    y = torch.linspace(0.01, 0.99, n_steps)

    # Create a grid of points
    grid_points = torch.zeros(n_steps * n_steps, 2)

    # Create meshgrid of points
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Convert to standard normal values using inverse CDF (probit function)
    from scipy.stats import norm
    X = torch.tensor(norm.ppf(X.numpy().astype(np.float32)))
    Y = torch.tensor(norm.ppf(Y.numpy().astype(np.float32)))

    # Combine into grid points
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).float().to(device)

    # Generate images from the latent points
    with torch.no_grad():
        model.eval()
        generated = model.decoder(grid_points)
        generated = generated.reshape(-1, 1, 28, 28)

    _, axes = plt.subplots(n_steps, n_steps, figsize=(8, 8))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(n_steps):
        for j in range(n_steps):
            axes[i,j].imshow(generated[i * n_steps + j].cpu().squeeze(), cmap='gray')
            axes[i,j].axis('off')

    plt.show()


def visualize_latent_uncertainty_grid(model: torch.nn.Module, device: torch.device, n_steps: int = 32):
    """
    This visualization code was partially taken from ChatGPT.
    """
    model.eval()

    x = torch.linspace(0.01, 0.99, n_steps)
    y = torch.linspace(0.01, 0.99, n_steps)

    X, Y = torch.meshgrid(x, y, indexing="ij")  

    from scipy.stats import norm
    X = torch.tensor(norm.ppf(X.numpy().astype(np.float32)), dtype=torch.float32)
    Y = torch.tensor(norm.ppf(Y.numpy().astype(np.float32)), dtype=torch.float32)

    grid_points = torch.stack([X.flatten(), Y.flatten()],dim=1).to(device)  

    with torch.no_grad():
        if hasattr(model, "decoder"):
            log_vol = log_riemann_volume(model.decoder, grid_points)
        else:
            log_vol = log_riemann_volume(model, grid_points)


    plt.figure(figsize=(6, 5))

    z1_min, z1_max = X.min().item(), X.max().item()
    z2_min, z2_max = Y.min().item(), Y.max().item()

    values = log_vol.cpu().reshape(n_steps, n_steps)
    label = "log Riemann volume element"

    im = plt.imshow(
        values,
        origin="lower",
        extent=[z1_min, z1_max, z2_min, z2_max],
        aspect="equal",
        cmap="viridis",
    )
    plt.xlabel("z1")
    plt.ylabel("z2")
    cbar = plt.colorbar(im)
    cbar.set_label(label)
    plt.title("Latent Riemannian volume over 2D grid")
    plt.tight_layout()
    plt.show()

def visualize_latent_uncertainty_grid_random(mu_decoder: torch.nn.Module, var_decoder: torch.nn.Module, device: torch.device, n_steps: int = 32):
    """
    This visualization code was partially taken from ChatGPT.
    """
    mu_decoder.eval()
    var_decoder.eval()

    x = torch.linspace(0.01, 0.99, n_steps)
    y = torch.linspace(0.01, 0.99, n_steps)

    X, Y = torch.meshgrid(x, y, indexing="ij")  

    from scipy.stats import norm
    X = torch.tensor(norm.ppf(X.numpy().astype(np.float32)), dtype=torch.float32)
    Y = torch.tensor(norm.ppf(Y.numpy().astype(np.float32)), dtype=torch.float32)

    grid_points = torch.stack([X.flatten(), Y.flatten()],dim=1).to(device)  

    with torch.no_grad():
        log_vol = log_random_riemann_volume(mu_decoder, var_decoder, grid_points)  

    plt.figure(figsize=(6, 5))

    z1_min, z1_max = X.min().item(), X.max().item()
    z2_min, z2_max = Y.min().item(), Y.max().item()

    values = log_vol.cpu().reshape(n_steps, n_steps)
    label = "log Riemann volume element"

    im = plt.imshow(
        values,
        origin="lower",
        extent=[z1_min, z1_max, z2_min, z2_max],
        aspect="equal",
        cmap="viridis",
    )
    plt.xlabel("z1")
    plt.ylabel("z2")
    cbar = plt.colorbar(im)
    cbar.set_label(label)
    plt.title("Latent Riemannian volume over 2D grid")
    plt.tight_layout()
    plt.show()



def visualize_samples(original, samples):
    fig, axes = plt.subplots(1, 1 + len(samples), figsize=((len(samples) + 1) * 2, 2))
    original_np = original.squeeze().detach().cpu().numpy()
    axes[0].imshow(original_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original input")
    axes[0].axis("off")

    for i, sample in enumerate(samples):
        sample = sample.reshape_as(original)
        sample_np = sample.squeeze().detach().cpu().numpy()
        axes[i+1].imshow(sample_np, cmap="gray", vmin=0, vmax=1)
        axes[i+1].set_title(f"Sample input {i+1}")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.show()  

def visualize_samples_only(samples):
    fig, axes = plt.subplots(1, len(samples), figsize=((len(samples)) * 2, 2))
    for i, sample in enumerate(samples):
        sample = sample.reshape(torch.Size([1, 1, 28, 28]))
        sample_np = sample.squeeze().detach().cpu().numpy()
        axes[i].imshow(sample_np, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"Sample input {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()  


def visualize_points_in_latent_space(model: torch.nn.Module, trainloader: torch.utils.data.DataLoader):
    """
        This code was taken and slightly modified from the auto-encoder exercise by Oskar Jørgensen.
    """
    labels = []
    latent_samples = []
    with torch.no_grad():
        model.eval()
        for x_batch, y_batch in trainloader:
            x_hat, z, mean, log_var = model(x_batch)
            latent_samples.extend(z)
            labels.extend(y_batch)
    latent_samples = [sample.detach().numpy() for sample in latent_samples]
    x_coords = []
    y_coords = []
    for (x, y) in latent_samples:
        x_coords.append(x)   
        y_coords.append(y)   

    plt.scatter(x_coords, y_coords, c=labels, cmap="tab10")
    plt.colorbar()  
    plt.plot()


def plot_overlapped(points1, points2, plot_in, t=None,
                    color1="C0", color2="C1",
                    *plot_args, **plot_kwargs):
    """
    This visualization code was generated by ChatGPT
    points1, points2: tensors/arrays of shape (B, T, 2) or (B, T, 1) depending on your case
    plot_in: typically matplotlib.pyplot
    t: needed if you're in the 1D case (like your first branch)
    """
    figs = []

    if points1.shape[0] != points2.shape[0]:
        raise ValueError("Both point arrays must have the same batch size")

    B = points1.shape[0]

    for b in range(B):
        # create fig/ax
        fig, ax = plot_in.subplots()

        # --- first set ---
        if points1.shape[-1] == 1:
            # y vs t
            ax.plot(t, points1[b], color=color1, *plot_args, **plot_kwargs)
        elif points1.shape[-1] == 2:
            ax.plot(points1[b, :, 0], points1[b, :, 1], color=color1, *plot_args, **plot_kwargs)
        else:
            raise ValueError("points1 has wrong last dim")

        # --- second set (overlap) ---
        if points2.shape[-1] == 1:
            ax.plot(t, points2[b], color=color2, *plot_args, **plot_kwargs)
        elif points2.shape[-1] == 2:
            ax.plot(points2[b, :, 0], points2[b, :, 1], color=color2, *plot_args, **plot_kwargs)
        else:
            raise ValueError("points2 has wrong last dim")

        figs.append(fig)

    return figs


def visualize_preds(model: torch.nn.Module, trainloader: torch.utils.data.DataLoader, device: torch.device, num_samples: int = 3):
    model.eval()
    with torch.no_grad():
        imgs = []
        preds = []
        actual = []
        train_iter = iter(trainloader)
        for _ in range(num_samples):
            img, label = next(train_iter)
            imgs.append(img)

            img = img.to(device)
            label = label.to(device)

            actual.append(label.item())
            pred = model.predict(img)
            preds.append(pred)

        for i in range(len(imgs)):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(imgs[i].squeeze(0, 1).numpy(), cmap='gray')
            plt.title(f"Actual: {actual[i]} vs. Pred: {preds[i]}")
            plt.axis('off')

        plt.show()

def visualize_img_pred(img, pred):
    plt.imshow(img.squeeze(0, 1).numpy(), cmap='gray')
    plt.title(f"Pred: {pred}")


def visualize_uncertainty(
    rbf_net,
    z_s,
    dim0=0,
    dim1=1,
    grid_points=100,
    device="cpu",
    log_scale=False,
    vmax=None, 
    z_label="train"        
):
    """
    THIS VISUALIZATION CODE WAS MAINLY GENERATED USING CHATGPT
    Visualize uncertainty (variance = 1/beta) over a 2D slice of latent space.

    rbf_net: trained RBFNet
    z_train: [N, z_dim] tensor of encoded train latents (used to set plot range)
    dim0, dim1: which latent dims to plot
    log_scale: if True, plot log(mean variance) instead of mean variance
    vmax: if not None, clamp mean variance to this max before plotting
    """
    rbf_net.eval()
    rbf_net.to(device)

    if type(z_s) is torch.Tensor:
        z_s = z_s.detach().cpu()
    z_min = z_s[:, [dim0, dim1]].min(dim=0).values
    z_max = z_s[:, [dim0, dim1]].max(dim=0).values

    # add a little padding
    pad = 0.1 * (z_max - z_min)
    z_min = z_min - pad
    z_max = z_max + pad

    # build grid
    x_lin = torch.linspace(z_min[0].item(), z_max[0].item(), grid_points)
    y_lin = torch.linspace(z_min[1].item(), z_max[1].item(), grid_points)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")  # [G,G]

    # flatten to (G*G, 2) and pad to full latent dim if needed
    z_dim = z_s.shape[1]
    Z = torch.zeros(grid_points * grid_points, z_dim)
    Z[:, dim0] = X.reshape(-1)
    Z[:, dim1] = Y.reshape(-1)

    Z = Z.to(device)
    with torch.no_grad():
        beta = rbf_net(Z)               # [G*G, D]
        # turn precision into variance
        var = 1 / beta                  # [G*G, D]
        # aggregate over output dims: mean variance
        var_mean = var.mean(dim=1)      # [G*G]

    # keep original stats (unclamped, linear)
    print("Uncertainty stats (mean variance over outputs):")
    print(f"  min:  {var_mean.min().item():.4f}")
    print(f"  max:  {var_mean.max().item():.4f}")
    print(f"  mean: {var_mean.mean().item():.4f}")
    print(f"  std:  {var_mean.std().item():.4f}")

    # transform for plotting
    var_plot = var_mean.clone()

    # 1) optional clamp
    if vmax is not None:
        var_plot = torch.clamp(var_plot, max=vmax)

    # 2) optional log-scale
    if log_scale:
        eps = 1e-8  # avoid log(0)
        var_plot = torch.log(var_plot + eps)

    var_img = var_plot.reshape(grid_points, grid_points).cpu().numpy()

    # -------------------------------------------------------
    # Plot heatmap
    # -------------------------------------------------------
    plt.figure(figsize=(5, 4))
    title = "Uncertainty (mean variance) over latent space"
    if log_scale:
        title = "Uncertainty (log mean variance) over latent space"
    plt.title(title)

    plt.imshow(
        var_img.T,
        origin="lower",
        extent=[z_min[0].item(), z_max[0].item(), z_min[1].item(), z_max[1].item()],
        cmap="magma"
    )
    cbar_label = "mean variance"
    if log_scale:
        cbar_label = "log(mean variance)"
    plt.colorbar(label=cbar_label)

    # overlay training latents
    plt.scatter(
        z_s[:, dim0].numpy(),
        z_s[:, dim1].numpy(),
        s=5,
        c="cyan",
        alpha=0.6,
        label=f"{z_label} z"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
