import torch


def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    """
    This code was taken from https://github.com/Gabe-YHLee/IRVAE-public
    """
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
    """
    This code was taken from https://github.com/Gabe-YHLee/IRVAE-public
    """
    J = jacobian_decoder_jvp_parallel(func, z, v=None)
    G = torch.einsum('nij,nik->njk', J, J)
    return G


def log_riemann_volume(func, z):
    G = get_pullbacked_Riemannian_metric(func, z)
    return torch.log(torch.sqrt(torch.det(G)) + 1e-8)

def log_random_riemann_volume(mean_func, var_func, z):
    G = get_pullbacked_Riemannian_metric(mean_func, z) + get_pullbacked_Riemannian_metric(var_func, z)
    return torch.log(torch.sqrt(torch.det(G)) + 1e-8)