import torch
from stochman.geodesic import geodesic_minimizing_energy
from stochman.curves import CubicSpline

def find_geodesic(model, z1, z2, attempts=100, num_nodes=20):
    geodesic_found = False
    curve = CubicSpline(z1, z2, num_nodes=num_nodes)
    attempt = 0
    while not geodesic_found and attempt < attempts:
        geodesic_found = geodesic_minimizing_energy(curve, model).item()
        attempt += 1
    if not geodesic_found:
        current_length = curve.euclidean_length().item()
        c, _ = model.connecting_geodesic(z1, z2)
        return curve if current_length < c.euclidean_length() else c
    return curve

def points_on_geodesic(curve: CubicSpline, num_points: int = 100, t0: float = 0.0, t1: float = 1.0):
    """
    This was created through combining code from the Stochman library.
    """
    with torch.no_grad():
        t = torch.linspace(t0, t1, num_points, dtype=curve.begin.dtype, device=curve.device)
        coeffs = curve._get_coeffs()  # Bx(num_edges)x4xD
        no_batch = t.ndim == 1
        if no_batch:
            t = t.expand(coeffs.shape[0], -1)  # Bx|t|
        retval = curve._eval_polynomials(t, coeffs)  # Bx|t|xD
        retval += curve._eval_straight_line(t)
        if no_batch and retval.shape[0] == 1:
            retval.squeeze_(0)  # |t|xD
        points = retval
        if len(points.shape) == 2:
                points.unsqueeze_(0)  # 1xNxD
    return points, t
