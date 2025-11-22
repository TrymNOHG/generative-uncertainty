import torch
import torch.nn as nn
import random

from collections import defaultdict

from sklearn.cluster import KMeans
from models.base import BaseModel

class RBFNet(BaseModel):
    """
    This network was greatly inspired by Latent Space Oddity: On the Curvature of Deep Generative Models by Arvanitidis et. al 2021.
    """
    def __init__(self, z_train: torch.Tensor, output_dim:int,  K: int, a: float, *args, **kwargs):
        """
        Parameters:
            z_train - Encoded trainset
            K - Number of centers.
            a - Hyperparameter for controlling curvature of Riemannian mentric
        """
        super().__init__(*args, **kwargs)

        kmeans = KMeans(n_clusters=K, random_state=42).fit(z_train)
        clusters = kmeans.labels_

        self.register_buffer("centers", torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)) # K dimension
        self.register_buffer("bandwidths", self._bandwidth(z_train, clusters, a)) # K dimension
        self.register_buffer("eps", torch.full((output_dim,), 1e-6)) # D dimensional

        self.W = nn.Linear(in_features=K, out_features=output_dim, bias=False) # D x K dimensions (Cannot be negative)
        self.loss_func = torch.nn.MSELoss()

        # To ensure non-negativity, during training an additional projection step should be used. But instead I just clamp it.

    def forward(self, x: torch.Tensor):
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0) 
        l2 = diff.pow(2).sum(dim=-1)
        v_k = torch.exp(-self.bandwidths * l2)
        beta = self.W(v_k) + self.eps
        return beta


    def loss(self, z, target_var):
        beta = self.forward(z).reshape_as(target_var).clamp_min(1e-6)
        nll = 0.5 * (-torch.log(beta) + beta * target_var) # From Gaussian NLL, but since beta = 1/var, then I can take -log(beta) to get more numerical stability.
        return nll.mean()


    def loss_legend(self,):
        return ["MSE"]
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor):
        return None

    def short_name(self,):
        return "RBF"

    def _bandwidth(self, z_train: torch.Tensor, clusters: list[int], a: float,):
        cluster_dict = defaultdict(list)
        for i, cluster in enumerate(clusters):
            cluster_dict[cluster].append(z_train[i])
        
        bandwidths = []
        for i, center in enumerate(self.centers):
            C_k = torch.stack(cluster_dict[i])
            l2 = (C_k - center).pow(2).sum(dim=1)
            bandwidth = 0.5 * (a * l2.mean()).pow(-2)
            bandwidth = bandwidth.clamp_max(1e6)
            bandwidths.append(bandwidth)
            
        return nn.Parameter(torch.tensor(bandwidths, dtype=torch.float32), requires_grad=False)
    
    def _project(self):
        # To ensure W stays positive.
        pass

