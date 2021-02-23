from .raytransform import RayTransform
from ..forward_operators import ForwardOperator

import torch

    
class RadonForwardOperator(ForwardOperator):
    def __init__(self, vol_geom, proj_geom):
        super().__init__()
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.raytransform = RayTransform(vol_geom, proj_geom)

    def __call__(self, x):
        return self.raytransform(x)

    def vec_jac_prod(self, x, z):
        return self.raytransform.adjoint(z)

    def pseudoinverse(self, y):
        return self.raytransform.fbp(y)

class PoissonForwardOperator(ForwardOperator):
    def __init__(self, vol_geom, proj_geom, mu_max=81.35858, N_in=10000):
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.raytransform = RayTransform(vol_geom, proj_geom)
        self.mu_max = mu_max
        self.N_in = N_in

    def __call__(self, x):
        return self.N_in * torch.exp(-self.mu_max * self.raytransform(x))

    def vec_jac_prod(self, x, z, fx=None):
        fx = self.raytransform(x) if fx is None else fx
        return -self.mu_max * self.raytransform.adjoint(fx * z)

    def pseudoinverse(self, y, thres_val=1e-8):
        return self.raytransform.fbp(-torch.log(torch.clamp(y / self.N_in, thres_val)) / self.mu_max)
    
    def simulate_counts(self, x):
        return torch.poisson(self(x))

    def sinogram_from_counts(self, y, thres_val=1e-8):
        return -torch.log(torch.clamp(y / self.N_in, thres_val)) / self.mu_max

    def simulate_sinograms(self, x):
        return self.sinogram_from_counts(self.simulate_counts(x))
