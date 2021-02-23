import astra
import numpy as np
import torch


class RayTransform:
    def __init__(self, vol_geom, proj_geom):
        self.im_id = astra.data2d.create('-vol', vol_geom)
        self.sino_id = astra.data2d.create('-sino', proj_geom)
        self._cuda = torch.cuda.is_available()
        self.proj_id = astra.projector.create({'type': 'cuda' if self._cuda else 'line',
                                               'VolumeGeometry': vol_geom,
                                               'ProjectionGeometry': proj_geom,
                                               'options': {}})
                                                                       
        forward_config = astra.astra_dict('FP_CUDA' if self._cuda else 'FP')
        forward_config['VolumeDataId'] = self.im_id
        forward_config['ProjectionDataId'] = self.sino_id
        forward_config['ProjectorId'] = self.proj_id
        self.forward_alg_id = astra.algorithm.create(forward_config)
        
        backward_config = astra.astra_dict('BP_CUDA' if self._cuda else 'BP')
        backward_config['ProjectionDataId'] = self.sino_id
        backward_config['ReconstructionDataId'] = self.im_id
        backward_config['ProjectorId'] = self.proj_id
        self.backward_alg_id = astra.algorithm.create(backward_config)
        
        fbp_config = astra.astra_dict('FBP_CUDA' if self._cuda else 'FBP')
        fbp_config['ProjectionDataId'] = self.sino_id
        fbp_config['ReconstructionDataId'] = self.im_id
        fbp_config['ProjectorId'] = self.proj_id
        self.fbp_alg_id = astra.algorithm.create(fbp_config)
        
        def mb_ray_transform(x):
            sinograms = []
            for i in range(x.shape[0]):
                astra.data2d.store(self.im_id, x[i, 0, :, :].detach().cpu().numpy())
                astra.algorithm.run(self.forward_alg_id)
                sinograms.append(astra.data2d.get(self.sino_id))
            sinograms = torch.tensor(np.stack(sinograms, axis=0), device=x.device)
            return sinograms.view(sinograms.shape[0], 1, *sinograms.shape[1:])
        
        def mb_backprojection(z):
            ims = []
            for i in range(z.shape[0]):
                astra.data2d.store(self.sino_id, z[i, 0, :, :].detach().cpu().numpy())
                astra.algorithm.run(self.backward_alg_id)
                ims.append(astra.data2d.get(self.im_id))
            ims = torch.tensor(np.stack(ims, axis=0), device=z.device)
            return ims.view(ims.shape[0], 1, *ims.shape[1:])
        
        def mb_fbp(z):   
            ims = []
            for i in range(z.shape[0]):
                astra.data2d.store(self.sino_id, z[i, 0, :, :].detach().cpu().numpy())
                astra.algorithm.run(self.fbp_alg_id)
                ims.append(astra.data2d.get(self.im_id))
            ims = torch.tensor(np.stack(ims, axis=0), device=z.device)
            return ims.view(ims.shape[0], 1, *ims.shape[1:])
        
        self._mb_ray_transform = mb_ray_transform
        self._mb_backprojection = mb_backprojection
        self._mb_fbp = mb_fbp
        
        class _RayTransform(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return self._mb_ray_transform(x)
            
            @staticmethod
            def backward(ctx, grad_out):
                return self._mb_backprojection(grad_out)
            
        class _BackProjection(torch.autograd.Function):
            @staticmethod
            def forward(ctx, z):
                return self._mb_backprojection(z)
            
            @staticmethod
            def backward(ctx, grad_out):
                return self._mb_ray_transform(grad_out)
        self._autograd_raytransform = _RayTransform.apply
        self._autograd_backprojection = _BackProjection.apply
        
    def __call__(self, x):
        return self._autograd_raytransform(x)
    
    def adjoint(self, z):
        return self._autograd_backprojection(z)
    
    def fbp(self, z):
        return self._mb_fbp(z)
    
    def __del__(self):
        astra.data2d.delete(self.im_id)
        self.im_id = None
        astra.data2d.delete(self.sino_id)
        self.sino_id = None
        astra.algorithm.delete(self.forward_alg_id)
        self.forward_alg_id = None
        astra.algorithm.delete(self.backward_alg_id)
        self.backward_alg_id = None
        astra.algorithm.delete(self.fbp_alg_id)
        self.fbp_alg_id = None
    
