from math import ceil, sqrt, pi

import astra
import numpy as np


def generate_parallel_beam_geometry_params(N_pix=200, N_views=150, vol_size=0.36, N_detectors=None):
    vol_geom_params = [N_pix, N_pix, -vol_size/2, vol_size/2, -vol_size/2, vol_size/2]
    N_detectors = ceil(2 * sqrt(2) * N_pix) if N_detectors is None else N_detectors
    angles = np.linspace(0, pi, N_views + 1)[:-1]
    proj_geom_params = ['parallel', vol_size / (2 * N_pix), N_detectors, angles]
    return {'vol_geom': vol_geom_params, 'proj_geom': proj_geom_params}


def generate_fan_beam_geometry_params(N_pix=200, N_views=800, vol_size=0.36, N_detectors=None):
    vol_geom_params = [N_pix, N_pix, -vol_size/2, vol_size/2, -vol_size/2, vol_size/2]
    N_detectors = ceil(2 * sqrt(2) * N_pix) if N_detectors is None else N_detectors
    radius = 2 * vol_size
    full_detector_width = 2 * radius * vol_size / (radius - vol_size / 2)
    proj_geom_params = ['fanflat', full_detector_width / (2 * N_pix), N_detectors, np.linspace(0, 2 * pi, N_views), radius, radius]
    return {'vol_geom': vol_geom_params, 'proj_geom': proj_geom_params}


def generate_geometry_params(**kwargs):
    if 'geom_type' in kwargs:
        geom_type = kwargs.pop('geom_type')
    else:
        geom_type = 'parallel'
    if geom_type == 'parallel':
        return generate_parallel_beam_geometry_params(**kwargs)
    elif geom_type == 'fanflat':
        return generate_fan_beam_geometry_params(**kwargs)
    else:
        raise NotImplementedError('specified geometry "{}" not implemented'.format(geom_type))


def generate_geometries(**kwargs):
    params = generate_geometry_params(**kwargs)
    return {'vol_geom': astra.create_vol_geom(*params['vol_geom']),
            'proj_geom': astra.create_proj_geom(*params['proj_geom'])}
