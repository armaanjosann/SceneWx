"""Training conditions ("cells"): one setting each of medium (P1), particles (P2) and depth (P4).

'repaired' depth is corrected with scene labels and 'scrambled' is raw depth flipped vertically. Fog has
no particles, so fog P2 equals base and fog full equals P1. Each condition is severity-matched per level
by calibrate_cells.py.
"""
import json
import os

import numpy as np

from . import compositor as C
from . import config
from . import fog as FOG
from . import rain as RAIN
from . import snow as SNOW

MODULES = {'fog': FOG, 'rain': RAIN, 'snow': SNOW}

# name -> property settings; the level is given separately
CELLS = {
    'base':      dict(medium='uniform', particles='fixed', depth='raw',       control=None),
    'P1':        dict(medium='depth',   particles='fixed', depth='raw',       control=None),
    'P2':        dict(medium='uniform', particles='depth', depth='raw',       control=None),
    'full':      dict(medium='depth',   particles='depth', depth='raw',       control=None),
    'P4':        dict(medium='depth',   particles='depth', depth='repaired',  control=None),
    'scrambled': dict(medium='depth',   particles='depth', depth='scrambled', control=None),
    'noise':     dict(medium=None,      particles=None,    depth=None,        control='noise'),
    'identity':  dict(medium=None,      particles=None,    depth=None,        control='identity'),
}
LEVELS = ('light', 'moderate', 'heavy')
ORDER = ['base', 'P1', 'P2', 'full', 'P4', 'scrambled', 'noise', 'identity']


def load_cell_calibration(path=None):
    """The frozen severity settings: {weather: {level: {target_severity, target_source, cells}}}."""
    path = path or config.calibration()
    if not os.path.exists(path):
        raise SystemExit(f'no cell calibration at {path}, set SCENEWX_CALIBRATION or run '
                         f'python -m scenewx.calibrate_cells')
    return json.load(open(path))


def cells_for(weather):
    """Cells that exist for a weather. Fog has no particles, so P2 == base and full == P1."""
    if weather == 'fog':
        return [c for c in ORDER if c != 'P2']
    return list(ORDER)


def scene_depth(cell, raw, rep):
    """Depth map used by this cell's scene-dependent parts. Whole frame only (the scramble flips the frame)."""
    d = CELLS[cell]['depth']
    if d == 'repaired':
        return rep
    if d == 'scrambled':
        return FOG.scramble_depth(raw)
    return raw


def depths_for_cell(weather, cell, raw, rep, params):
    """(d_medium, d_occlusion) for a cell, on the whole frame.

    A uniform medium uses the calibrated optical depth for fog and the frame's mean depth for rain and
    snow. Occlusion always uses the scene depth.
    """
    spec = CELLS[cell]
    if spec['control']:
        return None, None
    d_scene = scene_depth(cell, raw, rep)
    if spec['medium'] == 'depth':
        d_med = d_scene
    elif weather == 'fog':
        d_med = np.full_like(raw, params['optical_depth'] / params['beta'])
    else:
        d_med = np.full_like(raw, float(np.mean(np.minimum(raw, 1e3))))
    return d_med, d_scene


def noise_layer(bgr, sigma, seed):
    """Noise control: zero-mean Gaussian noise with std `sigma` in linear light."""
    rng = np.random.default_rng(seed)
    j = C.srgb_to_linear(bgr[:, :, ::-1].astype(np.float32) / 255.0)
    n = rng.normal(0.0, float(sigma), size=j.shape).astype(np.float32)
    out = np.clip(j + n, 0.0, 1.0)
    return (C.linear_to_srgb(out)[:, :, ::-1] * 255.0).round().astype(np.uint8)


def render_cell(bgr, d_med, d_occ, a_light, weather, cell, params, seed, f_px=None):
    """Render one cell on a frame or crop. `d_med`/`d_occ` are whole-frame maps from `depths_for_cell`,
    cropped with the image; `f_px` is the full frame's focal length (compositor.focal_for)."""
    spec = CELLS[cell]
    if spec['control'] == 'identity':
        return bgr.copy()
    if spec['control'] == 'noise':
        return noise_layer(bgr, params['sigma'], seed)
    if weather == 'fog':
        return C.render(bgr, d_med, params['beta'], a_light)
    mod = MODULES[weather]
    kw = dict(zs=C.layer_depths(), strength=params['strength'], seed=seed,
              uniform=(spec['particles'] == 'fixed'), occlude=(spec['particles'] == 'depth'),
              f_px=f_px, d_occ=d_occ)
    if weather == 'rain':
        return mod.render_particles(bgr, d_med, None, a_light, theta=mod.THETAS[seed % len(mod.THETAS)], **kw)
    return mod.render_particles(bgr, d_med, a_light, **kw)


def render_cell_frame(bgr, lab, raw, rep, weather, cell, params, seed, a_light=None):
    """Whole-frame convenience: depths, airlight, render. `a_light` overrides the fog airlight."""
    d_med, d_occ = depths_for_cell(weather, cell, raw, rep, params)
    if a_light is None:
        a_light = FOG.airlight(bgr, lab)
    return render_cell(bgr, d_med, d_occ, a_light, weather, cell, params, seed)


def opt_name(weather, cell, level, crop, iters, seed, arch='nafnet'):
    """Config / experiment name, e.g. nafnet_fog_full_moderate_c256_i60000_s10."""
    return f'{arch}_{weather}_{cell}_{level}_c{crop}_i{iters}_s{seed}'


def parse_opt(opt):
    """Inverse of opt_name; returns None for names that do not follow it."""
    parts = opt.split('_')
    if len(parts) < 7 or not parts[-3].startswith('c') or not parts[-2].startswith('i'):
        return None
    seed = int(parts[-1][1:]); iters = int(parts[-2][1:]); crop = int(parts[-3][1:])
    level = parts[-4]; cell = parts[-5]
    rest = parts[1:-5]
    return dict(weather='_'.join(rest), cell=cell, level=level, crop=crop, iters=iters, seed=seed,
                arch=parts[0])
