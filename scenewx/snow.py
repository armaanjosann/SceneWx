"""Snow: CSD flake masks (Chen et al., ICCV 2021) placed in depth, used as compositor alpha.

The masks are not included; point SCENEWX_CSD_MASKS at a folder of CSD mask .tif files.
`uniform=True` gives the single-layer model: the mask alpha-blended as shipped, no occlusion.
"""
import glob

import cv2
import numpy as np

from . import compositor as C
from . import config

SNOW_VT = 0.010      # v * t_exp in metres: ~1 m/s fall, 1/100 s exposure
_L_REF = 3.0         # px, reference streak length for the linear blur compensation
GAIN_REF = 2.0
BETA_REF = 0.006     # veil extinction at strength 1, 1/m
BASE_Q = 0.040
LEVELS = {'light': 0.45, 'moderate': 1.0, 'heavy': 2.2}

_MASKS = None


def mask_paths():
    """Sorted CSD mask files. Masks are drawn by index, so a different file set gives different snow."""
    global _MASKS
    if _MASKS is None:
        root = config.csd_masks()
        _MASKS = sorted(glob.glob(f'{root}/*.tif'))
        if not _MASKS:
            raise SystemExit(f'no CSD masks in {root}')
    return _MASKS


def load_mask(rng):
    """One CSD flake layer, as a float alpha in [0, 1]."""
    p = mask_paths()[int(rng.integers(0, len(mask_paths())))]
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m.astype(np.float32) / 255.0)


def flake_layers(shape, zs, gain, rng, uniform=False, z_ref=8.0, theta=85, f=None):
    """One alpha layer per distance, same construction as rain but with CSD flakes."""
    w_cov = C.layer_weights(zs)
    f = C.focal_for(shape[1]) if f is None else f
    lref = _L_REF * f / C.F_PX
    if uniform:
        # tile rather than resize, so the 4:3 mask is not stretched to 2:1
        a = C.scale_and_tile(load_mask(rng), 1.0, shape, rng)
        return [(0.0, np.clip(gain * float(w_cov.sum()) * a, 0, 1))]

    out = []
    for z, wc in zip(zs, w_cov):
        n = C.scale_and_tile(load_mask(rng), z_ref / float(z), shape, rng)
        q = float(np.clip(BASE_Q * float(wc), 1e-4, 0.9))
        thr = np.quantile(n[n > 0], 1.0 - min(q / max((n > 0).mean(), 1e-6), 1.0)) if (n > 0).any() else 1.0
        n = np.where(n >= thr, n, 0.0).astype(np.float32)
        a = C.shape_layer(n, float(z), theta=theta, v_t=SNOW_VT, f=f)
        a = a * (C.streak_length(float(z), SNOW_VT, f) / lref)
        out.append((float(z), np.clip(gain * a, 0, 1)))
    return out


def render_particles(bgr, d, a_light, zs, strength, seed, uniform=False, occlude=True, f_px=None,
                     d_occ=None):
    """Render snow on a frame or crop. `strength` scales the veil and the flakes together."""
    rng = np.random.default_rng(seed)
    layers = flake_layers(bgr.shape[:2], zs, GAIN_REF * strength, rng, uniform=uniform, f=f_px)
    return C.render(bgr, d, BETA_REF * strength, a_light,
                    [(z, al, a_light) for z, al in layers], occlude=occlude, d_occ=d_occ)
