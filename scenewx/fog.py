"""Fog: the compositor with no particle layers, t = exp(-beta d).

Extinction per level from Sakaridis et al. (2018): beta = 0.005 / 0.010 / 0.030 per metre (meteorological
optical range about 600 / 300 / 100 m). Other fog conditions are calibrated to the 'full' severity.
"""
import json
import os

import numpy as np

from . import compositor as C
from . import config

BETAS = {'light': 0.005, 'moderate': 0.010, 'heavy': 0.030}   # 1/m, 'full' condition; MOR 600/300/100 m

_A_CACHE = None


def _airlight_cache_path():
    return os.path.join(config.depth_cache(), 'airlight.json')


def airlight_cached(bgr, label_ids, key):
    """`airlight`, cached on disk (SCENEWX_DEPTH_CACHE/airlight.json) under `key`."""
    global _A_CACHE
    path = _airlight_cache_path()
    if _A_CACHE is None:
        _A_CACHE = json.load(open(path)) if os.path.exists(path) else {}
    if key not in _A_CACHE:
        _A_CACHE[key] = [float(v) for v in airlight(bgr, label_ids)]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f'.{os.getpid()}.tmp'
        json.dump(_A_CACHE, open(tmp, 'w'))
        os.replace(tmp, path)
    return np.float32(_A_CACHE[key])


def airlight(bgr, label_ids=None, pct=95.0):
    """Fog airlight (linear RGB): the `pct` percentile of linear luminance, nearly neutral in colour.

    p95 matches the airlight implied by Foggy Cityscapes-DBF (beta = 0.01) to within 4% on average.
    `label_ids` is unused.
    """
    lin = C.srgb_to_linear(bgr[:, :, ::-1].astype(np.float32) / 255.0)
    lum = lin.mean(axis=2)
    level = float(np.clip(np.percentile(lum, pct), 0.30, 0.95))
    bright = lin[lum >= np.percentile(lum, 99.0)].reshape(-1, 3).mean(axis=0)
    chroma = bright / max(float(bright.max()), 1e-6)
    col = 0.65 * np.ones(3, np.float32) + 0.35 * chroma          # mostly neutral, a little scene tint
    return (col / col.max() * level).astype(np.float32)


def scramble_depth(d_self):
    """Scrambled-depth control: the frame's depth flipped vertically (same histogram, Spearman with
    the true depth -0.56 to -0.73). Apply to the whole frame, not a crop."""
    return np.ascontiguousarray(d_self[::-1]).astype(np.float32)
