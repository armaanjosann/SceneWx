"""Shared calibration pieces: the calibration sample, mean severity over it, and real-weather targets.

One parameter per (weather, level, condition) is fitted on a held-out sample and frozen. Severity is
mean |X - Y| in linear luminance with sky excluded. The fit itself is in calibrate_cells.py.
"""
import cv2
import numpy as np

from . import compositor as C
from . import depth as D

SAMPLE_SCALE = 0.5      # half resolution (1024x512), same as rendering

# Severity targets from real paired data, light / moderate / heavy = p10 / median / p90:
#   rain  GT-RAIN (an upper bound: the clean frame is captured minutes later)
#   snow  WeatherBench snow (sprayed in front of a tripod camera)
# Fog has no real paired set; it is anchored on Sakaridis et al.'s beta values (fog.BETAS).
REAL_TARGETS = {
    'rain': {'light': 0.0280, 'moderate': 0.0428, 'heavy': 0.0817},
    'snow': {'light': 0.0151, 'moderate': 0.0361, 'heavy': 0.0810},
}


def load_sample(n, scale=SAMPLE_SCALE, split='val', city='frankfurt'):
    """The first `n` frames of one validation city, with raw and corrected depth, at `scale`."""
    out = []
    for nm in D.frames(n, split=split, city=city):
        bgr, lab, ins = D.load_frame(nm, split)
        raw = D.metric_depth(bgr, cache_key=f'{split}/{nm}')
        rep = D.repair(raw, lab, ins, bgr)
        if scale != 1.0:
            sz = (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale))
            bgr = cv2.resize(bgr, sz, interpolation=cv2.INTER_AREA)
            lab = cv2.resize(lab, sz, interpolation=cv2.INTER_NEAREST)
            raw = cv2.resize(raw, sz, interpolation=cv2.INTER_NEAREST)
            rep = cv2.resize(rep, sz, interpolation=cv2.INTER_NEAREST)
        out.append((nm, bgr, lab, raw, rep))
    return out


def mean_severity(sample, render):
    """Severity averaged over the sample, sky excluded."""
    return float(np.mean([C.severity(bgr, render(bgr, lab, raw, rep, i), mask=(lab != D.SKY_ID))
                          for i, (_, bgr, lab, raw, rep) in enumerate(sample)]))
