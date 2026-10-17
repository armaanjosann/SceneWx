"""Rain: the streak model of Li, Cheong & Tan (CVPR 2019, `datagen/render_rain.m`), placed in depth.

Each streak layer sits at a distance that sets its size, length, blur and visibility. Streaks are
alpha-composited with the airlight as their radiance. `uniform=True` gives the single-layer model:
one streak length for the whole frame, no size variation, no occlusion.
"""
import cv2
import numpy as np

from . import compositor as C

# Li render_rain.m: noise mean m = density*(0.2 + U(0,0.05)) with density = -4; variance v = 0.7+U(0,0.3)
LI_DENSITY, LI_INTENSITY = -4.0, 0.7
THETAS = (80, 85, 90, 95, 100)                 # Li generate_dataset.m: theta = s*5 + 75
RAIN_VT = 0.05                                 # v * t_exp in metres (5 m/s fall, 1/100 s exposure)
BASE_Q = 0.045       # fraction of pixels seeded with a drop, per unit shell weight
_L_REF = 13.0        # px; streak length at the reference distance, for Li's linear compensation
GAIN_REF = 2.2       # streak alpha at strength 1, set by eye
BETA_REF = 0.0040   # veil extinction at strength 1, 1/m (MOR about 750 m)
LEVELS = {'light': 0.45, 'moderate': 1.0, 'heavy': 2.2}          # rainfall strength


def li_noise(shape, rng):
    """Li's noise field: imnoise(zeros, 'gaussian', m, v) then a bicubic rescale and random crop."""
    h, w = shape
    m = LI_DENSITY * (0.2 + rng.random() * 0.05)
    v = LI_INTENSITY + rng.random() * 0.3
    s = 1.01 + rng.random() * 0.2
    n = np.clip(rng.normal(m, np.sqrt(v), (h, w)), 0, 1).astype(np.float32)
    big = cv2.resize(n, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_CUBIC)
    oy = rng.integers(0, big.shape[0] - h + 1)
    ox = rng.integers(0, big.shape[1] - w + 1)
    return big[oy:oy + h, ox:ox + w]


def streak_layers(shape, zs, gain, theta, rng, uniform=False, z_ref=8.0, f=None):
    """One alpha layer per distance; returns [(z, alpha)].

    Near layers are magnified and far ones shrunk, streak length is v t f / z px, and alpha is scaled
    linearly with streak length as in Li's code. uniform=True uses one blur length from Li's range
    [21, 60] px.
    """
    w_cov = C.layer_weights(zs)
    f = C.focal_for(shape[1]) if f is None else f
    lref = _L_REF * f / C.F_PX
    if uniform:
        l = float(rng.integers(21, 61)) * f / C.F_PX
        a = li_noise(shape, rng)
        a = cv2.filter2D(a, -1, C.motion_kernel(l, theta), borderType=cv2.BORDER_REFLECT)
        a = C.remove_dc(a) * (l / _L_REF)
        return [(0.0, np.clip(gain * float(w_cov.sum()) * a, 0, 1))]

    out = []
    for z, wc in zip(zs, w_cov):
        n = li_noise(shape, rng)
        n = C.scale_and_tile(n, z_ref / float(z), shape, rng)
        # the shell weight sets how many pixels are seeded, not their amplitude
        q = float(np.clip(BASE_Q * float(wc), 1e-4, 0.9))
        n = np.where(n >= np.quantile(n, 1.0 - q), n, 0.0).astype(np.float32)
        a = C.shape_layer(n, float(z), theta=theta, v_t=RAIN_VT, f=f)
        a = a * (C.streak_length(float(z), RAIN_VT, f) / lref)
        out.append((float(z), np.clip(gain * a, 0, 1)))
    return out


def render_particles(bgr, d, beta, a_light, zs, strength, theta, seed, uniform=False, occlude=True,
                     f_px=None, d_occ=None):
    """Render rain on a frame or crop. `strength` scales the veil and the streaks together.

    `beta` is unused (the veil is BETA_REF * strength).
    """
    rng = np.random.default_rng(seed)
    layers = streak_layers(bgr.shape[:2], zs, GAIN_REF * strength, theta, rng, uniform=uniform,
                           f=f_px)
    return C.render(bgr, d, BETA_REF * strength, a_light,
                    [(z, al, a_light) for z, al in layers], occlude=occlude, d_occ=d_occ)
