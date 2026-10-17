"""Layered particle and medium compositor shared by all weathers, computed in linear light.

Scene: C = J t(d) + A (1 - t(d)), t(d) = exp(-beta d), beta in 1/m, depth d in metres. Each particle
layer at distance z, from far to near and only where z < d: C = alpha (P t(z) + A (1 - t(z))) + (1 - alpha) C

  python -m scenewx.compositor --self-test
"""
import argparse

import cv2
import numpy as np

F_PX = 2262.0          # Cityscapes focal length in pixels at 2048 px width
NATIVE_W = 2048.0


def focal_for(width):
    """Focal length in pixels for a frame resized to `width` (cropping does not change it)."""
    return F_PX * float(width) / NATIVE_W
Z_FOCUS = 12.0         # metres; driving camera focus distance
DEFOCUS_K = 14.0       # px.m; set so a particle at 1 m is a soft blob


# --- colour space ---

def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


# --- layer geometry ---

def layer_depths(z_near=1.5, z_far=50.0, k=5):
    """Log-spaced particle distances, near to far."""
    return np.geomspace(z_near, z_far, k).astype(np.float32)


def layer_weights(zs):
    """Coverage weight per layer, proportional to the depth interval it covers, normalised to mean 1."""
    zs = np.asarray(zs, dtype=np.float32)
    edges = np.concatenate([[zs[0] ** 2 / zs[1]], np.sqrt(zs[:-1] * zs[1:]), [zs[-1] ** 2 / zs[-2]]])
    dz = np.diff(edges)
    return (dz / dz.mean()).astype(np.float32)


def scale_and_tile(src, factor, shape, rng=None):
    """Scale a particle layer by `factor` (<1 means further away) and tile it to `shape`.

    Tiling keeps coverage constant: smaller particles, but more of them.
    """
    h, w = shape
    fh = max(8, int(round(src.shape[0] * factor)))
    fw = max(8, int(round(src.shape[1] * factor)))
    small = cv2.resize(src, (fw, fh), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
    ny, nx = int(np.ceil(h / fh)) + 1, int(np.ceil(w / fw)) + 1
    tiled = np.tile(small, (ny, nx))
    oy = rng.integers(0, fh) if rng is not None else 0
    ox = rng.integers(0, fw) if rng is not None else 0
    return tiled[oy:oy + h, ox:ox + w]


def defocus_sigma(z, z_focus=Z_FOCUS, k=DEFOCUS_K, f=F_PX):
    """Thin-lens circle of confusion, in pixels. Scales with focal length."""
    return float(k * (f / F_PX) * abs(1.0 / max(z, 1e-3) - 1.0 / z_focus))


def streak_length(z, v_t=0.05, f=F_PX):
    """Motion-blur length in pixels for a particle falling v m/s over an exposure of t s (v_t = v*t)."""
    return max(1.0, v_t * f / max(z, 1e-3))


def motion_kernel(length, theta_deg):
    """fspecial('motion', len, theta): a normalised line at angle theta, degrees CCW."""
    n = max(1, int(round(length)))
    if n == 1:
        return np.ones((1, 1), np.float32)
    k = np.zeros((n, n), np.float32)
    c = (n - 1) / 2.0
    t = np.deg2rad(theta_deg)
    dx, dy = np.cos(t), -np.sin(t)
    cv2.line(k, (int(round(c - dx * c)), int(round(c - dy * c))),
             (int(round(c + dx * c)), int(round(c + dy * c))), 1.0, 1, cv2.LINE_AA)
    s = k.sum()
    return k / s if s > 0 else k


def remove_dc(alpha):
    """Subtract the layer mean and clip at 0 (the uniform part is already modelled by the veil)."""
    return np.clip(alpha - float(np.mean(alpha)), 0.0, None)


def shape_layer(alpha, z, theta=None, v_t=0.05, z_focus=Z_FOCUS, k=DEFOCUS_K, dc=True, f=F_PX):
    """Apply distance-derived motion blur then defocus to one particle layer."""
    out = alpha
    if theta is not None:
        out = cv2.filter2D(out, -1, motion_kernel(streak_length(z, v_t, f), theta),
                           borderType=cv2.BORDER_REFLECT)
    s = defocus_sigma(z, z_focus, k, f)
    if s > 0.4:
        out = cv2.GaussianBlur(out, (0, 0), s)
    if dc:
        out = remove_dc(out)
    return np.clip(out, 0.0, 1.0)


# --- compositing ---

def _b3(x, shape):
    """Broadcast a scalar / 3-vector / map to HxWx3."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 0:
        return np.full(shape, float(a), np.float32)
    if a.ndim == 1:
        return np.broadcast_to(a.reshape(1, 1, 3), shape).astype(np.float32)
    if a.ndim == 2:
        return np.repeat(a[:, :, None], 3, axis=2).astype(np.float32)
    return a.astype(np.float32)


def composite(j_lin, d, beta, a_light, layers=(), occlude=True, soft=0.0, d_occ=None):
    """The layered model above. `layers` is a sequence of (z, alpha HxW, P), in any order.

    occlude=False gives the paste-on model. `soft` (metres) softens the occlusion step (0 = hard).
    `d_occ` is the occlusion depth when it differs from the medium depth `d` (P1 off, P2 on).
    """
    shape = j_lin.shape
    a3 = _b3(a_light, shape)
    beta3 = _b3(beta, shape)
    d3 = _b3(d, shape)
    o3 = d3 if d_occ is None else _b3(d_occ, shape)

    t = np.exp(-beta3 * d3)
    c = j_lin * t + a3 * (1.0 - t)

    for z, alpha, p in sorted(layers, key=lambda L: -float(L[0])):
        a = _b3(alpha, shape)
        if occlude:
            if soft > 0:
                vis = np.clip((o3 - z) / soft, 0.0, 1.0)
            else:
                vis = (o3 > z).astype(np.float32)
            a = a * vis
        tz = np.exp(-beta3 * float(z))
        p_seen = _b3(p, shape) * tz + a3 * (1.0 - tz)
        c = a * p_seen + (1.0 - a) * c
    return c


def render(bgr_u8, d, beta, a_light, layers=(), occlude=True, soft=0.0, d_occ=None):
    """sRGB uint8 in, sRGB uint8 out."""
    j = srgb_to_linear(bgr_u8[:, :, ::-1].astype(np.float32) / 255.0)
    c = composite(j, d, beta, a_light, layers, occlude, soft, d_occ)
    return (linear_to_srgb(c)[:, :, ::-1] * 255.0).round().astype(np.uint8)


# --- severity ---

def severity(clean_bgr, wx_bgr, mask=None):
    """Severity: mean |X - Y| in linear luminance. `mask` (True = count the pixel) is used to exclude sky."""
    a = srgb_to_linear(clean_bgr.astype(np.float32) / 255.0).mean(axis=2)
    b = srgb_to_linear(wx_bgr.astype(np.float32) / 255.0).mean(axis=2)
    d = np.abs(a - b)
    if mask is None:
        return float(d.mean())
    m = mask.astype(bool)
    return float(d[m].mean()) if m.any() else float(d.mean())


def solve_for_severity(render_fn, target, lo=1e-4, hi=1.0, iters=18):
    """Bisect a scalar knob so `render_fn(knob)` hits `target` severity. Monotone knobs only."""
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if render_fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# --- checks ---

def self_test():
    rng = np.random.default_rng(0)
    h, w = 64, 96
    j = rng.random((h, w, 3)).astype(np.float32)
    d = rng.uniform(2, 60, (h, w)).astype(np.float32)
    beta, a = 0.02, np.float32([0.85, 0.87, 0.9])

    # 1. no particles == plain Koschmieder
    t = np.exp(-beta * d)[:, :, None]
    assert np.allclose(composite(j, d, beta, a), j * t + a.reshape(1, 1, 3) * (1 - t), atol=1e-6)

    # 2. z=0, occlusion off == the paste-on model
    alpha = rng.random((h, w)).astype(np.float32)
    p = np.float32([1, 1, 1])
    got = composite(j, d, beta, a, [(0.0, alpha, p)], occlude=False)
    base = j * t + a.reshape(1, 1, 3) * (1 - t)
    want = alpha[:, :, None] * p.reshape(1, 1, 3) + (1 - alpha[:, :, None]) * base
    assert np.allclose(got, want, atol=1e-6)

    # 3. sky at infinity -> pure airlight, and every layer stays visible
    d_sky = np.full((h, w), 1e5, np.float32)
    got = composite(j, d_sky, beta, a, [(50.0, np.ones((h, w), np.float32), p)])
    tz = np.exp(-beta * 50.0)
    assert np.allclose(got, p.reshape(1, 1, 3) * tz + a.reshape(1, 1, 3) * (1 - tz), atol=1e-6)

    # 4. a particle behind the scene is removed
    d_near = np.full((h, w), 5.0, np.float32)
    got = composite(j, d_near, beta, a, [(30.0, np.ones((h, w), np.float32), p)])
    assert np.allclose(got, composite(j, d_near, beta, a), atol=1e-6)

    # 5. colour round trip
    x = np.linspace(0, 1, 256, dtype=np.float32)
    assert np.abs(linear_to_srgb(srgb_to_linear(x)) - x).max() < 1e-5

    # 6. coverage weights sum to the layer count (mean 1)
    zs = layer_depths()
    assert abs(layer_weights(zs).mean() - 1.0) < 1e-5

    # 7. tiling preserves mean coverage as the layer recedes
    src = (rng.random((120, 160)) < 0.05).astype(np.float32)
    near = scale_and_tile(src, 1.0, (400, 600)).mean()
    far = scale_and_tile(src, 0.25, (400, 600)).mean()
    assert abs(near - far) < 0.02, (near, far)

    print('compositor self-test: 7/7 passed')
    print(f'  layer depths   {np.round(zs, 1)}')
    print(f'  layer weights  {np.round(layer_weights(zs), 3)}')
    print(f'  streak length  ' + '  '.join(f'{z:.0f}m:{streak_length(z):.0f}px' for z in zs))
    print(f'  defocus sigma  ' + '  '.join(f'{z:.0f}m:{defocus_sigma(z):.1f}px' for z in zs))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
