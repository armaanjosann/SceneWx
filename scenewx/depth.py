"""Metric depth in metres (Depth Anything V2 Metric-Outdoor-Large) and its label-based correction, cached.

The correction applies the dual-reference cross-bilateral filter of Sakaridis et al. (ECCV 2018) to depth
per instance, then sets sky (labelId 23) to SKY_DEPTH and the ego vehicle (labelId 1) to EGO_DEPTH.
Training reads depth only from the cache, so build it first:

  python -m scenewx.depth --precompute train
  python -m scenewx.depth --precompute val
"""
import argparse
import os

import cv2
import numpy as np

from . import config

MODEL_ID = 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf'
SKY_ID, EGO_ID = 23, 1
SKY_DEPTH = 1e5
EGO_DEPTH = 2.0
MU, SIGMA_S, SIGMA_C = 5.0, 20.0, 10.0     # filter parameters from Sakaridis et al. 2018

_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        import torch
        from transformers import pipeline
        dev = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        _PIPE = pipeline('depth-estimation', model=MODEL_ID, device=dev)
    return _PIPE


CACHE_W = 1024        # cache width in px (renders are 1024x512)
CM_MAX = 655.0        # metres; the uint16 centimetre limit


def save_depth(path, d):
    """Save depth as 16-bit centimetres, downscaled to CACHE_W wide."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = int(round(d.shape[0] * CACHE_W / d.shape[1]))
    small = cv2.resize(d, (CACHE_W, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, (np.clip(small, 0, CM_MAX) * 100).astype(np.uint16))


def load_depth(path, shape=None):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
    if shape is not None and d.shape[:2] != tuple(shape[:2]):
        d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return d


def metric_depth(bgr, cache_key=None):
    """Depth in metres at the input resolution. Cached under SCENEWX_DEPTH_CACHE/<cache_key>.png."""
    if cache_key:
        pp = os.path.join(config.depth_cache(), cache_key + '.png')
        if os.path.exists(pp):
            return load_depth(pp, bgr.shape)
        p = os.path.join(config.depth_cache(), cache_key + '.npy')
        if os.path.exists(p):
            return np.load(p)
    from PIL import Image
    h, w = bgr.shape[:2]
    out = _pipe()(Image.fromarray(bgr[:, :, ::-1]))
    d = np.asarray(out['predicted_depth'], dtype=np.float32)
    if d.ndim == 3:
        d = d[0]
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    if cache_key:
        save_depth(os.path.join(config.depth_cache(), cache_key + '.png'), d)
    return d


def dual_reference_filter(d, bgr, region, mu=MU, sigma_s=SIGMA_S, sigma_c=SIGMA_C):
    """Sakaridis et al. (ECCV 2018) Eq. 2 applied to depth; `region` holds instance or label ids.

    Uses cv2.ximgproc.jointBilateralFilter if opencv-contrib is installed, else a plain bilateral filter.
    The two give different results, so build the cache with the OpenCV build you report results with.
    """
    k = int(max(3, round(sigma_s * 3)) // 2 * 2 + 1)
    num = np.zeros_like(d, dtype=np.float32)
    den = np.zeros_like(d, dtype=np.float32)
    for rid in np.unique(region):
        m = (region == rid).astype(np.float32)
        num += m * cv2.GaussianBlur(d * m, (k, k), sigma_s)
        den += m * cv2.GaussianBlur(m, (k, k), sigma_s)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    if hasattr(cv2, 'ximgproc'):
        colour = cv2.ximgproc.jointBilateralFilter(lab, d, -1, sigma_c, sigma_s)
    else:
        colour = cv2.bilateralFilter(d, -1, sigma_c, sigma_s)
    return (num + mu * colour) / np.maximum(den + mu, 1e-6)


def repair(d, label_ids, instance_ids, bgr):
    """Label-based correction; returns depth in metres with sky at SKY_DEPTH."""
    d = d.astype(np.float32).copy()
    sky = (label_ids == SKY_ID)
    region = np.where(instance_ids >= 1000, instance_ids.astype(np.int32),
                      label_ids.astype(np.int32) - 1000)
    d = np.where(sky, d, dual_reference_filter(d, bgr, region))
    d[label_ids == EGO_ID] = EGO_DEPTH
    d[sky] = SKY_DEPTH
    return np.maximum(d, 0.1)


def load_frame(name, split='val', want_instances=True):
    """Clear frame (BGR), labelIds and optionally instanceIds for a Cityscapes frame id such as
    'weimar_000011_000019'. Instances are only needed to build the corrected depth."""
    city = name.split('_')[0]
    cs, gt = config.cityscapes(), config.gtfine()
    bgr = cv2.imread(f'{cs}/{split}/{city}/{name}_leftImg8bit.png')
    lab = cv2.imread(f'{gt}/{split}/{city}/{name}_gtFine_labelIds.png', cv2.IMREAD_UNCHANGED)
    ins = (cv2.imread(f'{gt}/{split}/{city}/{name}_gtFine_instanceIds.png', cv2.IMREAD_UNCHANGED)
           if want_instances else None)
    return bgr, lab, ins


def has_repaired(cache_key):
    return os.path.exists(os.path.join(config.depth_cache(), 'repaired', cache_key + '.png'))


def repaired_depth(bgr, label_ids, instance_ids, cache_key=None):
    """Corrected depth, cached like the raw map."""
    if cache_key:
        p = os.path.join(config.depth_cache(), 'repaired', cache_key + '.png')
        if os.path.exists(p):
            d = load_depth(p, bgr.shape)
            # resizing the cached map blurs the sky edge, so restore sky and ego from the labels
            d[label_ids == EGO_ID] = EGO_DEPTH
            d[label_ids == SKY_ID] = SKY_DEPTH
            return d
    d = repair(metric_depth(bgr, cache_key=cache_key), label_ids, instance_ids, bgr)
    if cache_key:
        save_depth(os.path.join(config.depth_cache(), 'repaired', cache_key + '.png'), np.minimum(d, CM_MAX))
    return d


def frames(n=4, split='val', city='frankfurt'):
    """First `n` frame ids of one city, sorted (the calibration sample)."""
    import glob
    cs = config.cityscapes()
    return sorted(os.path.basename(p)[:-16] for p in glob.glob(f'{cs}/{split}/{city}/*_leftImg8bit.png'))[:n]


def frames_for(split):
    """Every frame id of a split, sorted by city then name."""
    cs = config.cityscapes()
    out = []
    for city in sorted(os.listdir(f'{cs}/{split}')):
        if city.startswith('.'):
            continue
        for p in sorted(os.listdir(f'{cs}/{split}/{city}')):
            if p.endswith('_leftImg8bit.png'):
                out.append(p[:-16])
    return out


def precompute(split='train', limit=None):
    """Cache raw and corrected metric depth for a whole split (about 1 s per frame on a laptop GPU)."""
    import time
    names = frames_for(split)
    if limit:
        names = names[:limit]
    t0, made, bad = time.perf_counter(), 0, []
    for i, nm in enumerate(names):
        if has_repaired(f'{split}/{nm}'):
            continue
        bgr, lab, ins = load_frame(nm, split)
        # skip and record frames that fail to decode
        if bgr is None or lab is None or ins is None:
            bad.append(nm)
            continue
        repaired_depth(bgr, lab, ins, cache_key=f'{split}/{nm}')   # also fills the raw cache
        made += 1
        if made % 25 == 0:
            el = time.perf_counter() - t0
            print(f'  {i + 1}/{len(names)}  {made} new  {el / 60:.1f} min  '
                  f'({el / made:.2f} s/frame)', flush=True)
    print(f'done: {made} new depth maps for {split} ({len(names)} frames total)')
    if bad:
        bp = os.path.join(config.depth_cache(), f'unreadable_{split}.txt')
        open(bp, 'w').write('\n'.join(bad) + '\n')
        print(f'  {len(bad)} frames could not be decoded; listed in {bp}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--precompute', default=None, help='split name, e.g. train')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    if a.precompute:
        precompute(a.precompute, a.limit)
