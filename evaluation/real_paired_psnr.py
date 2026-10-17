"""Paired PSNR on real rain and snow: GT-RAIN (Ba et al. 2022) and the WeatherBench test sets.

Reports input and output PSNR against the clean frame and the change |out - in|. GT-RAIN clean frames
are captured minutes apart, so illumination drift is included. Data: SCENEWX_GTRAIN (scene folders at
<root>/*/*) and SCENEWX_WEATHERBENCH (<root>/<weather>/test/{input,target}).

  python evaluation/real_paired_psnr.py --opt nafnet_rain_full_moderate_c256_i60000_s10 \
      --ckpt <net_g_60000.pth> --out outputs/eval/rain_full_paired.json
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from restorers import load_restorer, restore, mean_abs, psnr, default_device   # noqa: E402
from scenewx import config                                                     # noqa: E402

MAX_SIDE = 1024          # px, longest side after downscaling


def load(p):
    im = cv2.imread(p)
    if im is None:
        return None
    s = MAX_SIDE / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return im


def gt_rain(per_scene=4):
    """(rainy, clean) pairs: the first `per_scene` rainy frames of every scene folder."""
    for d in sorted(glob.glob(f'{config.get("SCENEWX_GTRAIN")}/*/*')):
        cs = glob.glob(f'{d}/*-C-*.png')
        if not cs:
            continue
        for r in sorted(glob.glob(f'{d}/*-R-*.png'))[:per_scene]:
            yield r, cs[0]


def weatherbench(weather, n=60):
    for p in sorted(glob.glob(f'{config.get("SCENEWX_WEATHERBENCH")}/{weather}/test/input/*'))[:n]:
        t = p.replace('/input/', '/target/')
        if os.path.exists(t):
            yield p, t


def score(net, pairs, dev):
    pi, po, ch = [], [], []
    for ip, tp in pairs:
        x, t = load(ip), load(tp)
        if x is None or t is None or x.shape != t.shape:
            continue
        y = restore(net, x, dev)
        pi.append(psnr(x, t)); po.append(psnr(y, t)); ch.append(mean_abs(y, x))
    return dict(n=len(pi), psnr_in=float(np.mean(pi)), psnr_out=float(np.mean(po)),
                gain_dB=float(np.mean(po) - np.mean(pi)), change=float(np.mean(ch)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', required=True, help='run name, e.g. nafnet_rain_full_moderate_c256_i60000_s10')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    arch, rest = a.opt.split('_', 1)
    weather = rest.split('_')[0]
    dev = torch.device(a.device or default_device())
    net = load_restorer(a.ckpt, dev, width=32, arch=arch)
    res = dict(opt=a.opt, weather=weather)
    if weather == 'rain':
        res['gt_rain'] = score(net, gt_rain(), dev)
    if weather in ('rain', 'snow'):
        res[f'weatherbench_{weather}'] = score(net, weatherbench(weather), dev)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
