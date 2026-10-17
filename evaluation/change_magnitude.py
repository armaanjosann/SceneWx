"""How much a restorer changes real frames of its weather and held-out synthetic frames, next to the
change needed on the synthetic side. Also reports synthetic PSNR. Real frames: the first --n-real ACDC
val frames (SCENEWX_ACDC); synthetic: the first --n-syn Cityscapes val frames (needs the val depth cache).

  python evaluation/change_magnitude.py --opt nafnet_fog_full_moderate_c256_i60000_s10 \
      --ckpt <net_g_60000.pth> --out outputs/eval/fog_full_change.json
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from restorers import load_restorer, restore, mean_abs, psnr, default_device   # noqa: E402
from scenewx import cells as CELLS, config                                     # noqa: E402


def real_glob(weather):
    """ACDC val frames of a weather."""
    return f'{config.get("SCENEWX_ACDC")}/rgb_anon/{weather}/val/*/*_rgb_anon.png'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', required=True, help='run name, e.g. nafnet_fog_full_moderate_c256_i60000_s10')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-real', type=int, default=20)
    ap.add_argument('--n-syn', type=int, default=20)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    from scenewx import scenewx_dataset as SD
    pc = CELLS.parse_opt(a.opt)
    if pc is None:
        raise SystemExit(f'not a run name of the form <arch>_<weather>_<cell>_<level>_c<crop>_i<iters>_s<seed>: {a.opt}')
    weather, cell, level = pc['weather'], pc['cell'], pc['level']
    dev = torch.device(a.device or default_device())
    net = load_restorer(a.ckpt, dev, width=32, arch=pc['arch'])

    reals = sorted(glob.glob(real_glob(weather)))[:a.n_real]
    ch_real = [mean_abs(restore(net, cv2.imread(p), dev), cv2.imread(p)) for p in reals]

    res = dict(opt=a.opt, weather=weather, cell=cell, level=level,
               real_set=f'acdc_{weather}', n_real=len(reals), real_change=float(np.mean(ch_real)))
    # held-out synthetic: val split, never trained on
    lv = level if level != 'mixed' else 'moderate'
    ds = SD.SceneWxPairedDataset(dict(weather=weather, cell=cell, level=lv, split='val',
                                      gt_size=512, phase='val', limit=a.n_syn, seed=999))
    need, got, resid, p_in, p_out = [], [], [], [], []
    to8 = lambda t: (t.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).round().astype(np.uint8)
    for i in range(len(ds)):
        it = ds[i]
        lq, gt = to8(it['lq']), to8(it['gt'])
        out = restore(net, lq, dev)
        need.append(mean_abs(gt, lq)); got.append(mean_abs(out, lq)); resid.append(mean_abs(out, gt))
        p_in.append(psnr(lq, gt)); p_out.append(psnr(out, gt))
    res.update(n_syn=len(ds), syn_needed_change=float(np.mean(need)), syn_change=float(np.mean(got)),
               syn_residual=float(np.mean(resid)), syn_psnr_in=float(np.mean(p_in)),
               syn_psnr_out=float(np.mean(p_out)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
