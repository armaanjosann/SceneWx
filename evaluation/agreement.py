"""Does a restorer act on real weather the way it acts on its own synthetic weather? Run on a checkpoint.

Reports the change ratio mean|R(x)-x| / mean|y-x| on real pairs (GT-RAIN, WeatherBench), the ratio of
change on ACDC to change on synthetic val frames, and the Spearman correlation of the change map with
metric depth (sky excluded where ACDC labels exist), with the same statistic on synthetic frames.

  python evaluation/agreement.py --opt nafnet_fog_full_moderate_c256_i60000_s10 \
      --ckpt <net_g_60000.pth> --out outputs/eval/fog_full_agreement.json
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

from restorers import load_restorer, restore, mean_abs, default_device        # noqa: E402
from change_magnitude import real_glob                                         # noqa: E402
import real_paired_psnr as RP                                                  # noqa: E402
from scenewx import cells as CELLS, config, depth as D                         # noqa: E402

SKY = 23
STAT_W = 480          # px, width of the downsampled maps used for the Spearman


def change_map(out, inp):
    return np.abs(out.astype(np.float32) - inp.astype(np.float32)).mean(axis=2)


def small(a, interp=cv2.INTER_AREA):
    h = int(round(a.shape[0] * STAT_W / a.shape[1]))
    return cv2.resize(a, (STAT_W, h), interpolation=interp)


def rho(delta, depth, mask=None):
    """Spearman between a change map and depth, optionally inside a mask (True = keep)."""
    d, z = small(delta), small(depth)
    if mask is not None:
        m = small(mask.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
        d, z = d[m], z[m]
    if d.size < 100 or np.std(d) == 0 or np.std(z) == 0:
        return float('nan')
    return float(spearmanr(d.ravel(), z.ravel()).correlation)


def acdc_label(path):
    """ACDC gt labelIds next to an rgb_anon frame, if the split has labels."""
    p = path.replace('/rgb_anon/', '/gt/').replace('_rgb_anon.png', '_gt_labelIds.png')
    return cv2.imread(p, cv2.IMREAD_UNCHANGED) if os.path.exists(p) else None


def real_spatial(net, weather, n, dev):
    frames = sorted(glob.glob(real_glob(weather)))[:n]
    os.makedirs(os.path.join(config.depth_cache(), 'real'), exist_ok=True)
    rows = []
    for p in frames:
        x = cv2.imread(p)
        y = restore(net, x, dev)
        delta = change_map(y, x)
        stem = os.path.basename(p).replace('_rgb_anon.png', '').replace('.png', '').replace('.jpg', '')
        d = D.metric_depth(x, cache_key=f'real/{weather}_{stem}')
        lab = acdc_label(p)
        r = dict(frame=stem, change=float(delta.mean()), rho_all=rho(delta, d))
        if lab is not None and lab.shape[:2] == d.shape[:2]:
            r['rho_nosky'] = rho(delta, d, mask=(lab != SKY))
            r['change_nosky'] = float(delta[lab != SKY].mean())
        rows.append(r)
    agg = dict(n_real=len(rows), real_change=float(np.mean([r['change'] for r in rows])))
    for k in ('rho_all', 'rho_nosky'):
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
        if v.size:
            agg[k] = float(v.mean()); agg[k + '_std'] = float(v.std()); agg[k + '_frac_pos'] = float((v > 0).mean())
    return agg, rows


def synthetic_side(net, weather, cell, level, n, dev):
    """Held-out synthetic val frames (whole 1024x512 frame): needed and model change, and their
    Spearman with the raw depth."""
    from scenewx import scenewx_dataset as SD
    lv = level if level != 'mixed' else 'moderate'
    ds = SD.SceneWxPairedDataset(dict(weather=weather, cell=cell, level=lv, split='val', gt_size=1024,
                                      phase='val', limit=n, seed=999))
    need, got, rn, rm, rn_ns, rm_ns = [], [], [], [], [], []
    to8 = lambda t: (t.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).round().astype(np.uint8)
    for i in range(len(ds)):
        it = ds[i]
        lq, gt = to8(it['lq']), to8(it['gt'])
        out = restore(net, lq, dev)
        _, lab, raw, _ = ds._load(it['lq_path'])
        raw, lab = raw[:lq.shape[0], :lq.shape[1]], lab[:lq.shape[0], :lq.shape[1]]
        dn, dm = change_map(gt, lq), change_map(out, lq)
        need.append(dn.mean()); got.append(dm.mean())
        rn.append(rho(dn, raw)); rm.append(rho(dm, raw))
        rn_ns.append(rho(dn, raw, lab != SKY)); rm_ns.append(rho(dm, raw, lab != SKY))
    f = lambda v: float(np.nanmean(v)) if len(v) else float('nan')
    return dict(n_syn=len(ds), syn_needed_change=f(need), syn_change=f(got),
                syn_rho_needed=f(rn), syn_rho_model=f(rm), syn_rho_needed_nosky=f(rn_ns), syn_rho_model_nosky=f(rm_ns))


def paired_magnitude(net, pairs, dev, tag=''):
    """Change ratio and the Spearman with depth on real pairs, for the model's change and the needed
    change |target - input|."""
    got, need, rn, rm = [], [], [], []
    for ip, tp in pairs:
        x, t = RP.load(ip), RP.load(tp)
        if x is None or t is None or x.shape != t.shape:
            continue
        y = restore(net, x, dev)
        got.append(mean_abs(y, x)); need.append(mean_abs(t, x))
        stem = f'{tag}_{os.path.splitext(os.path.basename(ip))[0]}'
        d = D.metric_depth(x, cache_key=f'real/{stem}')
        rn.append(rho(change_map(t, x), d)); rm.append(rho(change_map(y, x), d))
    f = lambda v: float(np.nanmean(v)) if len(v) else float('nan')
    return dict(n=len(got), change=float(np.mean(got)), needed=float(np.mean(need)),
                ratio=float(np.mean(got) / max(np.mean(need), 1e-6)),
                rho_needed_real=f(rn), rho_model_real=f(rm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--device', default=None)
    ap.add_argument('--n-real', type=int, default=40); ap.add_argument('--n-syn', type=int, default=20)
    ap.add_argument('--no-paired', action='store_true')
    a = ap.parse_args()
    pc = CELLS.parse_opt(a.opt)
    if pc is None:
        sys.exit(f'not a run name of the form <arch>_<weather>_<cell>_<level>_c<crop>_i<iters>_s<seed>: {a.opt}')
    dev = torch.device(a.device or default_device())
    net = load_restorer(a.ckpt, dev, width=32, arch=pc['arch'])
    weather, cell, level = pc['weather'], pc['cell'], pc['level']
    res = dict(opt=a.opt, weather=weather, cell=cell, level=level, real_set=f'acdc_{weather}',
               depth_model=D.MODEL_ID, stat_width=STAT_W)
    agg, rows = real_spatial(net, weather, a.n_real, dev)
    res.update(agg); res['real_frames'] = rows
    res.update(synthetic_side(net, weather, cell, level, a.n_syn, dev))
    res['unpaired_ratio'] = res['real_change'] / max(res['syn_change'], 1e-6)
    if not a.no_paired:
        if weather == 'rain':
            res['gt_rain'] = paired_magnitude(net, RP.gt_rain(), dev, 'gtrain')
        if weather in ('rain', 'snow'):
            res[f'weatherbench_{weather}'] = paired_magnitude(net, RP.weatherbench(weather), dev, f'wb{weather}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'real_frames'}))


if __name__ == '__main__':
    main()
