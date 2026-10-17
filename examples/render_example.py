"""Render one Cityscapes frame under SceneWx conditions at 1024x512 with the frozen calibration.

  python examples/render_example.py --frame weimar_000011_000019 --split train
  python examples/render_example.py --frame weimar_000011_000019 --split train --all --levels light moderate heavy

Writes clean.png and {weather}_{cell}_{level}.png (base and full, or every condition with --all). Needs
SCENEWX_CITYSCAPES, SCENEWX_GTFINE, SCENEWX_DEPTH_CACHE and, for snow, SCENEWX_CSD_MASKS.
Uncached depth is computed with Depth Anything V2 (needs the gtFine instanceIds).
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenewx import cells as CELLS, compositor as C, config, depth as D, fog as FOG   # noqa: E402

SCALE = 0.5          # 1024x512, the calibration scale


def load(name, split):
    """Frame, labels, raw and corrected depth at 1024x512, as in the training loader."""
    key = f'{split}/{name}'
    bgr, lab, ins = D.load_frame(name, split, want_instances=not D.has_repaired(key))
    if bgr is None or lab is None:
        raise SystemExit(f'could not read {name} from SCENEWX_CITYSCAPES / SCENEWX_GTFINE ({split} split)')
    raw = D.metric_depth(bgr, cache_key=key)
    rep = D.repaired_depth(bgr, lab, ins, cache_key=key)
    sz = (int(bgr.shape[1] * SCALE), int(bgr.shape[0] * SCALE))
    bgr = cv2.resize(bgr, sz, interpolation=cv2.INTER_AREA)
    lab = cv2.resize(lab, sz, interpolation=cv2.INTER_NEAREST)
    raw = cv2.resize(raw, sz, interpolation=cv2.INTER_NEAREST)
    rep = cv2.resize(rep, sz, interpolation=cv2.INTER_NEAREST)
    return bgr, lab, raw, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', default='weimar_000011_000019')
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', default=os.path.join(config.outputs(), 'render_example'))
    ap.add_argument('--weathers', nargs='+', default=['fog', 'rain', 'snow'])
    ap.add_argument('--levels', nargs='+', default=['moderate'], choices=list(CELLS.LEVELS))
    ap.add_argument('--all', action='store_true', help='every condition, not just base and full')
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    bgr, lab, raw, rep = load(a.frame, a.split)
    cv2.imwrite(f'{a.out}/clean.png', bgr)
    a_light = FOG.airlight(bgr, lab)
    f_px = C.focal_for(bgr.shape[1])
    calib = CELLS.load_cell_calibration()
    for w in a.weathers:
        for lvl in a.levels:
            for c in (CELLS.cells_for(w) if a.all else ['base', 'full']):
                p = calib[w][lvl]['cells'][c]
                d_med, d_occ = CELLS.depths_for_cell(w, c, raw, rep, p)
                img = CELLS.render_cell(bgr, d_med, d_occ, a_light, w, c, p, seed=a.seed, f_px=f_px)
                sev = C.severity(bgr, img, mask=(lab != D.SKY_ID))
                cv2.imwrite(f'{a.out}/{w}_{c}_{lvl}.png', img)
                print(f'{w:5s} {c:10s} {lvl:9s} severity {sev:.4f} (target '
                      f'{calib[w][lvl]["target_severity"]:.4f}, calibration-sample mean)')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
