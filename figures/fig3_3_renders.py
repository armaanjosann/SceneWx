"""Renders used by Figures 3.1 and 3.3: one Cityscapes training frame under the reference conditions,
seed 3, frozen calibration, 1024x512.

  python figures/fig3_3_renders.py --out outputs/fig3_3_renders

Writes {weather}_{cell}_{level}.png. The figure scripts read this folder via SCENEWX_RENDERS.
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenewx import cells as CELLS, compositor as C, config, fog as FOG          # noqa: E402
from scenewx import scenewx_dataset as SD                                        # noqa: E402

FRAME = 'weimar_000011_000019'
JOBS = [('fog', c, 'moderate') for c in ['base', 'full', 'P4', 'scrambled']] + \
       [('fog', 'full', l) for l in ['light', 'heavy']] + \
       [(w, c, 'moderate') for w in ['rain', 'snow'] for c in ['base', 'P2', 'full']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(config.outputs(), 'fig3_3_renders'))
    ap.add_argument('--frame', default=FRAME)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds = SD.SceneWxPairedDataset(dict(weather='fog', cell='base', level='moderate', split='train',
                                      gt_size=256, phase='val'))
    bgr, lab, raw, rep = ds._load(a.frame)
    a_light = FOG.airlight(bgr, lab)
    f_px = C.focal_for(bgr.shape[1])
    calib = CELLS.load_cell_calibration()
    for w, c, lvl in JOBS:
        p = calib[w][lvl]['cells'][c]
        d_med, d_occ = CELLS.depths_for_cell(w, c, raw, rep, p)
        img = CELLS.render_cell(bgr, d_med, d_occ, a_light, w, c, p, seed=3, f_px=f_px)
        cv2.imwrite(f'{a.out}/{w}_{c}_{lvl}.png', img)
        print('wrote', w, c, lvl, flush=True)


if __name__ == '__main__':
    main()
