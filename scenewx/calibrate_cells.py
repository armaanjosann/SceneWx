"""Severity-match every condition to its level's target, one knob per (weather, level, condition).

Fog solves beta or optical depth to match 'full' at Sakaridis's beta; rain and snow solve strength onto
the real targets; the noise control solves sigma. Writes outputs/calibration_cells.json (use it via
SCENEWX_CALIBRATION); the frozen copy is calibration/calibration_cells.json (12 Frankfurt val frames).

  python -m scenewx.calibrate_cells --sample 12
"""
import argparse
import json
import os

import numpy as np

from . import calibrate as CAL          # load_sample, mean_severity, REAL_TARGETS, SAMPLE_SCALE
from . import cells as CELLS
from . import compositor as C
from . import config
from . import fog as FOG


def _render_fn(weather, cell, params):
    def f(bgr, lab, raw, rep, i):
        return CELLS.render_cell_frame(bgr, lab, raw, rep, weather, cell, params, seed=i)
    return f


def sev(sample, weather, cell, params):
    return CAL.mean_severity(sample, _render_fn(weather, cell, params))


def calibrate_weather(sample, weather, level):
    res = {}
    if weather == 'fog':
        beta = FOG.BETAS[level]
        target = sev(sample, 'fog', 'full', dict(beta=beta))
        for cell in CELLS.cells_for('fog'):
            spec = CELLS.CELLS[cell]
            if cell == 'full':
                res[cell] = dict(beta=beta)
            elif spec['control'] == 'noise':
                v = C.solve_for_severity(lambda s_: sev(sample, 'fog', 'noise', dict(sigma=s_)),
                                         target, lo=1e-3, hi=1.0)
                res[cell] = dict(sigma=v)
            elif spec['control'] == 'identity':
                res[cell] = dict()
            elif spec['medium'] == 'depth':                       # P4, scrambled (P1 == full for fog)
                v = C.solve_for_severity(lambda b: sev(sample, 'fog', cell, dict(beta=b)),
                                         target, lo=beta / 40, hi=beta * 40)
                res[cell] = dict(beta=v)
            else:                                                 # base: flat veil, optical depth
                od = C.solve_for_severity(
                    lambda o: sev(sample, 'fog', cell, dict(beta=beta, optical_depth=o)),
                    target, lo=1e-3, hi=20.0)
                res[cell] = dict(beta=beta, optical_depth=od, t=float(np.exp(-od)))
        source = f'full cell at Sakaridis beta {beta} (MOR {2.996 / beta:.0f} m)'
        return target, res, source

    mod = CELLS.MODULES[weather]
    s0 = mod.LEVELS[level]
    target = CAL.REAL_TARGETS[weather][level]
    for cell in CELLS.cells_for(weather):
        spec = CELLS.CELLS[cell]
        if spec['control'] == 'identity':
            res[cell] = dict()
        elif spec['control'] == 'noise':
            v = C.solve_for_severity(lambda s_: sev(sample, weather, 'noise', dict(sigma=s_)),
                                     target, lo=1e-3, hi=1.0)
            res[cell] = dict(sigma=v)
        else:
            v = C.solve_for_severity(lambda ss: sev(sample, weather, cell, dict(strength=ss)),
                                     target, lo=s0 / 60, hi=s0 * 60)
            res[cell] = dict(strength=v)
    source = {'rain': 'GT-RAIN (real)', 'snow': 'WeatherBench snow (real, sprayed)'}[weather]
    return target, res, source


def run(weathers, levels, sample_n, out_path=None):
    out_path = out_path or os.path.join(config.outputs(), 'calibration_cells.json')
    sample = CAL.load_sample(sample_n)
    print(f'cell calibration: {list(weathers)} x {list(levels)} on {len(sample)} frames at {CAL.SAMPLE_SCALE}x\n', flush=True)
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out['_sample_frames'] = [s[0] for s in sample]
    out['_sample_scale'] = CAL.SAMPLE_SCALE
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    for w in weathers:
        out.setdefault(w, {})
        for level in levels:
            t, res, src = calibrate_weather(sample, w, level)
            out[w][level] = dict(target_severity=t, target_source=src, cells=res)
            print(f'{w:5s} {level:9s} target {t:.4f}  <- {src}')
            for cell, p in res.items():
                achieved = sev(sample, w, cell, p) if p or cell == 'identity' else float('nan')
                kv = '  '.join(f'{k}={v:.4f}' for k, v in p.items())
                print(f'            {cell:10s} {kv:45s} achieved {achieved:.4f}')
            print(flush=True)
            json.dump(out, open(out_path, 'w'), indent=1)       # save after every level
    print('wrote', out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weather', action='append', default=None)
    ap.add_argument('--level', action='append', default=None)
    ap.add_argument('--sample', type=int, default=12)
    ap.add_argument('--out', default=None, help='output JSON (default: outputs/calibration_cells.json)')
    a = ap.parse_args()
    run(tuple(a.weather or ['fog', 'snow', 'rain']), tuple(a.level or CELLS.LEVELS), a.sample, a.out)
