"""BasicSR dataset that renders SceneWx training crops on the fly, one (weather, cell, level) per set.

Option keys: weather, cell, level (or 'mixed'), split, gt_size, seed, phase (see configs/ for full files).
Depth maps are built on the whole frame and cropped with the image. Smoke test:

  python -m scenewx.scenewx_dataset --weather snow --cell full --level moderate
"""
import os

import cv2
import numpy as np
import torch
from torch.utils import data

from . import cells as CELLS
from . import compositor as C
from . import config
from . import depth as D
from . import fog as FOG


class SceneWxPairedDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.weather = opt['weather']
        self.cell = opt['cell']
        self.level = opt.get('level', 'moderate')
        self.split = opt.get('split', 'train')
        self.gt_size = int(opt.get('gt_size', 256))
        self.seed = int(opt.get('seed', 0))
        # 1024x512; must match the calibration scale, since severity changes with resolution
        self.scale_frame = float(opt.get('scale_frame', 0.5))
        self.phase = opt.get('phase', 'train')

        allc = CELLS.load_cell_calibration()[self.weather]
        lv = CELLS.LEVELS if self.level == 'mixed' else (self.level,)
        self.cell_cal = {l: allc[l]['cells'] for l in lv}
        for l in lv:
            if self.cell not in self.cell_cal[l]:
                raise ValueError(f'no cell calibration for {self.weather}/{l}/{self.cell}')
        self.params = self.cell_cal[lv[0]][self.cell]
        self.target_severity = allc[lv[0]]['target_severity']

        names = D.frames_for(self.split)
        key = lambda nm: f'{self.split}/{nm}'
        # frozen train frame list (SCENEWX_FRAMES), so every condition uses the same frames
        frozen = config.frames_file() if self.split == 'train' else None
        if frozen and os.path.exists(frozen):
            keep = set(l.strip() for l in open(frozen) if l.strip())
            names = [nm for nm in names if nm in keep]
        # only frames with cached corrected depth, so workers never run depth inference
        self.names = [nm for nm in names if D.has_repaired(key(nm))]
        dropped = len(names) - len(self.names)
        if dropped:
            print(f'[SceneWxPairedDataset] {self.split}: {len(self.names)} frames with cached depth, '
                  f'{dropped} without (skipped)')
        if opt.get('limit'):
            self.names = self.names[:int(opt['limit'])]
        self._frame = {}

    def __len__(self):
        return len(self.names)

    # --- helpers ---

    def _load(self, name):
        """Clear frame, labels, raw and corrected depth for one frame, at `scale_frame`."""
        key = f'{self.split}/{name}'
        bgr, lab, ins = D.load_frame(name, self.split, want_instances=not D.has_repaired(key))
        raw = D.metric_depth(bgr, cache_key=key)
        rep = D.repaired_depth(bgr, lab, ins, cache_key=key)
        if self.scale_frame != 1.0:
            sz = (int(bgr.shape[1] * self.scale_frame), int(bgr.shape[0] * self.scale_frame))
            bgr = cv2.resize(bgr, sz, interpolation=cv2.INTER_AREA)
            lab = cv2.resize(lab, sz, interpolation=cv2.INTER_NEAREST)
            raw = cv2.resize(raw, sz, interpolation=cv2.INTER_NEAREST)
            rep = cv2.resize(rep, sz, interpolation=cv2.INTER_NEAREST)
        return bgr, lab, raw, rep

    def _item_cell(self, name, rng):
        """One crop. Level is fixed, or sampled per item for 'mixed' (uniform over the three levels)."""
        bgr, lab, raw, rep = self._load(name)
        if name not in self._frame:
            self._frame[name] = (FOG.airlight_cached(bgr, lab, f'{self.split}/{name}'),
                                 C.focal_for(bgr.shape[1]))
        a_light, f_px = self._frame[name]
        level = self.level if self.level != 'mixed' else CELLS.LEVELS[int(rng.integers(0, 3))]
        params = self.cell_cal[level][self.cell]
        d_med, d_occ = CELLS.depths_for_cell(self.weather, self.cell, raw, rep, params)
        if d_med is None:                                    # noise / identity controls
            gt, = self._crop([bgr], rng)
            lq = CELLS.render_cell(gt, None, None, a_light, self.weather, self.cell, params,
                                   seed=int(rng.integers(0, 2 ** 31)), f_px=f_px)
            return gt, lq
        gt, dm, do = self._crop([bgr, d_med, d_occ], rng)
        lq = CELLS.render_cell(gt, dm, do, a_light, self.weather, self.cell, params,
                               seed=int(rng.integers(0, 2 ** 31)), f_px=f_px)
        return gt, lq

    def _crop(self, arrays, rng):
        h, w = arrays[0].shape[:2]
        g = self.gt_size
        if self.phase != 'train' or h <= g or w <= g:
            return [a[:g, :g] for a in arrays]
        y, x = rng.integers(0, h - g + 1), rng.integers(0, w - g + 1)
        return [a[y:y + g, x:x + g] for a in arrays]

    # --- item ---

    def __getitem__(self, index):
        name = self.names[index]
        # str hash is salted per process: set PYTHONHASHSEED for identical crops and seeds across runs
        rng = np.random.default_rng(hash((name, self.weather, self.cell, self.level, self.seed)) % (2 ** 32))
        gt, lq = self._item_cell(name, rng)

        if self.phase == 'train' and rng.random() < 0.5:
            gt, lq = gt[:, ::-1].copy(), lq[:, ::-1].copy()

        to_t = lambda x: torch.from_numpy(
            np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0)
        return {'lq': to_t(lq), 'gt': to_t(gt), 'lq_path': name, 'gt_path': name}


def smoke(weather='rain', cell='full', level='moderate', n=4, gt_size=256):
    """Render a few validation items and report speed and severity."""
    import time
    ds = SceneWxPairedDataset(dict(weather=weather, cell=cell, level=level, split='val',
                                   gt_size=gt_size, limit=n))
    print(f'{weather}/{cell}/{level}: {len(ds)} frames, target severity {ds.target_severity:.4f}, '
          f'params {ds.params}')
    t0 = time.perf_counter()
    sev = []
    n = min(n, len(ds))
    for i in range(n):
        it = ds[i]
        g = (it['gt'].numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        l = (it['lq'].numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        sev.append(C.severity(g, l))
    dt = time.perf_counter() - t0
    print(f'  {n} crops in {dt:.2f}s ({n / dt:.1f} crops/s)   crop severity '
          f'mean {np.mean(sev):.4f} (target {ds.target_severity:.4f})')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--weather', default='rain')
    ap.add_argument('--cell', default='full')
    ap.add_argument('--level', default='moderate')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--gt-size', type=int, default=256)
    a = ap.parse_args()
    smoke(a.weather, a.cell, a.level, a.n, a.gt_size)
