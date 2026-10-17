"""Figure 3.1: study design diagram.

Needs: the SceneWx data paths, the renders from fig3_3_renders.py (SCENEWX_RENDERS), one GT-RAIN
scene (SCENEWX_GTRAIN) and restored ACDC fog frames in SCENEWX_FIG_SRC (fog_input.png,
fog_best_seg.png, fog_best.png), which come from trained models and are not included.
"""
import os

import numpy as np
from PIL import Image, ImageFilter
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, FancyArrowPatch
from thesis_style import *                                                     # noqa: F401,F403
from thesis_style import config, frame_paths

CS, _, RAW, _ = frame_paths()
GTR = os.path.join(config.get('SCENEWX_GTRAIN'), 'GT-RAIN_test', 'Winter_Garden_0-1', 'Winter_Garden_0-1-Webcam-')
Y0, Y1, X0, X1 = 40, 424, 200, 776

W, H = 100.0, 52.0                      # drawing units; figure keeps the same aspect
fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')


def img(a, x, y, w, h, **kw):
    sub = fig.add_axes([x / W, y / H, w / W, h / H]); sub.imshow(a, aspect='auto', interpolation='lanczos', **kw)
    sub.set_xticks([]); sub.set_yticks([])
    for s in sub.spines.values():
        s.set_linewidth(0.4); s.set_edgecolor(GREY)


def box(x, y, w, h, lw=0.6, ls='-'):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK, lw=lw, ls=ls))


def arrow(x0, y0, x1, y1, lw=0.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=lw, color=INK,
                                 shrinkA=0, shrinkB=0))


def t(x, y, s, size=8, ha='center', va='center', color=INK):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color)


def load(p, crop=None, size=None):
    a = Image.open(p).convert('RGB')
    if size: a = a.resize(size, Image.LANCZOS)
    a = np.asarray(a)
    return a[crop[0]:crop[1], crop[2]:crop[3]] if crop else a


# column frames and headings
cols = [(0.5, 16.5, r'\textbf{1.\ Clear frames}'), (20, 28.5, r'\textbf{2.\ SceneWx}'),
        (51.5, 14, r'\textbf{3.\ Training}'), (68.5, 31, r'\textbf{4.\ Real evaluation}')]
for x, w, title in cols:
    box(x, 1, w, 45.5)
    t(x + w / 2, 49.2, title, size=8.5)
for x0, x1 in [(17, 20), (48.5, 51.5), (65.5, 68.5)]:
    arrow(x0, 24, x1, 24, lw=0.9)

# 1. clear frames and depth
clear = load(CS, size=(1024, 512))
img(clear, 1.75, 29, 14, 7); t(8.75, 26.6, 'Cityscapes', 7.5); t(8.75, 24.5, r'2{,}968 frames', 6, color=GREY)
d = np.array(Image.open(RAW)).astype(np.float32) / 100
img(np.clip(d, 2, 100), 1.75, 9.5, 14, 7, cmap='magma_r', norm=LogNorm(2, 100))
t(8.75, 6.6, 'estimated depth', 7.5); t(8.75, 4.5, 'Depth Anything V2', 6, color=GREY)

# 2. SceneWx: four properties, then one render per weather
props = [('haze', 'uniform $\\mid$ by distance'), ('particles', 'fixed $\\mid$ by distance'),
         ('depth', 'raw $\\mid$ corrected $\\mid$ flipped'), ('severity', 'light $\\mid$ moderate $\\mid$ heavy $\\mid$ mixed')]
for i, (name, opts) in enumerate(props):
    y = 43.2 - i * 4.6
    t(21.5, y, rf'\textit{{{name}}}', 7.5, ha='left'); t(21.5, y - 2.0, opts, 6, ha='left', color=GREY)
ax.plot([21.5, 47], [25.4, 25.4], color=RULE, lw=0.5)
t(34.25, 23.4, 'one property changed at a time', 6, color=GREY)
for i, w in enumerate(['fog', 'rain', 'snow']):
    a = load(f'{RENDERS}{w}_full_moderate.png', crop=(Y0, Y1, X0, X1))
    img(a, 22 + i * 8.6, 9.6, 7.9, 7.9 / 1.5)
    t(25.95 + i * 8.6, 7.6, w, 7)
t(34.25, 4.2, 'equal severity in every condition', 6, color=GREY)

# 3. training: a stack of training sets into one network
for k in range(3):
    xs, ys = 52.5 + k * 1.0, 33.0 - k * 1.0
    a = load(f'{RENDERS}rain_{["base", "P2", "full"][k]}_moderate.png', crop=(Y0, Y1, X0, X1))
    img(a, xs, ys, 10.0, 6.67)
t(58.5, 27.6, 'one training set', 6.5); t(58.5, 25.7, 'per condition', 6.5)
arrow(58.5, 23.8, 58.5, 19.6, lw=0.7)
box(53.25, 11.5, 10.5, 8, lw=0.7)
t(58.5, 16.7, 'NAFNet', 8); t(58.5, 14.0, 'same recipe', 6, color=GREY)
t(58.5, 7.6, '43 networks', 6.5); t(58.5, 5.5, r'+ 14 Restormer', 6, color=GREY)

# 4. real evaluation: three measures
rows = [(33.5, 'paired PSNR', 'GT-RAIN, WeatherBench'), (19.5, 'restore, then segment', 'ACDC, Foggy Zurich, Foggy Driving'),
        (5.5, 'behaviour', 'how much and where it edits')]
r_in = load(GTR + 'R-000.png', size=(320, 200)); r_cl = load(GTR + 'C-000.png', size=(320, 200))
fog_in = load(SRC + 'fog_input.png'); fog_seg = load(SRC + 'fog_best_seg.png'); fog_best = load(SRC + 'fog_best.png')
gi = np.asarray(Image.fromarray(fog_in).convert('L')).astype(float); gb = np.asarray(Image.fromarray(fog_best).convert('L')).astype(float)
change = np.asarray(Image.fromarray(np.abs(gb - gi).astype('uint8')).filter(ImageFilter.GaussianBlur(3))).astype(float)
pairs = [(r_in, r_cl, None), (fog_in, fog_seg, None), (fog_in, change, 'inferno')]
for (y, name, sub), (a, b, cm) in zip(rows, pairs):
    t(70, y + 10.0, name, 7.5, ha='left'); t(70, y + 8.0, sub, 6, ha='left', color=GREY)
    img(a, 70, y, 12.5, 6.6); img(b, 85.5, y, 12.5, 6.6, cmap=cm)
    arrow(83.0, y + 3.3, 85.2, y + 3.3, lw=0.6)
save(fig, 'fig3_1_study_design')
