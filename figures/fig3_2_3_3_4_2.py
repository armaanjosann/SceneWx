"""Figures 3.2 (depth maps), 3.3 (baseline vs full) and 4.2 (restored real frames) in the thesis style.

Figure 4.2 reads restored ACDC frames from SCENEWX_FIG_SRC ({weather}_{input,base,best}.png and, for
rain, *_nowiper.png); these come from trained models and are not included.
"""
import numpy as np
from PIL import Image
from matplotlib.colors import LogNorm
from thesis_style import *                                                     # noqa: F401,F403
from thesis_style import frame_paths

CS, LAB, RAW, REP = frame_paths()
Y0, Y1, X0, X1 = 40, 424, 200, 776          # crop used for the renders (1024 x 512 frame)


def depth(p):
    return np.array(Image.open(p)).astype(np.float32) / 100.0          # stored in cm


def grid(nr, nc, aspect, top=0.28, gap=0.06, row_gap=0.30, left=0.0):
    """Axes laid out exactly at TEXTWIDTH; returns fig, axes[r][c]."""
    pw = (TEXTWIDTH - left - gap * (nc - 1)) / nc; ph = pw / aspect
    H = top + nr * ph + (nr - 1) * row_gap + 0.02
    fig = plt.figure(figsize=(TEXTWIDTH, H))
    axes = [[fig.add_axes([(left + c * (pw + gap)) / TEXTWIDTH, (H - top - (r + 1) * ph - r * row_gap) / H,
                           pw / TEXTWIDTH, ph / H]) for c in range(nc)] for r in range(nr)]
    return fig, axes


# --- Figure 3.2: depth maps ---
clear = np.asarray(Image.open(CS).convert('RGB').resize((1024, 512), Image.LANCZOS))
lab = np.asarray(Image.open(LAB).resize((1024, 512), Image.NEAREST))
raw, rep = depth(RAW), depth(REP)
raw = np.asarray(Image.fromarray(raw).resize((1024, 512), Image.NEAREST)); rep = np.asarray(Image.fromarray(rep).resize((1024, 512), Image.NEAREST))
flip = raw[::-1].copy()
sky = lab == 23
fig, ax = grid(2, 2, 2.0, top=0.22, row_gap=0.26)
cmap = plt.get_cmap('magma_r').copy(); cmap.set_over('#9DB4CE')
norm = LogNorm(2, 100)
panels = [(clear, 'Clear frame'), (raw, 'Raw depth'), (rep, 'Corrected with scene labels'), (flip, 'Flipped vertically (control)')]
for k, (a, (img, t)) in enumerate(zip([ax[0][0], ax[0][1], ax[1][0], ax[1][1]], panels)):
    if img.ndim == 3:
        a.imshow(img, interpolation='lanczos')
    else:
        h = a.imshow(np.clip(img, 2, 1e6), cmap=cmap, norm=norm, interpolation='nearest')
    bare(a); a.set_title(rf'\textbf{{({"abcd"[k]})}} {t}', loc='left', pad=3)
cax = fig.add_axes([0.25, -0.035, 0.5, 0.022])
cb = fig.colorbar(h, cax=cax, orientation='horizontal', extend='max')
cb.set_ticks([2, 5, 10, 20, 50, 100]); cb.set_ticklabels(['2', '5', '10', '20', '50', '100'])
cb.outline.set_linewidth(0.5); cb.ax.tick_params(length=2)
cb.set_label('distance (m); sky shown in blue', labelpad=2)
save(fig, 'fig3_2_depth')

# --- Figure 3.3: baseline and full ---
fig, ax = grid(3, 2, 1.5, top=0.30, gap=0.05, row_gap=0.05, left=0.22)
for c, (title, sub) in enumerate([('Baseline', 'uniform haze, fixed-size particles'),
                                  ('Full', 'haze and particles change with distance')]):
    ax[0][c].set_title(rf'\textbf{{{title}}}' + '\n' + rf'{{\small {sub}}}', pad=4)
for r, w in enumerate(['fog', 'rain', 'snow']):
    for c, k in enumerate(['base', 'full']):
        im = np.asarray(Image.open(f'{RENDERS}{w}_{k}_moderate.png').convert('RGB'))[Y0:Y1, X0:X1]
        ax[r][c].imshow(im, interpolation='lanczos'); bare(ax[r][c])
    ax[r][0].text(-0.04, 0.5, w.capitalize(), transform=ax[r][0].transAxes, rotation=90, ha='right', va='center')
save(fig, 'fig3_3_conditions')

# --- Figure 4.2: restored real frames ---
fig, ax = grid(3, 3, 797 / 446, top=0.22, gap=0.04, row_gap=0.05, left=0.22)
cols = ['Real input', 'Baseline network', 'Best property network']
best = {'fog': 'haze by distance', 'rain': 'particles by distance', 'snow': 'particles by distance'}
for c, t in enumerate(cols):
    ax[0][c].set_title(t, pad=4)
for r, w in enumerate(['fog', 'rain', 'snow']):
    for c, k in enumerate(['input', 'base', 'best']):
        # rain uses a val frame with no wiper in view (GOPR0572 frame 836)
        f = f'{SRC}{w}_{k}_nowiper.png' if w == 'rain' else f'{SRC}{w}_{k}.png'
        ax[r][c].imshow(np.asarray(Image.open(f).convert('RGB')), interpolation='lanczos'); bare(ax[r][c])
    ax[r][0].text(-0.05, 0.5, f'ACDC {w}', transform=ax[r][0].transAxes, rotation=90, ha='right', va='center')
    ax[r][2].text(0.98, 0.04, rf'\textit{{{best[w]}}}', transform=ax[r][2].transAxes, ha='right', va='bottom',
                  fontsize=7, color='white', bbox=dict(boxstyle='square,pad=0.15', fc='black', ec='none', alpha=0.45))
save(fig, 'fig4_2_restored')
