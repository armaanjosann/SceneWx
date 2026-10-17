"""Shared style for the thesis figures: LaTeX Computer Modern at the text width (411.4 pt), PDF and PNG.

Needs LaTeX. Paths: SCENEWX_FIG_OUT [outputs/figures], SCENEWX_RENDERS [outputs/fig3_3_renders] and
SCENEWX_FIG_SRC, restored real frames from trained models, not included [outputs/figures/src].
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenewx import config                                                     # noqa: E402

TEXTWIDTH = 411.4 / 72.27          # inches
OUT = os.environ.get('SCENEWX_FIG_OUT') or os.path.join(config.outputs(), 'figures')
SRC = os.environ.get('SCENEWX_FIG_SRC') or os.path.join(OUT, 'src')
RENDERS = os.environ.get('SCENEWX_RENDERS') or os.path.join(config.outputs(), 'fig3_3_renders')
OUT, SRC, RENDERS = (os.path.join(p, '') for p in (OUT, SRC, RENDERS))   # trailing separator

# the frame shown in Figures 3.1-3.3
FRAME = 'weimar_000011_000019'


def frame_paths():
    """(clear image, labelIds, raw depth, corrected depth) of FRAME, from the SCENEWX_* paths."""
    city, split = FRAME.split('_')[0], 'train'
    return (f'{config.cityscapes()}/{split}/{city}/{FRAME}_leftImg8bit.png',
            f'{config.gtfine()}/{split}/{city}/{FRAME}_gtFine_labelIds.png',
            f'{config.depth_cache()}/{split}/{FRAME}.png',
            f'{config.depth_cache()}/repaired/{split}/{FRAME}.png')


INK = '#1A1A1A'
GREY = '#6B6B6B'
RULE = '#B5B5B5'
BAND = '#ECEEF1'
ACCENT = '#1F4E8C'     # accent colour
SECOND = '#C25A1C'     # second series only (Restormer)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.edgecolor': INK, 'axes.linewidth': 0.6, 'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
})


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def letter(ax, s, dy=0.0):
    """Panel label (a), (b) ... above the panel's top-left corner."""
    ax.text(0.0, 1.02 + dy, rf'\textbf{{({s})}}', transform=ax.transAxes, ha='left', va='bottom', color=INK)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(OUT + name + '.pdf')
    fig.savefig(OUT + name + '.png', dpi=300)
    plt.close(fig)
    print('saved', name)
