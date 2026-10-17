"""Figure 4.1: gain of each condition over the same-run baseline, per weather.

NAFNet filled circles, Restormer open diamonds, a line spans two runs. Values are typed in from Chapter 4.
"""
from thesis_style import *                                                     # noqa: F401,F403

GROUPS = [('Placement', ['Haze by distance', 'Particles by distance', 'Full']),
          ('Depth', ['Corrected depth', 'Scrambled depth']),
          ('Severity', ['Light severity', 'Heavy severity', 'Mixed severity'])]
# gain over the baseline trained in the same run: (NAFNet runs, Restormer runs)
D = [
    ('Fog', 'mIoU', 2.0, 32.42 - 27.55, {
        'Haze by distance': ((2.54, 4.62), ()), 'Full': ((3.19, 4.98), (1.95,)), 'Corrected depth': ((0.92,), ()),
        'Light severity': ((5.31, 6.33), (2.52,)), 'Heavy severity': ((-0.94, -2.27), ()), 'Mixed severity': ((3.38,), ()),
        'Scrambled depth': ((1.15,), (0.13,))}),
    ('Rain', 'dB', 0.4, 19.80 - 20.69, {
        'Haze by distance': ((0.15,), ()), 'Particles by distance': ((0.77, 1.07), (0.70, 0.09)), 'Full': ((0.37, 0.87), (0.41,)),
        'Corrected depth': ((0.27,), ()), 'Light severity': ((0.23,), ()), 'Heavy severity': ((0.80,), ()),
        'Mixed severity': ((0.77,), ()), 'Scrambled depth': ((0.40, 0.31), (0.70, 0.98))}),
    ('Snow', 'mIoU', 3.0, 44.60 - 30.04, {
        'Haze by distance': ((-0.63,), ()), 'Particles by distance': ((10.11, 14.35), (14.53,)), 'Full': ((3.16,), (3.71,)),
        'Corrected depth': ((4.87,), ()), 'Light severity': ((-2.77,), ()), 'Heavy severity': ((5.23,), ()),
        'Mixed severity': ((2.97,), ()), 'Scrambled depth': ((5.80,), (10.01,))}),
]
NAVY, ORANGE = ACCENT, SECOND

# y positions: a heading row, then the group's rows, then a small gap
ys, heads, y = {}, [], 0.0
for g, rows in GROUPS:
    heads.append((g, y)); y -= 0.85
    for r in rows:
        ys[r] = y; y -= 0.8
    y -= 0.25

fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.15), sharey=True)
for k, (ax, (w, unit, thr, unt, v)) in enumerate(zip(axes, D)):
    ax.axvspan(-thr, thr, color='#F0F1F4', lw=0, zorder=0)
    ax.axvline(0, color='#9A9A9A', lw=0.5, zorder=1)
    ax.axvline(unt, color='#555555', lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    for r, yr in ys.items():
        ax.axhline(yr, color='#D9D9D9', lw=0.4, ls=(0, (1, 1.5)), zorder=0)
        if r not in v:
            ax.text(0.98, yr, r'\textit{n/a}', transform=ax.get_yaxis_transform(), ha='right', va='center',
                    color=GREY, fontsize=7)
            continue
        nf, rs = v[r]
        for runs, col in [(nf, NAVY), (rs, ORANGE)]:
            if len(runs) == 2:
                ax.plot(runs, [yr, yr], color=col, lw=0.9, solid_capstyle='butt', zorder=2)
        if rs:
            ax.plot(sum(rs) / len(rs), yr, 'D', ms=3.4, mfc='white', mec=ORANGE, mew=0.9, zorder=4)
        ax.plot(sum(nf) / len(nf), yr, 'o', ms=3.6, color=NAVY, mec='white', mew=0.4, zorder=3)
    xs = [x for t in v.values() for runs in t for x in runs] + [unt, -thr, thr]
    lo, hi = min(xs), max(xs); pad = (hi - lo) * 0.08
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_title(rf'\textbf{{({"abc"[k]})}} {w}', loc='left', pad=4)
    ax.set_xlabel(f'gain ({unit})')
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
axes[0].set_yticks(list(ys.values())); axes[0].set_yticklabels(list(ys.keys()))
for g, yh in heads:
    axes[0].text(-0.035, yh, rf'\textsc{{{g.lower()}}}', transform=axes[0].get_yaxis_transform(),
                 ha='right', va='center', color=INK, fontsize=7.5)
axes[0].set_ylim(min(ys.values()) - 0.55, heads[0][1] + 0.45)

h = [plt.Line2D([], [], marker='o', ms=3.6, ls='-', lw=0.9, color=NAVY, mec='white', mew=0.4, label='NAFNet'),
     plt.Line2D([], [], marker='D', ms=3.4, ls='-', lw=0.9, color=ORANGE, mfc='white', mec=ORANGE, mew=0.9, label='Restormer'),
     plt.Rectangle((0, 0), 1, 1, color='#F0F1F4', label='run-to-run variation'),
     plt.Line2D([], [], ls=(0, (2.5, 2)), lw=0.6, color='#555555', label='untouched input')]
fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.0),
           handlelength=1.8, columnspacing=1.6)
fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=1.0)
save(fig, 'fig4_1_gains')
