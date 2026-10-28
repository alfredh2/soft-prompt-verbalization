"""Plot exact rule recovery against the number of examples shown.

Two panels on one shared accuracy axis. Left: the curve itself, one line per model,
with support consistency as a recessive reference line -- the models always explain the
examples they were shown, so the gap between that flat line and the rising ones is the
result. Right: the same model scored three ways, showing how much the random held-out
metric flatters a reconstruction compared with full-domain exactness.

    python scripts/plot_learning_curve.py \
        --curve gpt-5-mini=results/learning_curve/gpt5mini_v2/curve.json \
        --curve gpt-6.1-sol=results/learning_curve/gpt61sol/curve.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots 1 and 2 (light / dark steps of the same hues).
THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e4e3df", "series": ["#2a78d6", "#eb6834"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "grid": "#333330", "series": ["#3987e5", "#d95926"]},
}


def load(spec: str) -> tuple[str, dict]:
    label, _, path = spec.partition("=")
    data = json.loads(Path(path).read_text())
    return label or data.get("model", path), data["curve"]


def series(curve: dict, key: str) -> tuple[list[int], list[float]]:
    xs = sorted(int(k) for k in curve)
    pts = [(x, curve[str(x)][key]) for x in xs if curve[str(x)].get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def style_axes(ax, t: dict, xlabel: str, ylabel: str | None, title: str) -> None:
    ax.set_facecolor(t["surface"])
    ax.set_title(title, color=t["ink"], fontsize=11, pad=10, loc="left")
    ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
    ax.set_ylim(-0.03, 1.06)
    ax.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)


def render(curves: list[tuple[str, dict]], mode: str, out: Path) -> None:
    t = THEME[mode]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=t["surface"])

    # -- left: exact recovery per model -------------------------------------------
    drawn = []
    for i, (label, curve) in enumerate(curves):
        xs, ys = series(curve, "exact_recovery")
        c = t["series"][i % len(t["series"])]
        ax1.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=7,
                 markeredgecolor=t["surface"], markeredgewidth=2, label=label, zorder=3)
        drawn.append((label, xs, ys, c))

    # Label where the series are furthest apart, not at the end: these curves converge
    # (both finish at 0.684) and end labels collide illegibly.
    if len(drawn) == 2:
        (_, xa, ya, _), (_, xb, yb, _) = drawn
        common = [x for x in xa if x in xb]
        j = max(common, key=lambda x: abs(ya[xa.index(x)] - yb[xb.index(x)]))
        for label, xs, ys, c in drawn:
            y = ys[xs.index(j)]
            above = y >= (ya[xa.index(j)] + yb[xb.index(j)]) / 2
            ax1.annotate(label, (j, y), textcoords="offset points",
                         xytext=(-6, 11 if above else -17), color=c, fontsize=9,
                         ha="right", zorder=4)
    else:
        for label, xs, ys, c in drawn:
            ax1.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                         xytext=(8, 0), color=c, fontsize=9, va="center", zorder=4)

    sup_x, sup_y = series(curves[0][1], "support")
    ax1.plot(sup_x, sup_y, color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax1.annotate("support consistency = 1.00  (both models, every tier)",
                 (sup_x[0], 1.0), textcoords="offset points", xytext=(4, 7),
                 color=t["ink2"], fontsize=8.5, zorder=4)

    style_axes(ax1, t, "examples shown", "fraction of functions", "Exact rule recovery")
    leg = ax1.legend(frameon=False, fontsize=9, loc="lower right")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    # -- right: one model, three metrics ------------------------------------------
    label, curve = curves[-1]
    styles = [("support", "support (fits what it saw)", (0, (4, 3)), t["ink2"]),
              ("held_out", "random held-out accuracy", (0, (1, 2)), t["series"][0]),
              ("exact_recovery", "exact rule recovery", "solid", t["series"][1])]
    for key, name, ls, c in styles:
        xs, ys = series(curve, key)
        ax2.plot(xs, ys, color=c, linewidth=2 if ls == "solid" else 1.6,
                 linestyle=ls, marker="o" if ls == "solid" else None, markersize=7,
                 markeredgecolor=t["surface"], markeredgewidth=2, label=name, zorder=3)
    style_axes(ax2, t, "examples shown", None, f"Same model, three metrics — {label}")
    leg2 = ax2.legend(frameon=False, fontsize=9, loc="lower right")
    for txt in leg2.get_texts():
        txt.set_color(t["ink2"])

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", action="append", required=True,
                    help="label=path/to/curve.json (repeatable)")
    ap.add_argument("--out-dir", default="results/learning_curve")
    args = ap.parse_args()
    curves = [load(c) for c in args.curve]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "_dark"
        render(curves, mode, out_dir / f"learning_curve{suffix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
