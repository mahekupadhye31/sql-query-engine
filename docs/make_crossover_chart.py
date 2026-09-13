#!/usr/bin/env python3
"""Regenerates docs/crossover.png from bench_report.json (produced by
`python3 tools/bench.py --json-out bench_report.json`). Not part of the
engine -- a one-off chart-generation script for the README."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "bench_report.json").read_text())
points = sorted(data["crossover"]["points"], key=lambda p: p["selectivity"])

# One point (cardinality=100000) had zero matching rows -- selectivity 0.0
# exactly, which can't sit on a log axis. It's noted separately; the line
# starts at the next point (selectivity ~0.002%).
plotted = [p for p in points if p["selectivity"] > 0]
sel = [p["selectivity"] * 100 for p in plotted]
scan_ms = [p["scan_p50_ms"] for p in plotted]
seek_ms = [p["seek_p50_ms"] for p in plotted]
scan_reads = [p["scan_page_reads"] for p in plotted]
seek_reads = [p["seek_page_reads"] for p in plotted]

BLUE = "#2a78d6"   # scan
ORANGE = "#eb6834"  # seek
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
})

fig, (ax_ms, ax_reads) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, dpi=200)
fig.suptitle("Full scan vs. index seek, 100,000-row table", fontsize=14, fontweight="bold", color=INK, y=0.97)
fig.text(0.5, 0.925, "p50 query latency and page reads across match selectivity", ha="center",
          fontsize=10, color=INK_SECONDARY)

for ax, scan_y, seek_y, ylabel, crossover_x in [
    (ax_ms, scan_ms, seek_ms, "p50 latency (ms, log)", None),
    (ax_reads, scan_reads, seek_reads, "page reads (log)", None),
]:
    ax.plot(sel, scan_y, color=BLUE, linewidth=2, marker="o", markersize=5, label="full scan")
    ax.plot(sel, seek_y, color=ORANGE, linewidth=2, marker="o", markersize=5, label="index seek")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Page-reads crossover: between selectivity 0.72% (seek still cheaper) and
# 0.99% (seek now costlier) -- the real, I/O-cost crossover.
ax_reads.axvspan(0.724, 0.993, color=INK_MUTED, alpha=0.15, zorder=0)
ax_reads.text(1.35, 30000, "page-reads crossover\n0.72%–0.99%",
              fontsize=8.5, color=INK_SECONDARY, ha="left", va="center")

# Latency crossover: between selectivity 2.02% (seek still faster) and 100%
# (scan wins) -- much later, because per-row decode cost dominates latency.
ax_ms.axvspan(2.018, 100, color=INK_MUTED, alpha=0.08, zorder=0)
ax_ms.text(4.2, 1.6, "latency crossover: 2.02%–100%\n(decode cost dominates)",
           fontsize=8.5, color=INK_SECONDARY, ha="left", va="center")

ax_reads.set_xlabel("selectivity, % of rows matched (log)", fontsize=10)
ax_reads.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}%"))

legend = ax_ms.legend(loc="lower right", frameon=True, fontsize=9.5)
legend.get_frame().set_facecolor(SURFACE)
legend.get_frame().set_edgecolor("none")
legend.get_frame().set_alpha(0.95)
for text in legend.get_texts():
    text.set_color(INK)

fig.text(0.5, 0.01,
          "note: the cardinality=100,000 point matched zero rows (selectivity exactly 0%, unplottable on a "
          "log axis) — its seek cost was 3 page reads vs. scan's 758, even cheaper than the leftmost point shown.",
          ha="center", fontsize=7.5, color=INK_MUTED, wrap=True)

fig.tight_layout(rect=(0, 0.035, 1, 0.92))
out = ROOT / "docs" / "crossover.png"
fig.savefig(out, dpi=200, facecolor=SURFACE)
print(f"wrote {out}")
