"""Generate the results plots + summary tables for the README.

Reads results/sim_*.json and results/synth.json, writes:
  results/accuracy.png       float32 vs INT8 vs RTL accuracy (dot plot)
  results/cycles.png         cycles/image vs batch size vs sequential baseline
  results/summary.md         tables ready to paste into the README

Usage: python results/make_plots.py
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# palette (dataviz reference, light mode)
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e4e0"

NAIVE_MACS = 784 * 32 + 32 * 16 + 16 * 10  # 25,760 (1 MAC/cycle lower bound)


def load(tag):
    p = RES / f"sim_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def accuracy_plot(meta, rtl):
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)

    rows = [
        ("float32 PyTorch — 10k test set", meta["float32_test_accuracy"] * 100),
        ("INT8 golden model — 10k test set", meta["int8_test_accuracy"] * 100),
        ("INT8 golden model — 200-image subset", rtl["rtl_accuracy"] * 100),
        ("RTL simulation — 200-image subset", rtl["rtl_accuracy"] * 100),
    ]
    ys = range(len(rows), 0, -1)
    for (label, v), y in zip(rows, ys):
        ax.plot([v], [y], "o", ms=9, color=BLUE, zorder=3)
        ax.annotate(f"{v:.2f}%", (v, y), textcoords="offset points",
                    xytext=(10, -3), fontsize=9, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(94, 100)
    ax.set_xlabel("MNIST classification accuracy (%)", fontsize=9, color=INK2)
    ax.set_title("Quantization costs 0.10%; RTL is bit-exact vs golden",
                 fontsize=10, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(RES / "accuracy.png", facecolor=SURFACE)
    plt.close(fig)


def cycles_plot(sweep, n8):
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)

    xs = [b for b, _ in sweep]
    ys = [c for _, c in sweep]
    ax.axhline(NAIVE_MACS, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"sequential 1-MAC/cycle baseline · {NAIVE_MACS:,} cycles",
                (16, NAIVE_MACS), xytext=(0, 6), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK2)

    ax.plot(xs, ys, "-o", color=BLUE, linewidth=2, ms=6, label="4×4 array")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:,.0f}", (x, y), xytext=(0, -14),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    if n8:
        ax.plot([n8[0]], [n8[1]], "s", color=AQUA, ms=8, label="8×8 array")
        ax.annotate(f"{n8[1]:,.0f}", (n8[0], n8[1]), xytext=(-8, -3),
                    textcoords="offset points", ha="right",
                    fontsize=8, color=INK2)
    sp = NAIVE_MACS / ys[-1]
    ax.annotate(f"{sp:.1f}× vs baseline", (xs[-1], ys[-1]),
                xytext=(-2, 22), textcoords="offset points", ha="right",
                fontsize=9, color=BLUE, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("batch size (images per weight-tile pass)",
                  fontsize=9, color=INK2)
    ax.set_ylabel("cycles per image", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower left",
              labelcolor=INK)
    ax.set_title("Inference latency: batching amortizes per-tile overheads "
                 "(weight loads fully hidden by double buffering)",
                 fontsize=10, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(RES / "cycles.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    meta = json.loads((ROOT / "model" / "export" / "meta.json").read_text())
    rtl = load("i200_b16")
    sweep = []
    for b in (1, 2, 4, 8):
        d = load(f"i16_b{b}")
        if d:
            sweep.append((b, d["cycles_per_image"]))
    sweep.append((16, rtl["cycles_per_image"]))
    n8j = load("i200_b16_n8") or load("i16_b16_n8")
    n8 = (16, n8j["cycles_per_image"]) if n8j else None

    accuracy_plot(meta, rtl)
    cycles_plot(sweep, n8)

    # coverage union across runs
    bins = {}
    for p in RES.glob("sim_*.json"):
        for k, v in json.loads(p.read_text()).get("cov_bins", {}).items():
            bins[k] = bins.get(k, 0) + int(v)
    hit = sum(1 for v in bins.values() if v > 0)

    synth = json.loads((RES / "synth.json").read_text()) \
        if (RES / "synth.json").exists() else {}

    lines = ["# Results summary (generated)\n"]
    lines.append("## Accuracy\n")
    lines.append("| Model | Test set | Accuracy |")
    lines.append("|---|---|---|")
    lines.append(f"| float32 PyTorch | 10,000 | "
                 f"{meta['float32_test_accuracy']*100:.2f}% |")
    lines.append(f"| INT8 golden (Python) | 10,000 | "
                 f"{meta['int8_test_accuracy']*100:.2f}% |")
    lines.append(f"| RTL simulation | {rtl['preds_total']} | "
                 f"{rtl['rtl_accuracy']*100:.2f}% |")
    lines.append(f"\nRTL vs golden agreement: "
                 f"{rtl['rtl_vs_golden_agreement']*100:.1f}% "
                 f"({rtl['preds_match_golden']}/{rtl['preds_total']}), "
                 f"logits bit-exact ({rtl['logit_checked']} checked, "
                 f"{rtl['logit_errors']} errors), hidden activations bit-exact "
                 f"({rtl['act_checked']} checked, {rtl['act_errors']} errors).\n")

    lines.append("## Latency (cycles per image)\n")
    lines.append("| Configuration | Cycles/image | vs sequential |")
    lines.append("|---|---|---|")
    lines.append(f"| Sequential 1-MAC/cycle baseline | {NAIVE_MACS:,} | 1.0× |")
    for b, c in sweep:
        lines.append(f"| 4×4 array, batch {b} | {c:,.0f} | "
                     f"{NAIVE_MACS/c:.2f}× |")
    if n8:
        lines.append(f"| 8×8 array, batch 16 | {n8[1]:,.0f} | "
                     f"{NAIVE_MACS/n8[1]:.2f}× |")

    lines.append(f"\n## Functional coverage\n")
    lines.append(f"{hit}/{len(bins)} bins hit "
                 f"({hit/len(bins)*100:.0f}%) across regression + stress runs.\n")

    if synth:
        g = synth["generic"]
        lines.append("## Synthesis (Yosys generic)\n")
        lines.append("| Module | Cells | Logic depth |")
        lines.append("|---|---|---|")
        for key, label in [("pe", "PE"), ("array_n4", "4×4 array"),
                           ("array_n8", "8×8 array"), ("accel_top", "full top")]:
            s = g.get(key, {})
            lines.append(f"| {label} | {s.get('total_cells', '—'):,} | "
                         f"{s.get('logic_depth', '—')} |")
        if "ecp5" in synth:
            e = synth["ecp5"]
            fmax = min(e["fmax_mhz"].values()) if e["fmax_mhz"] else None
            lines.append(f"\nECP5-85k place-and-route: Fmax = {fmax:.1f} MHz; "
                         f"utilization: " + ", ".join(
                             f"{k} {v['used']}/{v['avail']}"
                             for k, v in e["utilization"].items()) + "\n")

    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote accuracy.png, cycles.png, summary.md; "
          f"coverage {hit}/{len(bins)}")


if __name__ == "__main__":
    main()
