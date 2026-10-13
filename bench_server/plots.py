"""plots.py <results.tsv> <out_dir>: PR #2501's three figures from plots_bench.sh's results, for PR #2501 and ours.

throughput_feed_train.png, throughput_train.png: MiB/s vs cores (median of the runs, whiskers min-max), one panel per
workload, endpoint labels = speedup vs PR #2501 at 8 cores. memory.png: peak RSS vs input size at 8 cores, labels =
peak RSS as % of PR #2501's at the largest size both finished.
"""
import collections
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
head, rows = rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]
OUT = sys.argv[2]
IMPLS = [("pr2501", "HF PR #2501", "#0072B2", "s"), ("ours", "ours", "#D55E00", "^")]
WORK = ["code/bytelevel", "english/bytelevel", "multilingual/bytelevel", "multilingual/whitespace"]
TITLE = {"code/bytelevel": "Code / ByteLevel", "english/bytelevel": "English / ByteLevel",
         "multilingual/bytelevel": "Multilingual / ByteLevel", "multilingual/whitespace": "Multilingual / Whitespace"}
bad = [r for r in rows if r["merges_vs_pr2501"] == "DIFFERENT"]
print(f"{len(rows)} runs, {len(bad)} with merges different from PR #2501")


def throughput(kind, fname, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, w, letter in zip(axes.flat, WORK, "abcd"):
        ends = []
        for impl, label, color, marker in IMPLS:
            by = collections.defaultdict(list)
            for r in rows:
                if r["figure"] == "throughput" and r["workload"] == w and r["impl"] == impl:
                    s = float(r["wall_s"]) if kind == "all" else float(r["train_s"]) if r["train_s"] != "NA" else None
                    if kind == "all" and r["feed_s"] != "NA" and r["train_s"] != "NA":
                        s = float(r["feed_s"]) + float(r["train_s"])
                    if s:
                        by[int(r["cores"])].append(float(r["mib"]) / s)
            if not by:
                continue
            xs = sorted(by)
            med = [statistics.median(by[x]) for x in xs]
            lo = [med[i] - min(by[x]) for i, x in enumerate(xs)]
            hi = [max(by[x]) - med[i] for i, x in enumerate(xs)]
            ax.errorbar(xs, med, yerr=[lo, hi], color=color, marker=marker, label=label, capsize=2, lw=1.5, ms=5)
            ends.append((impl, xs[-1], med[-1], color))
        base = {i: y for i, x, y, c in ends}.get("pr2501")
        for impl, x, y, color in ends:
            if base:
                ax.annotate(f"{y / base:.1f}×", (x, y), xytext=(8, 0), textcoords="offset points", color=color,
                            fontsize=9, fontweight="bold", va="center")
        ax.set_title(TITLE[w], loc="left", fontsize=11)
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)
        ax.set_xlabel("Cores"); ax.set_ylabel("Throughput (MiB/s)")
        ax.set_xticks([1, 2, 4, 6, 8]); ax.set_xlim(0, 9.5); ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.06, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.06, 0.935, "256 MiB per corpus | Target vocabulary size 100,000", fontsize=9, color="#555555")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.05, 0.92), ncol=2, frameon=False)
    fig.text(0.06, 0.01, "Points: median of three runs; whiskers: min-max. Endpoint labels: speedup vs PR #2501 at the same "
             "core count. Every run pinned to its cores.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(f"{OUT}/{fname}", dpi=130)


def memory():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, w, letter in zip(axes.flat, WORK, "abcd"):
        pts = {}
        for impl, label, color, marker in IMPLS:
            xy = sorted((float(r["mib"]), float(r["peak_rss_KB"]) / 1048576) for r in rows
                        if r["figure"] == "memory" and r["workload"] == w and r["impl"] == impl)
            if xy:
                ax.plot([x for x, _ in xy], [y for _, y in xy], color=color, marker=marker, label=label, lw=1.5, ms=5)
                pts[impl] = dict(xy)
        common = set.intersection(*(set(p) for p in pts.values())) if len(pts) == len(IMPLS) else set()
        if common:
            x = max(common)
            for impl, label, color, marker in IMPLS:
                y = pts[impl][x]
                ax.annotate(f"{100 * y / pts['pr2501'][x]:.0f}%", (x, y), xytext=(8, 0), textcoords="offset points",
                            color=color, fontsize=9, fontweight="bold", va="center")
        ax.set_title(TITLE[w], loc="left", fontsize=11)
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)
        ax.set_xlabel("Input size (MiB)"); ax.set_ylabel("Peak RSS (GiB)"); ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("BPE memory growth", x=0.06, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.06, 0.935, "8 cores | Target vocabulary size 100,000", fontsize=9, color="#555555")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.05, 0.92), ncol=2, frameon=False)
    fig.text(0.06, 0.01, "One run per point. Labels: peak RSS as % of PR #2501 at the largest size both finished.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(f"{OUT}/memory.png", dpi=130)


throughput("all", "throughput_feed_train.png", "BPE throughput: Feed + Train")
throughput("train", "throughput_train.png", "BPE throughput: Train only")
memory()
