"""yk_plots.py <out_dir> <arm> ...: PR #2501's three figures and tables, in its own setting, for the given arms.

Arms: pr2501 (from results_yikai/pr2501_*.csv, his harness), ours (our counting + PR #2501, results_yikai/ours*.tsv),
small (the small trainer, results_yikai/small_*.csv, his harness). The first arm is the reference of the endpoint
labels. Writes core-scaling.png (feed + train), train-core-scaling.png, memory-growth.png and tables.md.
"""
import collections
import csv
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = "results_yikai"
OUT, ARMS = sys.argv[1], sys.argv[2:]
CASES = ["code-bytelevel", "en-bytelevel", "zh-bytelevel", "zh-whitespace"]
TITLE = {"code-bytelevel": "Code / ByteLevel", "en-bytelevel": "English / ByteLevel",
         "zh-bytelevel": "Chinese / ByteLevel", "zh-whitespace": "Chinese / HF Whitespace"}
STYLE = {"main": ("HF main", "#555555", "o"), "pr2501": ("PR #2501", "#0072B2", "s"),
         "ours": ("PR #2501 + toki counting", "#D55E00", "^"), "small": ("small trainer", "#009E73", "D")}
GiB = 2 ** 30


def suite(matrix, growth, arm):
    """(runs: (case, workers) -> [(feed, train, rss GiB)], growth: case -> [(MiB, rss GiB)]) from his harness."""
    runs = collections.defaultdict(list)
    for r in csv.DictReader(open(matrix)):
        if r["arm"] == arm:
            # medians only in matrix.csv: one entry per cell
            runs[(r["case"], int(r["workers"]))].append(
                (float(r["median_feed_seconds"]), float(r["median_train_seconds"]), float(r["median_peak_rss_bytes"]) / GiB))
    grow = collections.defaultdict(dict)
    for r in csv.DictReader(open(growth)):
        if r["arm"] == arm and r["run_status"] == "ok":
            grow[r["case"]][int(r["requested_mib"])] = int(r["peak_rss_bytes"]) / GiB
    return runs, {c: sorted(v.items()) for c, v in grow.items()}


def ours():
    runs = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{D}/ours.tsv"), delimiter="\t"):
        runs[(r["case"], int(r["workers"]))].append((float(r["feed_s"]), float(r["train_s"]), int(r["peak_rss_KB"]) / 2 ** 20))
    grow = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{D}/ours_growth.tsv"), delimiter="\t"):
        grow[r["case"]].append((int(r["mib"]), int(r["peak_rss_KB"]) / 2 ** 20))
    return runs, dict(grow)


DATA = {}
for arm in ARMS:
    if arm == "pr2501":
        DATA[arm] = suite(f"{D}/pr2501_matrix.csv", f"{D}/pr2501_memory-growth.csv", "baseline")
    elif arm == "small":
        DATA[arm] = suite(f"{D}/small_matrix.csv", f"{D}/small_memory-growth.csv", "small")
    elif arm == "ours":
        DATA[arm] = ours()


def med(arm, case, w, f):
    x = DATA[arm][0].get((case, w))
    return None if not x else statistics.median(f(v) for v in x)


def scaling(fname, title, f):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, case, letter in zip(axes.flat, CASES, "abcd"):
        ref = med(ARMS[0], case, 8, f)
        for n, arm in enumerate(ARMS):
            ws = sorted(w for c, w in DATA[arm][0] if c == case)
            ys = [512 / med(arm, case, w, f) for w in ws]
            label, color, marker = STYLE[arm]
            ax.plot(ws, ys, color=color, marker=marker, label=label, lw=1.5, ms=5)
            if ws and ref:
                ax.annotate(f"{ref / med(arm, case, 8, f):.2f}×", (ws[-1], ys[-1]), xytext=(8, 7 * (len(ARMS) - 1 - 2 * n)),
                            textcoords="offset points", color=color, fontsize=9, fontweight="bold", va="center")
        ax.set_title(TITLE[case], loc="left", fontsize=11)
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)
        ax.set_xlabel("Workers"); ax.set_ylabel("Throughput (MiB/s)")
        ax.set_xticks([1, 2, 4, 6, 8]); ax.set_xlim(0, 9.8); ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.06, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.06, 0.935, "512 MiB per corpus | vocab 100,000 | min_frequency 2 | 2× AMD EPYC 7742, pinned", fontsize=9,
             color="#555555")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.05, 0.92), ncol=len(ARMS), frameon=False)
    fig.text(0.06, 0.01, f"Median of three runs. Endpoint labels: speedup vs {STYLE[ARMS[0]][0]} at 8 workers.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(f"{OUT}/{fname}", dpi=130)


def memory():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, case, letter in zip(axes.flat, CASES, "abcd"):
        ref = dict(DATA[ARMS[0]][1].get(case, []))
        for n, arm in enumerate(ARMS):
            xy = DATA[arm][1].get(case, [])
            label, color, marker = STYLE[arm]
            ax.plot([x for x, _ in xy], [y for _, y in xy], color=color, marker=marker, label=label, lw=1.5, ms=5)
            if xy and xy[-1][0] in ref:
                ax.annotate(f"{100 * xy[-1][1] / ref[xy[-1][0]]:.0f}%", xy[-1], xytext=(8, 7 * (len(ARMS) - 1 - 2 * n)), textcoords="offset points",
                            color=color, fontsize=9, fontweight="bold", va="center")
        ax.set_title(TITLE[case], loc="left", fontsize=11)
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)
        ax.set_xlabel("Raw input (MiB)"); ax.set_ylabel("Peak RSS (GiB)"); ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("BPE memory growth", x=0.06, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.06, 0.935, "8 workers | vocab 100,000 | min_frequency 2 | up to the whole prepared corpus", fontsize=9,
             color="#555555")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.05, 0.92), ncol=len(ARMS), frameon=False)
    fig.text(0.06, 0.01, f"One run per point. Labels: peak RSS as % of {STYLE[ARMS[0]][0]} at the largest input.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(f"{OUT}/memory-growth.png", dpi=130)


def tables():
    main = {r["case"]: r for r in csv.DictReader(open(f"{D}/main_vs_pr2501_matrix.csv")) if r["arm"] == "baseline"}
    heads = ["HF main"] + [STYLE[a][0] for a in ARMS]
    out = []
    for title, f, mainkey in (("Feed + Train at eight workers", lambda v: v[0] + v[1], "median_pipeline_seconds"),
                              ("Train only at eight workers", lambda v: v[1], "median_train_seconds")):
        out += [f"### {title}", "", "Seconds, median of three runs. Lower is better.", "",
                "| Workload | " + " | ".join(heads) + " |", "| --- |" + " ---: |" * len(heads)]
        for case in CASES:
            vals = [med(a, case, 8, f) for a in ARMS]
            best = min(vals)
            cells = [f"{float(main[case][mainkey]):.3f}"] + [f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in vals]
            out.append(f"| {TITLE[case]} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["### Matched-input memory", "", "Peak RSS in GiB, eight workers, 512 MiB.", "",
            "| Workload | " + " | ".join(heads) + " |", "| --- |" + " ---: |" * len(heads)]
    for case in CASES:
        vals = [max(v[2] for v in DATA[a][0][(case, 8)]) for a in ARMS]
        cells = [f"{float(main[case]['median_peak_rss_bytes']) / GiB:.3f}"] + [f"{v:.3f}" for v in vals]
        out.append(f"| {TITLE[case]} | " + " | ".join(cells) + " |")
    open(f"{OUT}/tables.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


scaling("core-scaling.png", "BPE throughput: Feed + Train", lambda v: v[0] + v[1])
scaling("train-core-scaling.png", "BPE throughput: Train only", lambda v: v[1])
memory()
tables()
