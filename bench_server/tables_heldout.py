"""tables_heldout.py <results.tsv> <out_dir>: the README tables from run.sh's results.

Writes <out_dir>/AVERAGE.md (one table per axis, average of the 4 datasets) and <out_dir>/BY_DATASET.md
(the same per dataset, with throughput). Time = median of the runs of a setting, memory = peak RSS (max of the runs).
"""
import collections
import csv
import math
import statistics
import sys

DS = ["english", "multilingual", "code", "mix"]
IMPLS = ["hfmain", "tiktoken", "ours"]
AXES = [  # (axis name in results, title, column header, value of the row, how to print it)
    ("threads", "Threads", "threads", "threads", str),
    ("vocab", "Vocab", "vocab", "vocab", lambda v: {"32768": "32k", "65536": "64k", "131072": "128k", "262144": "256k"}[v]),
    ("pretok", "Pre-tokenizer", "pre-tokenizer", "pretok", str),
    ("norm", "Normalizer", "normalizer", "norm", str),
]
ORDER = {
    "threads": ["2", "4", "8", "16"],
    "vocab": ["32768", "65536", "131072", "262144"],
    "pretok": ["bytelevel", "llama3", "gpt4o", "whitespace", "whitespace_split", "metaspace", "bert", "punctuation",
               "digits", "unicode_scripts", "char_delimiter"],
    "norm": ["none", "nfc", "nfd", "nfkc", "nfkd", "lowercase", "strip_accents", "nfkc_lowercase", "replace", "bert"],
}
TEST_GB = 0.5          # size of each test half (bytes / 10^9), for MB/s


def load(path):
    runs = collections.defaultdict(list)          # (axis, dataset, threads, pretok, vocab, norm, impl) -> [(s, MB)]
    checks = []
    for r in csv.DictReader(open(path), delimiter="\t"):
        k = (r["axis"], r["dataset"], r["threads"], r["pretok"], r["vocab"], r["norm"], r["impl"])
        runs[k].append((float(r["wall_s"]), float(r["peak_rss_MB"])))
        if r["ids_vs_hfmain"] != "ref":
            checks.append(r["ids_vs_hfmain"])
    return runs, checks


def cell(runs, key):
    """median time, max memory (GB) of a setting, or None."""
    x = runs.get(key)
    return None if not x else (statistics.median(t for t, _ in x), max(m for _, m in x) / 1024)


def rows_for(axis):
    """All (axis-in-results, threads, pretok, vocab, norm) of one axis, the default setting included."""
    default = ("default", "8", "bytelevel", "131072", "none")
    out = []
    for v in ORDER[axis]:
        s = dict(threads="8", pretok="bytelevel", vocab="131072", norm="none")
        s[{"threads": "threads", "vocab": "vocab", "pretok": "pretok", "norm": "norm"}[axis]] = v
        t = (s["threads"], s["pretok"], s["vocab"], s["norm"])
        out.append((v, default if t == default[1:] else (axis,) + t))
    return out


def fmt_t(x):
    return "🕐" if x is None else (f"{x:.0f}" if x >= 100 else f"{x:.2f}" if x < 10 else f"{x:.1f}")


def ours_cell(o, h, k, mbs=False):
    if o is None:
        return "🕐"
    sp = [f"{r[0] / o[0]:.1f}×" for r in (h, k) if r is not None]
    return f"**{fmt_t(o[0])}**" + (f" ({', '.join(sp)})" if sp else "")


def average_table(runs, axis, title, header, fmt):
    lines = [f"#### {title}", "",
             f"| {header} | HF main s | MEM | tiktoken s | MEM | **ours** s | MEM |", "|---|---|---|---|---|---|---|"]
    for v, key in rows_for(axis):
        avg = {}
        for impl in IMPLS:
            cs = [cell(runs, key[:1] + (ds,) + key[1:] + (impl,)) for ds in DS]
            avg[impl] = None if any(c is None for c in cs) else (sum(c[0] for c in cs) / 4, sum(c[1] for c in cs) / 4)
        h, k, o = avg["hfmain"], avg["tiktoken"], avg["ours"]
        tik_na = k is None and h is not None and (key[2] not in ("bytelevel", "llama3", "gpt4o") or key[4] != "none")
        kt = "n/a" if tik_na else fmt_t(k and k[0])
        km = "n/a" if tik_na else ("🕐" if k is None else f"{k[1]:.2f}")
        lines.append(f"| {fmt(v)} | {fmt_t(h and h[0])} | {'🕐' if h is None else f'{h[1]:.2f}'} | {kt} | {km} | "
                     f"{ours_cell(o, h, k)} | {'🕐' if o is None else f'**{o[1]:.2f}**'} |")
    return lines


def dataset_table(runs, axis, title, header, fmt):
    lines = [f"## {title}", "",
             f"| {header} | dataset | HF main s | MB/s | MEM | tiktoken s | MB/s | MEM | **ours** s | MB/s | MEM |",
             "|" + "---|" * 11]
    for v, key in rows_for(axis):
        for i, ds in enumerate(DS):
            c = {impl: cell(runs, key[:1] + (ds,) + key[1:] + (impl,)) for impl in IMPLS}
            tik_na = key[2] not in ("bytelevel", "llama3", "gpt4o") or key[4] != "none"
            part = []
            for impl in IMPLS:
                x = c[impl]
                if impl == "tiktoken" and tik_na:
                    part += ["n/a"] * 3
                elif x is None:
                    part += ["🕐"] * 3
                else:
                    t = ours_cell(x, c["hfmain"], c["tiktoken"]) if impl == "ours" else fmt_t(x[0])
                    m = f"**{x[1]:.2f}**" if impl == "ours" else f"{x[1]:.2f}"
                    part += [t, f"{TEST_GB * 1000 / x[0]:.0f}", m]
            lines.append(f"| {fmt(v) if i == 0 else ''} | {ds} | " + " | ".join(part) + " |")
    return lines


def headline(runs):
    lines = ["| | HF main | tiktoken | **ours** |", "|---|---|---|---|"]
    for t in ("2", "16"):
        c = {impl: cell(runs, ("headline", "english", t, "llama3", "32768", "none", impl)) for impl in IMPLS}
        h, k, o = c["hfmain"], c["tiktoken"], c["ours"]
        g = lambda x: "🕐" if x is None else f"{fmt_t(x[0])} s / {x[1]:.2f} GB"
        lines.append(f"| {t} threads | {g(h)} | {g(k)} | "
                     + ("🕐" if o is None else f"**{fmt_t(o[0])} s** ({', '.join(f'{r[0] / o[0]:.1f}×' for r in (h, k) if r)}) / **{o[1]:.2f} GB**") + " |")
    return lines


def thread_law(runs):
    """time ∝ threads^b per implementation (average of the datasets), least squares in log-log."""
    out = []
    for impl in IMPLS:
        pts = []
        for v, key in rows_for("threads"):
            cs = [cell(runs, key[:1] + (ds,) + key[1:] + (impl,)) for ds in DS]
            if all(cs):
                pts.append((int(v), sum(c[0] for c in cs) / 4))
        if len(pts) >= 3:
            lx = [math.log(x) for x, _ in pts]; ly = [math.log(y) for _, y in pts]
            mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
            b = sum((a - mx) * (c - my) for a, c in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
            out.append((impl, b, pts[0][1] / pts[-1][1]))
    return out


def main():
    runs, checks = load(sys.argv[1])
    out = sys.argv[2]
    avg = ["## Results (average of the 4 datasets; s = wall time, MEM = peak memory in GB)", "",
           "Encoding the 0.5 GB held-out half of each dataset. **ours** in parentheses: speedup vs HF main, then vs tiktoken.",
           "🕐 = not run yet; n/a = not supported (tiktoken: regex pre-tokenizers only, no normalizers).", ""]
    for a in AXES:
        avg += average_table(runs, a[0], a[1], a[2], a[4]) + [""]
    same = sum(c.startswith("same") for c in checks)
    avg += [f"Equivalence: {len(checks)} encodings compared token by token with HF main's ids, {same} identical.", ""]
    avg += ["## Headline (llama3 pre-tokenizer, no normalizer, vocab 32k, english)", ""] + headline(runs) + [""]
    law = thread_law(runs)
    if law:
        avg += ["## Thread scaling (time ∝ threads^b, 2 → 16 threads)", "", "| | b | 2 → 16 threads |", "|---|---|---|"]
        avg += [f"| {i} | {b:.2f} | {r:.1f}× faster |" for i, b, r in law] + [""]
    open(f"{out}/AVERAGE.md", "w").write("\n".join(avg))
    det = ["# Results by dataset", "", "Encoding the 0.5 GB held-out half of each dataset; MB/s = input MB per second.", ""]
    for a in AXES:
        det += dataset_table(runs, a[0], a[1], a[2], a[4]) + [""]
    open(f"{out}/BY_DATASET.md", "w").write("\n".join(det))
    print(f"{len(checks)} checks, {same} identical; wrote {out}/AVERAGE.md, {out}/BY_DATASET.md")


main()
