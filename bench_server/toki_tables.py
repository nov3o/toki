"""toki_tables.py <results.tsv> <out_dir>: the README tables from toki_axes.sh.

Writes <out_dir>/AVERAGE.md (default setting, then one table per axis, average of the 4 datasets) and
<out_dir>/BY_DATASET.md (the same per dataset). Time = median of the runs, memory = peak RSS (max of the runs).
Training: PR #2501 vs ours. Encoding: HF main vs ours. Ours in bold, with its speedup in parentheses.
"""
import collections
import statistics
import sys

DS = ["english", "multilingual", "code", "mix"]
DEFAULT = ("1GB", "8", "bytelevel", "131072", "none")
AXES = [  # (axis, title, column header, which field varies, values)
    ("threads", "Threads", "threads", 1, ["2", "4", "8", "16"]),
    ("vocab", "Vocab", "vocab", 3, ["32768", "65536", "131072", "262144"]),
    ("pretok", "Pre-tokenizer", "pre-tokenizer", 2, ["bytelevel", "llama3", "gpt4o", "whitespace", "whitespace_split",
                                                     "metaspace", "bert", "punctuation", "digits", "unicode_scripts",
                                                     "char_delimiter"]),
    ("norm", "Normalizer", "normalizer", 4, ["none", "nfc", "nfd", "nfkc", "nfkd", "lowercase", "strip_accents",
                                             "nfkc_lowercase", "replace", "bert"]),
    ("size", "Size", "size", 0, ["1GB", "10GB"]),
]
VOCAB = {"32768": "32k", "65536": "64k", "131072": "128k", "262144": "256k"}
PAIRS = [("train", "pr2501", "PR #2501 train"), ("encode", "hfmain", "HF main enc")]

runs = collections.defaultdict(list)
checks = collections.Counter()
lines = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
for r in (dict(zip(lines[0], l)) for l in lines[1:]):
    key = (r["task"], r["dataset"], r["size"], r["threads"], r["pretok"], r["vocab"], r["norm"], r["impl"])
    runs[key].append((float(r["wall_s"]), float(r["peak_rss_KB"]) / 1048576))
    if r["impl"] == "ours":
        checks[(r["task"], r["check"].split()[0])] += 1


def cell(task, ds, s, impl):
    x = runs.get((task, ds) + s + (impl,))
    return None if not x else (statistics.median(t for t, _ in x), max(m for _, m in x))


def avg(task, s, impl):
    cs = [cell(task, ds, s, impl) for ds in DS]
    return None if any(c is None for c in cs) else (sum(c[0] for c in cs) / 4, sum(c[1] for c in cs) / 4)


def t(x):
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def cells(get, s):
    out = []
    for task, base, _ in PAIRS:
        b, o = get(task, s, base), get(task, s, "ours")
        out += ["🕐", "🕐"] if b is None else [t(b[0]), f"{b[1]:.2f}"]
        if o is None:
            out += ["🕐", "🕐"]
        else:
            sp = f" ({b[0] / o[0]:.1f}×)" if b else ""
            out += [f"**{t(o[0])}**{sp}", f"**{o[1]:.2f}**"]
    return out


HEAD = "| PR #2501 train s | MEM | **ours** train s | MEM | HF main enc s | MEM | **ours** enc s | MEM |"


def setting(field, v):
    s = list(DEFAULT)
    s[field] = v
    return tuple(s)


def fmt(axis, v):
    return VOCAB.get(v, v) if axis == "vocab" else v.replace("GB", " GB") if axis == "size" else v


avg_out = ["#### Default setting", "", "| dataset " + HEAD, "|---" * 9 + "|"]
for ds in DS:
    avg_out.append(f"| {ds} | " + " | ".join(cells(lambda task, s, impl: cell(task, ds, s, impl), DEFAULT)) + " |")
avg_out.append("| **average** | " + " | ".join(cells(avg, DEFAULT)) + " |")
avg_out.append("")
by_out = ["# Results by dataset", ""]
for axis, title, header, field, values in AXES:
    avg_out += [f"#### {title}", "", f"| {header} " + HEAD, "|---" * 9 + "|"]
    by_out += [f"## {title}", "", f"| {header} | dataset " + HEAD, "|---" * 10 + "|"]
    for v in values:
        s = setting(field, v)
        avg_out.append(f"| {fmt(axis, v)} | " + " | ".join(cells(avg, s)) + " |")
        for i, ds in enumerate(DS):
            by_out.append(f"| {fmt(axis, v) if i == 0 else ''} | {ds} | "
                          + " | ".join(cells(lambda task, s, impl: cell(task, ds, s, impl), s)) + " |")
    avg_out.append("")
    by_out.append("")
eq = (f"Equivalence: {sum(v for (k, c), v in checks.items() if k == 'train')} training runs of ours, "
      f"{checks[('train', 'same')]} with merges identical to PR #2501's; "
      f"{sum(v for (k, c), v in checks.items() if k == 'encode')} encoding runs, "
      f"{checks[('encode', 'same')]} with ids identical to HF main's (token by token).")
avg_out = [eq, ""] + avg_out
open(f"{sys.argv[2]}/AVERAGE.md", "w").write("\n".join(avg_out))
open(f"{sys.argv[2]}/BY_DATASET.md", "w").write("\n".join(by_out))
print("\n".join(avg_out))
