"""union_tables.py <results_union dir>: markdown tables from the union runs (encode_10gb.txt, encode_sweep.txt,
train_sweep.txt; space-separated with a header line). Writes <dir>/SUMMARY.md and prints it.
Later lines of the same setting replace earlier ones (reruns). Averages are over the 4 datasets."""
import collections
import sys

D = sys.argv[1]
DS = ["english", "multilingual", "code", "mix"]


def rows(name):
    lines = [l.split() for l in open(f"{D}/{name}") if l.strip()]
    head = lines[0]
    return [dict(zip(head, l)) for l in lines[1:] if l != head]


def last(rs, key):
    out = {}
    for r in rs:
        out[key(r)] = r
    return out


def gb(kb):
    return float(kb) / 1024 / 1024


out = []
checks = collections.Counter()

# 10 GB encode
e10 = last(rows("encode_10gb.txt"), lambda r: (r["dataset"], r["cores"], r["impl"]))
out += ["## Encode 10 GB (byte-level, vocab 128k): HF main vs union", "",
        "| dataset | cores | HF main | **union** | speedup | ids |", "|---|---|---|---|---|---|"]
for ds in DS:
    for c in ("16", "64"):
        h, u = e10.get((ds, c, "hfmain")), e10.get((ds, c, "union"))
        if not h or not u:
            continue
        checks[u["ids"]] += 1
        out.append(f"| {ds} | {c} | {float(h['wall_s']):.1f} s / {gb(h['rss_KB']):.1f} GB | "
                   f"**{float(u['wall_s']):.2f} s / {gb(u['rss_KB']):.1f} GB** | "
                   f"{float(h['wall_s']) / float(u['wall_s']):.1f}× | {u['ids']} |")
out.append("")

# encode sweep (0.5 GB held-out halves)
es = last(rows("encode_sweep.txt"), lambda r: (r["dataset"], r["axis"], r["threads"], r["pretok"], r["norm"], r["impl"]))
for axis, col, title in (("pretok", "pretok", "pre-tokenizer"), ("norm", "norm", "normalizer")):
    out += [f"## Encode 0.5 GB held-out, by {title} (average of 4 datasets): HF main vs union", "",
            f"| {title} | HF main 8 cores | **union 8 cores** | HF main 16 cores | **union 16 cores** |",
            "|---|---|---|---|---|"]
    vals = sorted({k[3] if col == "pretok" else k[4] for k in es if k[1] == axis}, key=lambda v: list(dict.fromkeys(
        k[3] if col == "pretok" else k[4] for k in es if k[1] == axis)).index(v))
    for v in vals:
        cells = []
        for t in ("8", "16"):
            hs, us = [], []
            for ds in DS:
                pt, nm = (v, "none") if col == "pretok" else ("bytelevel", v)
                h, u = es.get((ds, axis, t, pt, nm, "hfmain")), es.get((ds, axis, t, pt, nm, "union"))
                if h and u:
                    hs.append(float(h["wall_s"])); us.append(float(u["wall_s"])); checks[u["ids"]] += 1
            if len(hs) == 4:
                h, u = sum(hs) / 4, sum(us) / 4
                cells += [f"{h:.2f} s", f"**{u:.2f} s** ({h / u:.1f}×)"]
            else:
                cells += ["🕐", "🕐"]
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    out.append("")

# train sweep (1 GB)
ts = last(rows("train_sweep.txt"), lambda r: (r["dataset"], r["axis"], r["pretok"], r["norm"], r["impl"]))
for axis, col, title in (("pretok", "pretok", "pre-tokenizer"), ("norm", "norm", "normalizer")):
    out += [f"## Train 1 GB, vocab 128k, 16 cores, by {title} (average of 4 datasets): ours vs union", "",
            f"| {title} | ours | **union** | speedup |", "|---|---|---|---|"]
    vals = list(dict.fromkeys(k[2] if col == "pretok" else k[3] for k in ts if k[1] == axis))
    for v in vals:
        os_, us, ms = [], [], []
        for ds in DS:
            pt, nm = (v, "none") if col == "pretok" else ("bytelevel", v)
            o, u = ts.get((ds, axis, pt, nm, "ours")), ts.get((ds, axis, pt, nm, "union"))
            if o and u:
                os_.append(float(o["wall_s"])); us.append(float(u["wall_s"])); ms.append(gb(u["rss_KB"]))
                checks[u["merges"]] += 1
        if len(os_) == 4:
            o, u = sum(os_) / 4, sum(us) / 4
            out.append(f"| {v} | {o:.1f} s | **{u:.1f} s** / {sum(ms) / 4:.1f} GB | {o / u:.1f}× |")
        else:
            out.append(f"| {v} | 🕐 | 🕐 | 🕐 |")
    out.append("")

out = [f"Identical outputs (ids token by token, merges byte by byte): {checks['same']} of {sum(checks.values())}.", ""] + out
open(f"{D}/SUMMARY.md", "w").write("\n".join(out))
print("\n".join(out))
