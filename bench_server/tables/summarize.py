# Builds tables/SUMMARY.md from tables/results_*.tsv (+ *_verify.tsv). Handles rows with/without the norm column.
import csv, collections, sys
H = "dataset size_GB impl threads pretok vocab norm step status wall_s inner_s peak_rss_MB tokens dir".split()
def load(p):
    rows = []
    for f in list(csv.reader(open(p), delimiter="\t"))[1:]:
        if len(f) == 13: f = f[:6] + ["none"] + f[6:]
        r = dict(zip(H, f)); rows.append(r)
    last = {}
    for r in rows: last[(r["dataset"], r["impl"], r["threads"], r["pretok"], r["vocab"], r["norm"], r["step"])] = r
    return list(last.values())
def verdicts(p):
    v = collections.defaultdict(set)
    for r in csv.DictReader(open(p), delimiter="\t"): v[r["run"].split("/")[-1] if "/" in r["run"] else r["run"]].add(r["result_vs_hf"])
    return v
def cell(d, impl, step, f="wall_s"):
    r = d.get((impl, step)); 
    if not r: return ""
    if r["status"] != "ok": return r["status"].replace("failed_", "")
    return f"{float(r[f]):.1f}" if f == "wall_s" else f"{float(r[f])/1024:.2f}"
out = []
for name, axis, title in [("pretok", "pretok", "Pre-tokenizers"), ("axes", None, "Threads and vocab"), ("norm", "norm", "Normalizers")]:
    try: rows = load(f"tables/results_{name}.tsv")
    except FileNotFoundError: continue
    try:
        vs = [r["result_vs_hf"] for r in csv.DictReader(open(f"tables/results_{name}_verify.tsv"), delimiter="\t")]
        ver = f"{len(vs)} comparisons vs HF, {sum(x != 'same' for x in vs)} different"
    except FileNotFoundError: ver = "not verified yet"
    g = collections.defaultdict(dict)
    for r in rows: g[(r["dataset"], r["threads"], r["pretok"], r["vocab"], r["norm"])][(r["impl"], r["step"])] = r
    out += [f"## {title} (1 GB) — {ver}", "",
            "time in s, RAM in GB (peak RSS). ours = SIMD build; nosimd = build without SIMD.", "",
            "| dataset | threads | pretok | vocab | norm | HF train | ours train | nosimd train | HF enc | tiktoken enc | ours enc | nosimd enc | HF train RAM | ours train RAM | HF enc RAM | ours enc RAM |",
            "|" + "---|" * 16]
    for k in sorted(g, key=lambda k: (k[0], int(k[1]), k[2], int(k[3]), k[4])):
        d = g[k]
        out.append("| " + " | ".join(list(k) + [cell(d,"hf","train"), cell(d,"ours_simd","train"), cell(d,"ours_nosimd","train"),
            cell(d,"hf","encode"), cell(d,"tiktoken","encode"), cell(d,"ours_simd","encode"), cell(d,"ours_nosimd","encode"),
            cell(d,"hf","train","peak_rss_MB"), cell(d,"ours_simd","train","peak_rss_MB"),
            cell(d,"hf","encode","peak_rss_MB"), cell(d,"ours_simd","encode","peak_rss_MB")]) + " |")
    out.append("")
open("tables/SUMMARY.md", "w").write("# Results (1 GB, AMD EPYC 7742)\n\n" + "\n".join(out))
