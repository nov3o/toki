"""join_shards.py <out>: concatenate <out>.k pieces in <out>.index order to <out>.joined (verification only)."""
import sys
o = sys.argv[1]; fs = {}
with open(o + ".joined", "wb") as g:
    for line in open(o + ".index"):
        k, off, n = map(int, line.split())
        f = fs.get(k) or fs.setdefault(k, open(f"{o}.{k}", "rb"))
        f.seek(off * 4); g.write(f.read(n * 4))
