"""After the benchmark: check that ours and HF produced the same thing.

  python3 verify.py <results_dir>              # every run in results_dir, ours_* vs hf
  python3 verify.py <file_a> <file_b>          # any two files (merges.txt, vocab.txt or ids.u32)

For each run dir it compares ours_simd, ours_nosimd and tiktoken against hf:
  train:  merges.txt (every merge step, in order) and vocab.txt (final vocab)
  encode: merges_used.txt (merges the encoder loaded) and the final token sequence:
          ids.sha256 (checksum, default) or ids.u32 (if run with KEEP_IDS=1)
Prints one line per comparison: same / DIFFERENT (with the first differing merge step / token position).
Writes <results_dir>/verify.tsv.
"""
import os
import sys


def first_diff_line(a, b):
    """0 if the two text files are identical, else the 1-based first differing line."""
    with open(a, "rb") as fa, open(b, "rb") as fb:
        n = 0
        while True:
            x, y = fa.readline(), fb.readline()
            n += 1
            if x != y:
                return n
            if not x:
                return 0


def first_diff_id(a, b, chunk=1 << 24):
    """-1 if the two uint32 id files are identical, else the 0-based first differing token position."""
    with open(a, "rb") as fa, open(b, "rb") as fb:
        pos = 0
        while True:
            x, y = fa.read(chunk), fb.read(chunk)
            if x != y:
                i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
                return (pos + i) // 4
            if not x:
                return -1
            pos += len(x)


def compare(a, b):
    if a.endswith(".u32"):                       # token by token: counts, empty files, first differing token
        na, nb = os.path.getsize(a) // 4, os.path.getsize(b) // 4
        if na == 0 or nb == 0:
            return f"EMPTY ({na} vs {nb} ids)"
        d = first_diff_id(a, b)
        if d < 0:
            return f"same ({na} ids)"
        ids = []
        for p in (a, b):
            with open(p, "rb") as f:
                f.seek(d * 4)
                x = f.read(4)
                ids.append(int.from_bytes(x, "little") if len(x) == 4 else "end")
        return f"DIFFERENT at token {d}: {ids[0]} vs {ids[1]} ({na} vs {nb} ids)"
    d = first_diff_line(a, b)
    return "same" if d == 0 else f"DIFFERENT at line {d}"


def main():
    if len(sys.argv) == 3:
        print(compare(sys.argv[1], sys.argv[2]))
        return
    res = sys.argv[1]
    rows = []
    for root, dirs, files in sorted(os.walk(res)):
        if "train_hf" not in dirs:
            continue
        impls = sorted({d.split("_", 1)[1] for d in dirs if d.startswith(("train_", "encode_"))} - {"hf"})
        for impl in impls:                            # ours_simd, ours_nosimd, tiktoken (encode only)
            for st, names in (("train", ["merges.txt", "vocab.txt"]), ("encode", ["merges_used.txt", "ids.u32", "ids.sha256"])):
                for name in names:
                    a = os.path.join(root, f"{st}_hf", name)
                    b = os.path.join(root, f"{st}_{impl}", name)
                    if st == "encode" and os.path.exists(b.replace("ids.sha256", "ids.bytecmp")) and name == "ids.sha256":
                        name, a, b = "ids.bytecmp", None, b.replace("ids.sha256", "ids.bytecmp")   # byte-by-byte result from run.sh
                        r = open(b).readline().strip()
                        rows.append((os.path.relpath(root, res), impl, st, name, r))
                        print(f"{rows[-1][0]:45s} {impl:12s} {st:6s} {name:16s} {r}")
                        continue
                    if os.path.exists(a) and os.path.exists(b):
                        r = compare(a, b)
                        rows.append((os.path.relpath(root, res), impl, st, name, r))
                        print(f"{rows[-1][0]:45s} {impl:12s} {st:6s} {name:16s} {r}")
    with open(os.path.join(res, "verify.tsv"), "w") as f:
        f.write("run\timpl\tstep\tfile\tresult_vs_hf\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    bad = [r for r in rows if not r[4].startswith("same")]
    print(f"{len(rows)} comparisons, {len(bad)} different")


main()
