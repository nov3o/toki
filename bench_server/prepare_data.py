"""Build the benchmark datasets: random samples of documents from a big pool per source.

  python3 prepare_data.py <data_dir> <pool_GB> <size_GB> [<size_GB> ...]
  e.g. server:  venv/bin/python prepare_data.py data 100 1 10     (pool = 10x the largest size)
       laptop:  SOURCES=code python3 prepare_data.py data 0.5 0.1

Sources (1 GB = 10^9 bytes):
  english       HuggingFaceFW/fineweb-edu sample/100BT (sample/10BT is only ~45 GB, too small for a 100 GB pool)
  multilingual  HuggingFaceFW/fineweb-2, 7 languages, equal bytes each
  code          codeparrot/github-code-clean
Steps:
  1. pool:   download shards in random order until pool_GB of documents -> data/pool/<src>.txt + <src>.offsets
  2. sample: shuffle the pool's documents (seeded), take them until size_GB -> data/<src>_<size>GB.txt
  3. mix:    one third of size_GB from each source's pool, documents shuffled together -> data/mix_<size>GB.txt
Every document ends with a newline. Seed: SEED env var (default 0).
"""
import array
import os
import random
import sys

GB = 10**9
SEED = int(os.environ.get("SEED", 0))
LANGS = ["rus_Cyrl", "cmn_Hani", "deu_Latn", "fra_Latn", "spa_Latn", "jpn_Jpan", "arb_Arab"]


def shards(src):
    """[(repo, file, column, label)] for a source; shard order is shuffled so the pool is a random split."""
    from huggingface_hub import list_repo_files
    rng = random.Random(SEED)
    if src == "english":
        fs = [f for f in list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset") if f.startswith("sample/100BT/")]
        rng.shuffle(fs)
        return [[("HuggingFaceFW/fineweb-edu", f, "text") for f in fs]]
    if src == "multilingual":
        allf = list_repo_files("HuggingFaceFW/fineweb-2", repo_type="dataset")
        out = []
        for l in LANGS:
            fs = [f for f in allf if f.startswith(f"data/{l}/train/")]
            rng.shuffle(fs)
            out.append([("HuggingFaceFW/fineweb-2", f, "text") for f in fs])
        return out                                    # one list per language, equal budget each
    if src == "code":
        fs = [f for f in list_repo_files("codeparrot/github-code-clean", repo_type="dataset") if f.endswith(".parquet")]
        rng.shuffle(fs)
        return [[("codeparrot/github-code-clean", f, "code") for f in fs]]
    raise SystemExit(f"unknown source {src}")


def build_pool(data, src, pool_bytes):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path, offp = f"{data}/pool/{src}.txt", f"{data}/pool/{src}.offsets"
    if os.path.exists(offp):
        print(f"[pool exists] {path}", flush=True)
        return
    os.makedirs(f"{data}/pool/_dl", exist_ok=True)
    offsets = array.array("Q", [0])                    # document i = bytes offsets[i]..offsets[i+1]
    with open(path, "wb") as out:
        groups = shards(src)
        for group in groups:
            budget = offsets[-1] + pool_bytes // len(groups)
            for repo, fn, col in group:
                p = hf_hub_download(repo, fn, repo_type="dataset", local_dir=f"{data}/pool/_dl")
                for b in pq.ParquetFile(p).iter_batches(columns=[col], batch_size=4096):
                    for t in b.column(0).to_pylist():
                        if not t:
                            continue
                        d = t.encode("utf-8")
                        if not d.endswith(b"\n"):
                            d += b"\n"
                        out.write(d)
                        offsets.append(offsets[-1] + len(d))
                        if offsets[-1] >= budget:
                            break
                    if offsets[-1] >= budget:
                        break
                os.remove(p)
                print(f"  pool {src}: {offsets[-1] / GB:.2f} GB ({fn})", flush=True)
                if offsets[-1] >= budget:
                    break
    with open(offp, "wb") as f:
        offsets.tofile(f)
    print(f"[pool done] {path}: {offsets[-1] / GB:.2f} GB, {len(offsets) - 1} documents", flush=True)


def load_offsets(data, src):
    offsets = array.array("Q")
    with open(f"{data}/pool/{src}.offsets", "rb") as f:
        offsets.frombytes(f.read())
    return offsets


def sample(data, srcs, size_bytes, out_path, seed):
    """Random documents from the pools of `srcs` (equal bytes per source), shuffled together, until size_bytes."""
    if os.path.exists(out_path):
        print(f"[exists] {out_path}", flush=True)
        return
    rng = random.Random(seed)
    picks = []                                         # (src, doc index)
    per_src = size_bytes // len(srcs)
    for src in srcs:
        off = load_offsets(data, src)
        order = list(range(len(off) - 1))
        rng.shuffle(order)
        total = 0
        for i in order:
            if total >= per_src:
                break
            picks.append((src, i))
            total += off[i + 1] - off[i]
        if total < per_src:
            print(f"  warning: pool {src} has only {total / GB:.2f} GB", flush=True)
    rng.shuffle(picks)
    pools = {s: open(f"{data}/pool/{s}.txt", "rb") for s in srcs}
    offs = {s: load_offsets(data, s) for s in srcs}
    written = 0
    with open(out_path + ".tmp", "wb") as out:
        for src, i in picks:
            if written >= size_bytes:
                break
            f, off = pools[src], offs[src]
            f.seek(off[i])
            d = f.read(min(off[i + 1] - off[i], size_bytes - written))
            out.write(d)
            written += len(d)
    os.rename(out_path + ".tmp", out_path)
    print(f"[done] {out_path}: {written / GB:.3f} GB", flush=True)


def main():
    data, pool_gb, sizes = sys.argv[1], float(sys.argv[2]), [float(x) for x in sys.argv[3:]]
    srcs = os.environ.get("SOURCES", "english multilingual code").split()
    for src in srcs:
        build_pool(data, src, int(pool_gb * GB))
    for size in sizes:
        tag = f"{size:g}GB"
        for src in srcs:
            sample(data, [src], int(size * GB), f"{data}/{src}_{tag}.txt", seed=SEED)
        if len(srcs) == 3:
            sample(data, srcs, int(size * GB), f"{data}/mix_{tag}.txt", seed=SEED + 1)


main()
