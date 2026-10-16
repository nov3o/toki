"""benchmark.py: HF `tokenizers` vs toki, from Python, on your own text.

    python3 benchmark.py --hf <python of the HF venv> --toki <python of the toki venv> <corpus.txt> [--threads N] [--vocab V]

The corpus is cut in two halves at a line break: training uses the first, encoding the second (no overlap). Each step
runs in its own process, in its own venv, so time and peak memory are per step:
- train: HF's `BpeTrainer` (GPT-2 byte-level; skipped when the HF build has no trainer, as `main`'s 1.0 bindings) vs
  toki's `train_bpe`; merges compared when both ran;
- encode the other half with the vocab toki trained: HF's `encode_batch` on ~256 KB pieces vs toki's `encode_file`;
  ids compared.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

WORKER = r'''
import hashlib, json, re, resource, sys, time
import tokenizers
task, args = sys.argv[1], json.loads(sys.argv[2])
t0 = time.perf_counter()
out = {"version": tokenizers.__version__}
if task == "train_hf":
    if not hasattr(tokenizers, "trainers"):
        print(json.dumps({"skipped": "this build has no trainer"})); sys.exit()
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=args["vocab"], show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train([args["train"]], tr)
    merges = json.loads(tok.to_str())["model"]["merges"]
    out["merges"] = len(merges)
    merges = [" ".join(m) if isinstance(m, list) else m for m in merges]
    out["merges_sha"] = hashlib.sha256(json.dumps(merges).encode()).hexdigest()
    open(args["model"] + ".hf_merges", "w").write("\n".join(merges))
elif task == "train_toki":
    vocab, merges = tokenizers.train_bpe(args["train"], args["vocab"], pre_tokenizer="bytelevel", threads=args["threads"])
    out["merges"] = len(merges)
    out["merges_sha"] = hashlib.sha256(json.dumps([f"{l} {r}" for l, r in merges]).encode()).hexdigest()
    open(args["model"] + ".toki_merges", "w").write("\n".join(f"{l} {r}" for l, r in merges))
    json.dump({"vocab": vocab, "merges": [[l, r] for l, r in merges]}, open(args["model"], "w"))
elif task == "encode_hf":
    import numpy as np
    tok = tokenizers.Tokenizer.from_file(args["tokenizer"])
    text = open(args["encode"], encoding="utf-8").read()
    cut = re.compile(r"[A-Za-z0-9]\n[A-Za-z0-9]")  # encoding restarts identically before such a "\n"
    pieces, i = [], 0
    while i < len(text):
        m = cut.search(text, i + (1 << 18))
        j = m.start() + 1 if m else len(text)
        pieces.append(text[i:j]); i = j
    h, n = hashlib.sha256(), 0
    for k in range(0, len(pieces), 256):
        for e in tok.encode_batch(pieces[k:k + 256], add_special_tokens=False):
            ids = np.asarray(e.ids_array if hasattr(e, "ids_array") else e.ids, dtype=np.uint32)  # numpy where the build has it
            h.update(ids.tobytes()); n += len(ids)
    out["ids"], out["ids_sha"] = n, h.hexdigest()
elif task == "encode_toki":
    tok = tokenizers.Tokenizer.from_file(args["tokenizer"])
    h, n = hashlib.sha256(), 0
    for ids in tok.encode_file(args["encode"], threads=args["threads"]):
        h.update(ids.tobytes()); n += len(ids)
    out["ids"], out["ids_sha"] = n, h.hexdigest()
out["seconds"] = time.perf_counter() - t0
out["peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(json.dumps(out))
'''


def run(python, task, args, threads):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    p = subprocess.run([python, "-c", WORKER, task, json.dumps(args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env)
    if p.returncode:
        sys.exit(f"{task} failed:\n{p.stderr}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def byte_level_tokenizer(model, path):
    """GPT-2 byte-level tokenizer.json from a trained vocab, with all 256 byte tokens (added at the end if missing)."""
    vocab = model["vocab"]
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs + [256 + k for k, _ in enumerate(b for b in range(256) if b not in bs)]
    for ch in (chr(c) for c in cs):
        vocab.setdefault(ch, len(vocab))
    bl = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
    json.dump({"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
               "pre_tokenizer": bl, "post_processor": None, "decoder": bl,
               "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                         "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                         "vocab": vocab, "merges": model["merges"]}}, open(path, "w"))


def halves(corpus, tmp):
    """The corpus cut at the line break nearest its middle: (train, encode). Copied in 16 MB blocks, so this process
    stays small (a child process inherits its parent's peak memory in the kernel's count)."""
    size = os.path.getsize(corpus)
    paths = os.path.join(tmp, "train.txt"), os.path.join(tmp, "encode.txt")
    with open(corpus, "rb") as f:
        f.seek(size // 2)
        mid = size // 2 + len(f.readline())
        f.seek(0)
        for p, n in zip(paths, (mid, size - mid)):
            with open(p, "wb") as out:
                while n:
                    b = f.read(min(n, 1 << 24))
                    out.write(b)
                    n -= len(b)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--hf", required=True, help="python of the venv with HF tokenizers")
    ap.add_argument("--toki", required=True, help="python of the venv with toki (nov3o/tokenizers)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--vocab", type=int, default=131072)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=os.environ.get("BENCH_TMP")) as tmp:
        train, enc = halves(a.corpus, tmp)
        args = {"train": train, "encode": enc, "vocab": a.vocab, "threads": a.threads,
                "model": os.path.join(tmp, "model.json"), "tokenizer": os.path.join(tmp, "tokenizer.json")}
        print(f"corpus {a.corpus}: train {os.path.getsize(train) / 1e9:.2f} GB, encode {os.path.getsize(enc) / 1e9:.2f} GB, "
              f"{a.threads} threads, vocab {a.vocab}\n")
        rows = []
        th, tt = run(a.hf, "train_hf", args, a.threads), run(a.toki, "train_toki", args, a.threads)
        rows.append(("train", th, tt))
        byte_level_tokenizer(json.load(open(args["model"])), args["tokenizer"])
        eh, et = run(a.hf, "encode_hf", args, a.threads), run(a.toki, "encode_toki", args, a.threads)
        rows.append(("encode", eh, et))

        print(f"HF tokenizers {eh['version']}  vs  toki {et['version']}\n")
        print(f"{'':8}{'HF':>22}{'toki':>22}{'speedup':>10}{'memory':>9}")
        for name, h, t in rows:
            if "skipped" in h:
                print(f"{name:8}{'(no trainer in this build)':>22}{t['seconds']:>11.2f} s{t['peak_gb']:>7.2f} GB")
                continue
            print(f"{name:8}{h['seconds']:>11.2f} s{h['peak_gb']:>7.2f} GB{t['seconds']:>11.2f} s{t['peak_gb']:>7.2f} GB"
                  f"{h['seconds'] / t['seconds']:>9.1f}×{100 * (t['peak_gb'] / h['peak_gb'] - 1):>+8.0f}%")
        if "skipped" not in th:
            hm, tm = (open(args["model"] + x).read().split("\n") for x in (".hf_merges", ".toki_merges"))
            n = min(len(hm), len(tm))
            same = hm[:n] == tm[:n]
            print(f"\nmerges identical: {same} (HF {len(hm)}, toki {len(tm)} merges", end="")
            print(f"; HF starts from all 256 bytes, toki from the bytes in the text, so the shorter list is the other's"
                  f" prefix)" if same and len(hm) != len(tm) else ")")
        print("ids identical:   ", eh["ids_sha"] == et["ids_sha"], f"({et['ids']} ids)")


if __name__ == "__main__":
    main()
