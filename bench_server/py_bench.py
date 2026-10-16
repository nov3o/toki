"""py_bench.py <mode> <tokenizer.json | vocab> <corpus>: the Python-binding side of the toki benchmark.

encode_file: `Tokenizer.encode_file`, the file streamed from Rust in numpy id arrays (written to <out> if given).
encode_batch: the file cut into ~256 KB pieces before a "\\n" between ASCII letters/digits (as the Rust driver
`hfenc`), 256 pieces per `encode_batch` call. encode_whole: the file as one `encode` call. train: `train_bpe`. train_pr2501: `train_bpe_pr2501`.
Prints ids= (or merges=)."""
import sys
import tokenizers

mode, arg, path = sys.argv[1:4]
if mode in ("train", "train_pr2501"):
    # train <vocab> <corpus> <threads> [pretok] [normalizers, comma-separated] [merges out]
    # train_pr2501 <vocab> <corpus> <tokenizer.json> [byte_level 0|1] [merges out]
    if mode == "train":
        a = sys.argv[4:] + [""] * 4
        vocab, merges = tokenizers.train_bpe(path, int(arg), pre_tokenizer=a[1] or "bytelevel",
                                             normalizers=[x for x in a[2].split(",") if x], threads=int(a[0]))
        out = a[3]
    else:
        a = sys.argv[4:] + [""] * 3
        vocab, merges = tokenizers.train_bpe_pr2501(path, a[0], int(arg), byte_level=a[1] != "0")
        out = a[2]
    if out:
        open(out, "w").write("".join(f"{l} {r}\n" for l, r in merges))
    print(f"merges={len(merges)}")
    sys.exit()
tok = tokenizers.Tokenizer.from_file(arg)
if mode == "encode_file":
    out = open(sys.argv[4], "wb") if len(sys.argv) > 4 else None
    ids = 0
    for a in tok.encode_file(path, threads=int(sys.argv[5]) if len(sys.argv) > 5 else None):
        ids += len(a)
        if out:
            a.tofile(out)
    print(f"ids={ids}")
    sys.exit()
text = open(path, "rb").read().decode("utf-8", "replace")
if mode == "encode_whole":
    print(f"ids={len(tok.encode(text, add_special_tokens=False).ids_array)}")
    sys.exit()
pieces, start, n = [], 0, len(text)
while start < n:
    c = start + (256 << 10)
    while c + 1 < n and not (text[c] == "\n" and text[c - 1].isascii() and text[c - 1].isalnum()
                              and text[c + 1].isascii() and text[c + 1].isalnum()):
        c += 1
    c = min(c, n) if c + 1 < n else n
    pieces.append(text[start:c])
    start = c
ids = 0
for i in range(0, len(pieces), 256):
    ids += sum(len(e.ids_array) for e in tok.encode_batch(pieces[i:i + 256], add_special_tokens=False))
print(f"ids={ids}")
