"""Small helpers shared by the train/encode scripts: cutting the corpus into pieces, writing merges / vocab / ids files."""
import json
import os
import struct


def save_merges_and_vocab(tokenizer_json, merges_out, vocab_out):
    """merges_out: one merge per line "left right", in merge order (line i = merge step i).
    vocab_out:  "id<TAB>token" in id order. Same format our Rust trainer writes."""
    model = json.load(open(tokenizer_json, encoding="utf-8"))["model"]
    with open(merges_out, "w", encoding="utf-8") as f:
        for m in model["merges"]:
            f.write((" ".join(m) if isinstance(m, list) else m) + "\n")
    with open(vocab_out, "w", encoding="utf-8") as f:
        for tok, i in sorted(model["vocab"].items(), key=lambda x: x[1]):
            f.write(f"{i}\t{tok}\n")


def ids_path(out_dir):
    """Where the token ids go: $IDS_PATH if set (run.sh points it at /dev/shm = RAM, so all engines write
    at memory speed, not the shared filesystem's), else out_dir/ids.u32."""
    return os.environ.get("IDS_PATH") or os.path.join(out_dir, "ids.u32")


def write_ids(f, ids):
    """Token ids as little-endian uint32 (same as our encoder)."""
    f.write(struct.pack(f"<{len(ids)}I", *ids))


# ---- cutting the corpus for encoders that take a list of texts (HF, tiktoken)
# Our encoder treats the whole file as one text and only cuts where pre-tokenization provably restarts.
# HF and tiktoken get pieces cut at points of exactly the same kind, so all three encode the same text.
# The rules are rust_bpe/src/pretok.rs safe_cut (keep the two in sync).
PIECE = 256 << 10     # bytes per piece

_PUNCT = set(b"""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""")


def _alnum(c):
    return 48 <= c <= 57 or 65 <= c <= 90 or 97 <= c <= 122


def safe_cut_at(pretok, b, c):
    if c <= 0 or c >= len(b) or b[c] >= 0x80:   # b[c] ASCII: normalizers can't join across the cut
        return False
    if pretok in ("bytelevel", "llama3", "gpt4o"):
        return b[c] == 10 and _alnum(b[c - 1]) and c + 1 < len(b) and _alnum(b[c + 1])
    if pretok in ("whitespace", "whitespace_split", "bert"):
        return b[c - 1] == 10
    if pretok == "metaspace":
        return b[c] == 32
    if pretok == "char_delimiter":
        return b[c - 1] == 32
    if pretok == "punctuation":
        return b[c - 1] in _PUNCT
    if pretok == "digits":
        return 48 <= b[c - 1] <= 57
    if pretok == "unicode_scripts":
        return (65 <= b[c - 1] <= 90 or 97 <= b[c - 1] <= 122) and b[c] in _PUNCT
    raise ValueError(pretok)


def safe_cut(buf, pretok):
    """Last safe cut position in buf, 0 if none. Searches backwards from the nearest candidate byte (fast)."""
    anchor = {"metaspace": b" ", "char_delimiter": b" "}.get(pretok, b"\n")
    if pretok in ("punctuation", "digits", "unicode_scripts"):   # many possible anchors: scan back byte by byte
        for c in range(len(buf) - 1, 0, -1):
            if safe_cut_at(pretok, buf, c):
                return c
        return 0
    i = buf.rfind(anchor)
    while i >= 0:
        for c in (i, i + 1):                    # the anchor byte is at c (metaspace, unicode_scripts) or c-1
            if safe_cut_at(pretok, buf, c):
                return c
        i = buf.rfind(anchor, 0, i)
    return 0


def pieces(path, pretok="bytelevel"):
    """Yield the file as str pieces whose concatenation is the whole file."""
    rest = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(PIECE)
            buf = rest + block
            if not block:
                if buf:
                    yield buf.decode("utf-8", errors="replace")
                return
            c = safe_cut(buf, pretok)
            if c == 0:              # no safe point yet: read more
                rest = buf
                continue
            yield buf[:c].decode("utf-8", errors="replace")
            rest = buf[c:]
