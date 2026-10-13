"""Export a HuggingFace byte-level BPE tokenizer.json to the flat HFBP format
read by hf_bpe_enc.

  python3 export_hfbp.py <tokenizer.json> <out.hfbp>
  python3 export_hfbp.py --gpt2-bench <vocab.json> <merges.txt> <out.hfbp> [<saved.json>]

--gpt2-bench builds the exact tokenizer of HF's bpe_benchmark.rs /
ci_benchmark.rs: GPT-2 vocab+merges, ByteLevel::default() (add_prefix_space),
added token "ing" (non-special) and special token "[ENT]" (single_word).

Format (little endian):
  "HFBP" u32 version=3
  u32 pretok (0 = GPT-2 regex, 1 = Llama-3 regex, 2 = GPT-4o regex)
  u8 add_prefix_space, u8 ignore_merges, u16 0
  u32 n_vocab, u32 n_merges
  n_vocab x (u16 len, raw bytes)            -- id -> byte string (len 0 = unused id)
  n_merges x (u32 left, u32 right, u32 merged) in rank order
  u32 n_added, then n_added x (u32 id, u8 special, u8 single_word, u8 lstrip,
                               u8 rstrip, u8 normalized, u16 len, utf-8 bytes)
  u32 unk id (0xFFFFFFFF = none)                                         -- version 2
  u32 len, normalizer spec (rust_bpe/src/normalize.rs; empty = none)     -- version 3
pretok 10..17 = character-level pre-tokenizers (whitespace, whitespace_split, metaspace, bert, punctuation,
digits, unicode_scripts, char_delimiter); their vocab strings are raw UTF-8.
"""
import json
import struct
import sys


def byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def pretok_kind(pre):
    if pre is None:
        raise SystemExit("no pre_tokenizer")
    if pre["type"] == "ByteLevel":
        if not pre.get("use_regex", True):
            raise SystemExit("ByteLevel without regex is not supported")
        return 0, pre.get("add_prefix_space", True)
    if pre["type"] == "Sequence":
        subs = pre["pretokenizers"]
        split = [p for p in subs if p["type"] == "Split"]
        bl = [p for p in subs if p["type"] == "ByteLevel"]
        if len(split) == 1 and len(bl) == 1 and not bl[0].get("use_regex", True) and split[0]["behavior"] == "Isolated":
            rx = split[0]["pattern"].get("Regex", "")
            if "\\p{Lu}" in rx and "\\p{N}{1,3}" in rx:      # GPT-4o / o200k
                return 2, bl[0].get("add_prefix_space", False)
            if "\\p{N}{1,3}" in rx:                            # Llama-3
                return 1, bl[0].get("add_prefix_space", False)
    # character-level pre-tokenizers (codes = rust_bpe/src/pretok.rs from_code), with the options HF uses by default
    t = pre["type"]
    if t == "Metaspace" and (pre.get("replacement") != "\u2581" or pre.get("prepend_scheme") != "always" or not pre.get("split", True)):
        raise SystemExit("Metaspace: only replacement='▁', prepend_scheme='always', split=True")
    if t == "Digits" and not pre.get("individual_digits"):
        raise SystemExit("Digits: only individual_digits=True")
    if t == "CharDelimiterSplit" and pre.get("delimiter") != " ":
        raise SystemExit("CharDelimiterSplit: only delimiter=' '")
    if t == "Punctuation" and pre.get("behavior", "Isolated") != "Isolated":
        raise SystemExit("Punctuation: only behavior='isolated'")
    codes = {"Whitespace": 10, "WhitespaceSplit": 11, "Metaspace": 12, "BertPreTokenizer": 13, "Punctuation": 14,
             "Digits": 15, "UnicodeScripts": 16, "CharDelimiterSplit": 17}
    if t in codes:
        return codes[t], False
    raise SystemExit(f"unsupported pre_tokenizer: {json.dumps(pre)[:200]}")


def normalizer_spec(n):
    """HF normalizer JSON -> our spec string (rust_bpe/src/normalize.rs). Steps that need the whole text
    (Prepend, Strip) are refused: the encoder cuts long files into chunks."""
    US, RS = "\x1f", "\x1e"
    if n is None:
        return ""
    t = n["type"]
    if t == "Sequence":
        return RS.join(x for x in (normalizer_spec(m) for m in n["normalizers"]) if x)
    simple = {"NFC": "nfc", "NFD": "nfd", "NFKC": "nfkc", "NFKD": "nfkd", "Lowercase": "lowercase", "StripAccents": "strip_accents"}
    if t in simple:
        return simple[t]
    if t == "Replace":
        if "String" not in n["pattern"]:
            raise SystemExit("Replace: only literal string patterns")
        return "replace" + US + n["pattern"]["String"] + US + n["content"]
    if t == "BertNormalizer":
        sa = n.get("strip_accents")
        sa = n.get("lowercase", True) if sa is None else sa
        return US.join(["bert"] + ["1" if x else "0" for x in (n.get("clean_text", True), n.get("handle_chinese_chars", True), sa, n.get("lowercase", True))])
    raise SystemExit(f"unsupported normalizer for chunked encoding: {t}")


def export(d, out):
    U = byte_decoder()
    m = d["model"]
    assert m["type"] == "BPE"
    norm = normalizer_spec(d.get("normalizer"))
    kind, prefix = pretok_kind(d.get("pre_tokenizer"))
    char_level = kind >= 10                      # vocab strings are raw text, not byte-level characters
    vocab = m["vocab"]
    added = d.get("added_tokens", [])
    n_vocab = max(max(vocab.values()), max((a["id"] for a in added), default=0)) + 1
    idb = [b""] * n_vocab
    for t, i in vocab.items():
        idb[i] = t.encode("utf-8") if char_level else bytes(U[c] for c in t)
    merges = []
    for x in m["merges"]:
        a, b = x if isinstance(x, list) else x.split(" ", 1)
        merges.append((vocab[a], vocab[b], vocab[a + b]))
    for a in added:
        assert "\n" not in a["content"], "added tokens containing newlines break safe cut points"
    with open(out, "wb") as f:
        f.write(b"HFBP" + struct.pack("<IIBBHII", 3, kind, int(prefix), int(bool(m.get("ignore_merges"))), 0, n_vocab, len(merges)))
        for b in idb:
            f.write(struct.pack("<H", len(b)) + b)
        for t in merges:
            f.write(struct.pack("<III", *t))
        f.write(struct.pack("<I", len(added)))
        for a in added:
            c = a["content"].encode()
            f.write(struct.pack("<IBBBBBH", a["id"], a["special"], a["single_word"], a["lstrip"], a["rstrip"], a["normalized"], len(c)) + c)
        unk = m.get("unk_token")
        f.write(struct.pack("<I", vocab[unk] if unk is not None else 0xFFFFFFFF))   # version 2: unk id
        nb = norm.encode("utf-8")
        f.write(struct.pack("<I", len(nb)) + nb)                                     # version 3: normalizer spec
    print(f"{out}: pretok={kind} add_prefix_space={prefix} ignore_merges={bool(m.get('ignore_merges'))} "
          f"vocab={n_vocab} merges={len(merges)} added={len(added)}")


def gpt2_bench_tokenizer(vocab, merges):
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    tk = Tokenizer(models.BPE.from_file(vocab, merges))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel()
    tk.decoder = decoders.ByteLevel()
    tk.add_tokens([AddedToken("ing", single_word=False)])
    tk.add_special_tokens([AddedToken("[ENT]", single_word=True)])
    return tk


if __name__ == "__main__":
    if sys.argv[1] == "--gpt2-bench":
        tk = gpt2_bench_tokenizer(sys.argv[2], sys.argv[3])
        if len(sys.argv) > 5:
            tk.save(sys.argv[5])
        export(json.loads(tk.to_str()), sys.argv[4])
    else:
        export(json.load(open(sys.argv[1])), sys.argv[2])
