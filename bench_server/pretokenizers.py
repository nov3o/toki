"""Every pre-tokenizer we benchmark, by name.

  make(name)  -> the HF `tokenizers` pre-tokenizer object
  OURS[name]  -> the same pre-tokenizer in our Rust code (BB_PRETOK), if we have it

Byte-level ones ("bytelevel", "llama3", "gpt4o") turn text into bytes first, so there are no unknown characters.
The rest split on characters, so BPE runs over raw Unicode characters (with an [UNK] token).
"""
from tokenizers import Regex, pre_tokenizers as pt

# GPT-2 regex (what pre_tokenizers.ByteLevel uses internally)
GPT2_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# GPT-4o regex (tiktoken o200k_base; also what Mistral's Tekken uses), copied from tiktoken_ext/openai_public.py
GPT4O_REGEX = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])
LLAMA3_REGEX = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"""
                r"""|\s*[\r\n]+|\s+(?!\S)|\s+""")

ALL = ["bytelevel", "llama3", "gpt4o", "whitespace", "whitespace_split", "metaspace", "bert",
       "punctuation", "digits", "unicode_scripts", "char_delimiter"]
BYTE_LEVEL = {"bytelevel", "llama3", "gpt4o"}
REGEX = {"bytelevel": GPT2_REGEX, "llama3": LLAMA3_REGEX, "gpt4o": GPT4O_REGEX}   # for tiktoken

# Our trainer and encoder support all of them (same names, rust_bpe/src/pretok.rs).
OURS = {name: name for name in ALL}


def make(name):
    if name == "bytelevel":          # GPT-2 regex, then bytes
        return pt.ByteLevel(add_prefix_space=False)
    if name == "llama3":             # Llama-3 regex, then bytes
        return pt.Sequence([pt.Split(Regex(LLAMA3_REGEX), behavior="isolated"),
                            pt.ByteLevel(add_prefix_space=False, use_regex=False)])
    if name == "gpt4o":              # GPT-4o / Tekken regex, then bytes
        return pt.Sequence([pt.Split(Regex(GPT4O_REGEX), behavior="isolated"),
                            pt.ByteLevel(add_prefix_space=False, use_regex=False)])
    if name == "whitespace":         # \w+|[^\w\s]+
        return pt.Whitespace()
    if name == "whitespace_split":   # split on whitespace only
        return pt.WhitespaceSplit()
    if name == "metaspace":          # SentencePiece style, space -> ▁
        return pt.Metaspace()
    if name == "bert":               # whitespace + punctuation
        return pt.BertPreTokenizer()
    if name == "punctuation":
        return pt.Punctuation()
    if name == "digits":
        return pt.Digits(individual_digits=True)
    if name == "unicode_scripts":
        return pt.UnicodeScripts()
    if name == "char_delimiter":
        return pt.CharDelimiterSplit(" ")
    raise SystemExit(f"unknown pre-tokenizer: {name}")


if __name__ == "__main__":           # used by run.sh: is <pretok> supported by <impl> <stage>?
    import sys
    impl, stage, name = sys.argv[1:4]
    if impl == "tiktoken":
        ok = stage == "encode" and name in BYTE_LEVEL
    else:
        ok = impl == "hf" or name in OURS
    sys.exit(0 if ok else 1)
