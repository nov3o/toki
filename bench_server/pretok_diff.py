"""pretok_diff.py <tokenizer.json> <corpus> <dump from hfbench HB_DUMP=1>: first line where HF 0.22's pre-tokens
(its normalizer, then its pre-tokenizer, on each line with its '\\n') differ from another build's dump."""
import json, sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1])
other = {}
for l in open(sys.argv[3], encoding="utf-8"):
    k, w = l.rstrip("\n").split("\t", 1)
    other.setdefault(int(k), []).append(json.loads(w))
text = open(sys.argv[2], encoding="utf-8").read()
for k, line in enumerate(text.splitlines(keepends=True)):
    s = tk.normalizer.normalize_str(line) if tk.normalizer else line
    ref = [w for w, _ in tk.pre_tokenizer.pre_tokenize_str(s)] if tk.pre_tokenizer else [s]
    if ref != other.get(k, []):
        o = other.get(k, [])
        i = next((i for i, (a, b) in enumerate(zip(ref, o)) if a != b), min(len(ref), len(o)))
        print(f"line {k}: first differing pre-token {i}\n  0.22: {ref[max(0, i-3):i+4]}\n  main: {o[max(0, i-3):i+4]}")
        break
else:
    print("all lines identical")
