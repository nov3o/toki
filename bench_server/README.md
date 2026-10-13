# bench_server — HF tokenizers vs ours (vs tiktoken), time + memory, then equivalence

Compares BPE **training** (HF `tokenizers` vs ours) and **encoding** (HF vs OpenAI `tiktoken` vs ours) on the same
data, one run at a time, and afterwards checks that every merge step, the final vocab and every token id are identical.

## How it runs (call chain)

```
setup.sh                    venv-<arch> + ../rust_bpe_py/build.sh (our binding, twice: build/<arch>/simd, build/<arch>/nosimd)
prepare_data.py             100 GB pool per source -> random 1 GB / 10 GB samples + a mix
run.sh <data> <results>     loops over datasets x sizes x configs, strictly one step at a time
  └─ for each config (baseline, then one axis changed at a time), for each impl (hf, ours_simd, ours_nosimd):
      /usr/bin/time -f "%e %M" python train_<hf|ours>.py  <corpus> <pretok> <vocab> <dir> <norm>  -> wall s, peak RSS
      /usr/bin/time -f "%e %M" python encode_<hf|ours>.py <tokenizer.json> <corpus> <dir> <pretok> -> wall s, peak RSS
     then tiktoken (byte-level pre-tokenizers, no normalizer) encodes with HF's vocab
      one line per step -> results/results.tsv
verify.py <results>         ours / tiktoken vs hf: merges.txt, vocab.txt, merges_used.txt, ids.sha256 -> verify.tsv
```

| file | what |
|---|---|
| `setup.sh` | venv (tokenizers, tiktoken, pyarrow, huggingface_hub) + binding builds |
| `prepare_data.py` | pool of 10x the largest size per source, random document samples, mix |
| `pretokenizers.py` | the 11 pre-tokenizers by name (HF objects; all supported by ours) |
| `normalizers.py` | the normalizers by name (HF objects + the spec string for ours) |
| `train_hf.py` / `train_ours.py` | train; write `tokenizer.json`, `merges.txt` (one line per merge step), `vocab.txt` (final vocab) |
| `encode_hf.py` / `encode_ours.py` / `encode_tiktoken.py` | encode the whole file; ids go to RAM, `ids.sha256` is kept |
| `common.py` | shared helpers; `pieces()` cuts files where pre-tokenization restarts (same rules as ours) |
| `export_hfbp.py` | tokenizer.json -> our flat model file (`.hfbp`) |
| `run.sh` | the loops and the metric collection (read this first) |
| `verify.py` | equivalence check, after the benchmark |

**Axes** (env vars; empty = baseline only). Baseline: 8 threads, `bytelevel` (GPT-2), vocab 131072, no normalizer.

| env | values |
|---|---|
| `ALL_THREADS` | 2 4 8 16 |
| `ALL_PRETOKS` | bytelevel llama3 gpt4o whitespace whitespace_split metaspace bert punctuation digits unicode_scripts char_delimiter |
| `ALL_VOCABS` | 32768 65536 131072 262144 |
| `ALL_NORMS` | none nfc nfd nfkc nfkd lowercase strip_accents nfkc_lowercase replace bert prepend strip (prepend/strip: train only) |
| `DATASETS`, `SIZES`, `IMPLS` | english multilingual code mix; 1 10 (GB = 10^9 bytes); hf ours_simd ours_nosimd |

Threads: HF reads `RAYON_NUM_THREADS`; ours `BB_THREADS` (train) / `HB_THREADS` (encode); tiktoken gets them as an argument.
SIMD or not: `PYTHONPATH` points at `../rust_bpe_py/build/<arch>/simd` or `.../nosimd`.
Token ids are written to `/dev/shm` and replaced by their sha256 (`KEEP_IDS=1` keeps `ids.u32`).

## Exactness

Ours reproduces HF bit for bit: same merge order and tie-break, and every pre-tokenizer / normalizer uses the same
character rules as HF (Rust std, `unicode_categories`, `unicode-normalization-alignments`, HF's script table, the
`regex` crate's `\w`/`\s`). Long files are encoded in chunks cut only where the pre-tokenizer (and normalizer)
provably restarts (`rust_bpe/src/pretok.rs: safe_cut`); HF and tiktoken get pieces cut by the same rules.

## On bristen (x86, AVX2; quiet login node)

Run setup.sh on the machine that runs the benchmark: the simd build targets the CPU it is compiled on.

```bash
cd $SCRATCH/tok_bench/bench_server
PYTHON=python3.11 ./setup.sh
venv-x86_64/bin/python prepare_data.py data 100 1 10
SIZES=1 ./run.sh data results
venv-x86_64/bin/python verify.py results
```

Results: `results/results.tsv`, `results/verify.tsv`, per-step files in `results/<dataset>_<size>GB/<config>/<step>_<impl>/`.
If run.sh is interrupted, start it again: finished configs are skipped. Results so far: RESULTS.md.

## Smoke test on a small machine (3 MB, baseline only)

```bash
head -c 3000000 data/code_1GB.txt > data/code_0.003GB.txt
ALL_THREADS= ALL_PRETOKS= ALL_VOCABS= ALL_NORMS= DATASETS=code SIZES=0.003 ./run.sh data results_test
```
