#!/bin/bash
# Benchmark: HF tokenizers vs ours (rust_bpe_py, simd + nosimd builds). Train, then encode, for every config.
# Plus OpenAI tiktoken, encode only (it can't train), with the vocab HF trained.
#
#   ./run.sh <data_dir> <results_dir>
#
# data_dir has <dataset>_<size>GB.txt files (made by prepare_data.py), e.g. code_1GB.txt, mix_10GB.txt.
# Everything runs sequentially. Each step is ONE python process wrapped in /usr/bin/time, which gives:
#   wall time (s) and peak memory (max RSS, KB -> MB) of that process.
# The python script also prints its own inner time (train_s / encode_s: only the train/encode call).
# One line per step goes to <results_dir>/results.tsv; the files each step wrote go to <results_dir>/<run>/<step>/.
# Check equivalence afterwards with: python3 verify.py <results_dir>
# Token ids are written to RAM (/dev/shm) and replaced by their sha256 (ids.sha256); KEEP_IDS=1 keeps the ids.u32 files.
# STEP_TIMEOUT=<seconds> stops a step that runs longer (status "timeout"); default: no limit.
set -u
cd "$(dirname "$0")"
DATA=${1:?data dir}
RES=${2:?results dir}
ARCH=$(uname -m)
PY=${PY:-venv-$ARCH/bin/python}
mkdir -p "$RES"
SHM=/dev/shm/tok_bench_$USER              # encode output (token ids) goes to RAM, then only its checksum is kept
mkdir -p "$SHM"

DATASETS=${DATASETS:-"english multilingual code mix"}
SIZES=${SIZES:-"1 10"}                 # GB (10^9 bytes)
IMPLS=${IMPLS:-"hf ours_simd ours_nosimd"}

# Baseline, and the values tried for each axis (one axis changes at a time; the others stay at baseline)
BASE_THREADS=${BASE_THREADS:-8};         ALL_THREADS=${ALL_THREADS-"2 4 8 16"}
BASE_PRETOK=${BASE_PRETOK:-bytelevel};  ALL_PRETOKS=${ALL_PRETOKS-"bytelevel llama3 gpt4o whitespace whitespace_split metaspace bert punctuation digits unicode_scripts char_delimiter"}
BASE_VOCAB=${BASE_VOCAB:-131072};       ALL_VOCABS=${ALL_VOCABS-"32768 65536 131072 262144"}
BASE_NORM=${BASE_NORM:-none};         ALL_NORMS=${ALL_NORMS-"none nfc nfd nfkc nfkd lowercase strip_accents nfkc_lowercase replace bert prepend strip"}

TSV=$RES/results.tsv
[ -f "$TSV" ] || echo -e "dataset\tsize_GB\timpl\tthreads\tpretok\tvocab\tnorm\tstep\tstatus\twall_s\tinner_s\tpeak_rss_MB\ttokens\tdir" > "$TSV"

# step <impl> <step: train|encode> <corpus> <threads> <pretok> <vocab> <norm> <run_dir>
step() {
    local impl=$1 st=$2 corpus=$3 t=$4 pretok=$5 vocab=$6 norm=$7 run=$8
    local dir=$run/${st}_$impl
    local train_dir=$run/train_$impl                      # encode uses the tokenizer this impl trained
    [ "$impl" = tiktoken ] && train_dir=$run/train_hf     # tiktoken can't train: it encodes with HF's vocab
    mkdir -p "$dir"
    if [ -f "$dir/ok" ]; then echo "skip (ok): $dir"; return 0; fi     # finished in an earlier, interrupted run

    # 1. which python script, and which threads / build env vars
    local script env
    case $impl in
        hf)          script=${st}_hf.py;   env="RAYON_NUM_THREADS=$t TOKENIZERS_PARALLELISM=true" ;;
        ours_simd)   script=${st}_ours.py; env="BB_THREADS=$t HB_THREADS=$t PYTHONPATH=../rust_bpe_py/build/$ARCH/simd" ;;
        ours_nosimd) script=${st}_ours.py; env="BB_THREADS=$t HB_THREADS=$t PYTHONPATH=../rust_bpe_py/build/$ARCH/nosimd" ;;
        tiktoken)    script=encode_tiktoken.py; env="RAYON_NUM_THREADS=$t" ;;   # threads passed as an argument below
    esac
    if [ "$st" = train ]; then args="$corpus $pretok $vocab $dir $norm"
    elif [ "$impl" = tiktoken ]; then args="$train_dir/tokenizer.json $corpus $pretok $t $dir"
    else args="$train_dir/tokenizer.json $corpus $dir $pretok"; fi

    # 2. run it under /usr/bin/time: %e = wall seconds, %M = peak RSS in KB
    #    encode writes its ids to RAM (/dev/shm); afterwards only their sha256 is kept, in ids.sha256
    local ids=$SHM/ids.u32
    echo "$(date +%T) $impl $st $(basename "$corpus") t=$t pretok=$pretok vocab=$vocab norm=$norm"
    env $env IDS_PATH=$ids timeout ${STEP_TIMEOUT:-0} /usr/bin/time -f "%e %M" -o "$dir/time.txt" $PY $script $args > "$dir/log.txt" 2>&1
    local rc=$?
    [ $rc -eq 124 ] && echo "${STEP_TIMEOUT} 0" > "$dir/time.txt"     # timed out: wall = the limit, memory unknown
    if [ -f "$ids" ]; then
        sha256sum < "$ids" | cut -d" " -f1 > "$dir/ids.sha256"
        # byte-by-byte check: HF's ids stay in RAM until the other impls of this config are compared against them
        if [ "$impl" = hf ]; then mv "$ids" "$SHM/hf_ids.u32"
        elif [ -f "$SHM/hf_ids.u32" ]; then
            $PY verify.py "$SHM/hf_ids.u32" "$ids" > "$dir/ids.bytecmp"      # token by token, see verify.py compare()
        fi
        if [ -n "${KEEP_IDS:-}" ] && [ -f "$ids" ]; then mv "$ids" "$dir/ids.u32"; else rm -f "$ids"; fi
    fi

    # 3. collect metrics: wall + peak RSS from time.txt; inner time + tokens from the script's log
    local wall rss inner tokens status=ok
    read wall rss < <(tail -1 "$dir/time.txt")
    inner=$(grep -o "${st}_s=[0-9.]*" "$dir/log.txt" | tail -1 | cut -d= -f2)
    tokens=$(grep -o "tokens=[0-9]*" "$dir/log.txt" | tail -1 | cut -d= -f2)
    [ $rc -ne 0 ] && status="failed_rc$rc"
    [ $rc -eq 124 ] && status="timeout"
    echo -e "$(basename "$corpus" .txt | sed 's/_[0-9.]*GB$//')\t$SIZE\t$impl\t$t\t$pretok\t$vocab\t$norm\t$st\t$status\t$wall\t${inner:-}\t$((rss / 1024))\t${tokens:-}\t$dir" >> "$TSV"
    echo "    -> $status wall=${wall}s inner=${inner:-?}s peak=$((rss / 1024))MB"
    [ $rc -eq 0 ] && touch "$dir/ok"
    return $rc
}

# config <corpus> <threads> <pretok> <vocab> <norm>: train + encode with every impl that supports this pretok,
# then encode with tiktoken (byte-level pre-tokenizers, no normalizer), using HF's trained vocab.
# Normalizers that need the whole text (prepend, strip) are train-only.
config() {
    local corpus=$1 t=$2 pretok=$3 vocab=$4 norm=$5
    local run=$RES/$(basename "$corpus" .txt)/t${t}_${pretok}_v${vocab}
    [ "$norm" != none ] && run=${run}_n$norm
    local enc=1
    $PY -c "import normalizers, sys; sys.exit('$norm' in normalizers.TRAIN_ONLY)" || enc=0
    if [ -f "$run/done" ]; then echo "skip (done): $run"; return; fi
    rm -f "$SHM/hf_ids.u32"
    for impl in $IMPLS; do
        local who=${impl%%_*}                                  # hf | ours
        if $PY pretokenizers.py "$who" train "$pretok"; then
            step "$impl" train "$corpus" "$t" "$pretok" "$vocab" "$norm" "$run" || continue
            if [ $enc = 1 ] && $PY pretokenizers.py "$who" encode "$pretok"; then
                step "$impl" encode "$corpus" "$t" "$pretok" "$vocab" "$norm" "$run"
            fi
        fi
    done
    if [ "$norm" = none ] && [ -f "$run/train_hf/tokenizer.json" ] && $PY pretokenizers.py tiktoken encode "$pretok"; then
        step tiktoken encode "$corpus" "$t" "$pretok" "$vocab" "$norm" "$run"
    fi
    [ -n "${KEEP_IDS:-}" ] && [ -f "$SHM/hf_ids.u32" ] && mv "$SHM/hf_ids.u32" "$run/encode_hf/ids.u32"
    rm -f "$SHM/hf_ids.u32"
    touch "$run/done"
}

for SIZE in $SIZES; do
    for ds in $DATASETS; do
        corpus=$DATA/${ds}_${SIZE}GB.txt
        [ -f "$corpus" ] || { echo "missing $corpus, skipping"; continue; }
        config "$corpus" $BASE_THREADS $BASE_PRETOK $BASE_VOCAB $BASE_NORM    # baseline
        for t in $ALL_THREADS;  do config "$corpus" $t $BASE_PRETOK $BASE_VOCAB $BASE_NORM; done
        for p in $ALL_PRETOKS;  do config "$corpus" $BASE_THREADS $p $BASE_VOCAB $BASE_NORM; done
        for v in $ALL_VOCABS;   do config "$corpus" $BASE_THREADS $BASE_PRETOK $v $BASE_NORM; done
        for n in $ALL_NORMS;    do config "$corpus" $BASE_THREADS $BASE_PRETOK $BASE_VOCAB $n; done
    done
done
echo "done. results: $TSV   verify: $PY verify.py $RES"
