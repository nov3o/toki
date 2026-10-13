#!/bin/bash
# recheck_ids.sh <results_dir> [threads] [run dirs...]: re-encode every finished config (or only the given ones) (ids only, not timed) and compare
# ours / tiktoken against HF byte by byte; writes ids.bytecmp next to each encode dir, then run verify.py.
RES=$1; T=${2:-8}; ARCH=$(uname -m); PY=venv-$ARCH/bin/python; SHM=/dev/shm/tok_recheck_$USER; mkdir -p $SHM
shift 2 2>/dev/null; RUNS=${*:-$(find "$RES" -type d -name encode_hf | xargs -n1 dirname | sort)}
for run in $RUNS; do
    name=$(basename "$run"); ds=$(basename "$(dirname "$run")")              # t8_<pretok>_v<vocab>[_n<norm>], <dataset>_<size>GB
    pretok=$(echo "$name" | sed -E 's/^t[0-9]+_(.*)_v[0-9]+(_n.*)?$/\1/')
    corpus=data_small/$ds.txt; [ -f "$corpus" ] || corpus=data/$ds.txt
    rm -f $SHM/*.u32; mkdir -p $SHM/o
    RAYON_NUM_THREADS=$T TOKENIZERS_PARALLELISM=true IDS_PATH=$SHM/hf.u32 $PY encode_hf.py "$run/train_hf/tokenizer.json" "$corpus" $SHM/o "$pretok" > $SHM/log 2>&1
    for d in "$run"/encode_*; do
        impl=${d##*/encode_}; [ "$impl" = hf ] && continue
        case $impl in
            ours_simd|ours_nosimd) HB_THREADS=$T PYTHONPATH=../rust_bpe_py/build/$ARCH/${impl#ours_} IDS_PATH=$SHM/b.u32 \
                $PY encode_ours.py "$run/train_$impl/tokenizer.json" "$corpus" $SHM/o "$pretok" >/dev/null 2>&1 ;;
            tiktoken) IDS_PATH=$SHM/b.u32 $PY encode_tiktoken.py "$run/train_hf/tokenizer.json" "$corpus" "$pretok" $T $SHM/o >/dev/null 2>&1 ;;
        esac
        r=$($PY verify.py $SHM/hf.u32 $SHM/b.u32); echo "$r" > "$d/ids.bytecmp"
        echo "$(date +%T) $ds/$name $impl $r"
        rm -f $SHM/b.u32
    done
done
rm -rf $SHM
