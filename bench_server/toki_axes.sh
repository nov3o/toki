#!/bin/bash
# toki_axes.sh <results_dir>: the toki tables. Default setting (1 GB, 8 threads, GPT-2 byte-level, vocab 131072, no
# normalizer), then one axis at a time: threads 2/4/16, vocab 32k/64k/256k, pre-tokenizer, normalizer, size 10 GB.
# 4 datasets. Training: PR #2501 vs ours on the whole sample. Encoding: HF main vs ours on the held-out test half, with
# the vocab trained on the train half (results_enc_ho3/vocab). Every run pinned to exactly <threads> cores from
# BASE (0); order baseline, ours, ours, baseline. Merges compared with PR #2501's, ids with HF main's.
# ours: ../ours/build.sh (or set OURS_TRAIN / OURS_ENCODE)
OURS_TRAIN=${OURS_TRAIN:-../ours/train/target/release/toki-train}; OURS_ENCODE=${OURS_ENCODE:-../ours/encode/target/release/toki-encode}
cd "$(dirname "$0")"; RES=$1; mkdir -p $RES; BASE=${BASE:-0}; P=/dev/shm/prof_$USER/ta; mkdir -p $P
TSV=$RES/results.tsv; PY=venv-x86_64/bin/python
[ -f $TSV ] || echo -e "task\taxis\tdataset\tsize\tthreads\tpretok\tvocab\tnorm\timpl\trun\twall_s\tpeak_rss_KB\tcheck" > $TSV
run() {  # run <task> <axis> <ds> <size> <threads> <pretok> <vocab> <norm> <impl> <run>
    local task=$1 ds=$3 size=$4 t=$5 pt=$6 v=$7 nm=$8 impl=$9 r=${10}
    grep -qP "^$task\t$2\t$ds\t$size\t$t\t$pt\t$v\t$nm\t$impl\t$r\t" $TSV && return
    local tok=results_enc_ho3/vocab/${ds}_${pt}_v$( [ $task = train ] && echo 131072 || echo $v )_${nm}/tokenizer.json
    [ $task = train ] && [ ! -f $tok ] && tok=results_enc_ho3/vocab/${ds}_${pt}_v131072_none/tokenizer.json
    local bl=0 sp="[UNK]"; case $pt in bytelevel|llama3|gpt4o) bl=1; sp="";; esac
    local spec; spec=$($PY -c "import normalizers; print(normalizers.spec('$nm'))")
    local C=data_small/${ds}_1GB.txt T=data_small/${ds}_test.txt
    [ $size = 10GB ] && C=$SCRATCH/tok_bench/bench_server/data/${ds}_10GB.txt && T=$C
    local o=$P/$impl; mkdir -p $o; : > $P/$impl.u32
    case $task/$impl in
        train/pr2501) cmd="hfbench2-pr2501/target/release/hfbench $tok $C $v $o $bl $sp" ;;
        train/ours)   cmd="env HB_PRETOK=$pt HB_THREADS=$t MIMALLOC_PURGE_DELAY=0 $OURS_TRAIN $C $v $o" ;;
        encode/hfmain) cmd="../hfenc-thr/release/hfenc $tok $T $pt $P/$impl.u32" ;;
        encode/ours)  cmd="env HB_THREADS=$t $OURS_ENCODE $tok $T $pt $P/$impl.u32" ;;
    esac
    HB_NORMALIZER="$spec" timeout 7200 taskset -c $BASE-$((BASE + t - 1)) /usr/bin/time -f "%e %M" -o $P/tm $cmd > $P/log 2>&1
    local c
    case $task/$impl in
        train/pr2501) cp $o/merges.txt $P/ref_merges.txt; c=ref ;;
        train/ours) cmp -s $o/merges.txt $P/ref_merges.txt && c=same || c=DIFFERENT ;;
        encode/hfmain) cp $P/$impl.u32 $P/ref.u32; c="ref ($(( $(stat -c%s $P/ref.u32) / 4 )) ids)" ;;
        encode/ours) [ -s $P/ref.u32 ] && cmp -s $P/ours.u32 $P/ref.u32 && c="same ($(( $(stat -c%s $P/ours.u32) / 4 )) ids)" || c=DIFFERENT ;;
    esac
    echo -e "$task\t$2\t$ds\t$size\t$t\t$pt\t$v\t$nm\t$impl\t$r\t$(tail -1 $P/tm | tr ' ' '\t')\t$c" >> $TSV
}
setting() {  # setting <axis> <ds> <size> <threads> <pretok> <vocab> <norm>
    run train "$@" pr2501 1; run train "$@" ours 1; run train "$@" ours 2; run train "$@" pr2501 2
    run encode "$@" hfmain 1; run encode "$@" ours 1; run encode "$@" ours 2; run encode "$@" hfmain 2
}
for ds in english multilingual code mix; do
    setting default $ds 1GB 8 bytelevel 131072 none
    for t in 2 4 16; do setting threads $ds 1GB $t bytelevel 131072 none; done
    for v in 32768 65536 262144; do setting vocab $ds 1GB 8 bytelevel $v none; done
    for pt in llama3 gpt4o whitespace whitespace_split metaspace bert punctuation digits unicode_scripts char_delimiter; do
        setting pretok $ds 1GB 8 $pt 131072 none; done
    for nm in nfc nfd nfkc nfkd lowercase strip_accents nfkc_lowercase replace bert; do setting norm $ds 1GB 8 bytelevel 131072 $nm; done
done
for ds in english multilingual code mix; do setting size $ds 10GB 8 bytelevel 131072 none; done
rm -f $P/*.u32; echo done
