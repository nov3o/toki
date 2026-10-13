#!/bin/bash
# plots_bench.sh <results_dir>: the measurements behind PR #2501's three figures, for HF main, PR #2501 and ours.
#   1. throughput vs cores (1 2 4 6 8), 256 MiB per workload, vocab 100 000, 3 runs each: feed + train and train only
#   2. memory growth: peak RSS vs input size (16 MiB .. 1 GiB), 8 cores, one run each
# Workloads: code / byte-level, english / byte-level, multilingual / byte-level, multilingual / whitespace (PR #2501 used
# Chinese; our multilingual set is 7 languages). HF main and PR #2501 train through hfbench2 (HF's own pre-tokenizer,
# lines fed with Trainer::feed); ours = our counting + PR #2501's merge loop. Every run pinned to cores BASE.. (16).
# Merges of every run compared with PR #2501's.
cd "$(dirname "$0")"; RES=$1; mkdir -p $RES; BASE=${BASE:-16}; P=/dev/shm/prof_$USER/plots; mkdir -p $P
TSV=$RES/results.tsv; [ -f $TSV ] || echo -e "figure\tworkload\tmib\tcores\timpl\trun\twall_s\tpeak_rss_KB\tfeed_s\ttrain_s\tmerges_vs_pr2501" > $TSV
PY=venv-x86_64/bin/python; V=100000
one() {  # one <figure> <workload> <mib> <cores> <impl> <run>
    local ds=${2%/*} pt=${2#*/}; local c=$P/${ds}_$3MiB.txt tok=results_enc_ho3/vocab/${ds}_${pt}_v131072_none/tokenizer.json
    [ -f $c ] || head -c $(( $3 * 1048576 )) data_small/${ds}_1GB.txt > $c
    local bl=0 sp="[UNK]"; [ $pt = bytelevel ] && bl=1 && sp=""
    local o=$P/out_$5; mkdir -p $o
    case $5 in
        pr2501) cmd="hfbench2-pr2501/target/release/hfbench $tok $c $V $o $bl $sp" ;;
        hfmain) cmd="hfbench2-main/target/release/hfbench $tok $c $V $o $bl $sp" ;;
        ours)   cmd="env HB_PRETOK=$pt HB_THREADS=$4 MIMALLOC_PURGE_DELAY=0 ../union-pr2501/target/release/trunion $c $V $o" ;;
    esac
    timeout 7200 taskset -c $BASE-$((BASE + $4 - 1)) /usr/bin/time -f "%e %M" -o $P/tm $cmd > $P/log 2>&1
    local f t r; f=$(grep -o "feed_s=[0-9.]*" $P/log | cut -d= -f2); t=$(grep -o "train_s=[0-9.]*" $P/log | cut -d= -f2)
    if [ $5 = pr2501 ]; then cp $o/merges.txt $P/ref.txt; r=ref; elif cmp -s $o/merges.txt $P/ref.txt; then r=same; else r=DIFFERENT; fi
    echo -e "$1\t$2\t$3\t$4\t$5\t$6\t$(tail -1 $P/tm | tr ' ' '\t')\t${f:-NA}\t${t:-NA}\t$r" >> $TSV
}
W="code/bytelevel english/bytelevel multilingual/bytelevel multilingual/whitespace"
for w in $W; do for cores in 1 2 4 6 8; do for run in 1 2 3; do
    for impl in pr2501 hfmain ours; do one throughput $w 256 $cores $impl $run; done
done; done; done
for w in $W; do for mib in 16 32 64 128 256 512 1024; do
    for impl in pr2501 hfmain ours; do one memory $w $mib 8 $impl 1; done
done; done
rm -rf $P; echo done
