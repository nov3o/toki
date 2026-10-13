#!/bin/bash
# enc_10gb_union.sh: union encoder vs HF main, 4 datasets x 10 GB, 16 and 64 cores (cores 64..), ids compared byte by byte.
cd $SCRATCH/tok_bench_v3/bench_server; P=/dev/shm/prof_$USER; mkdir -p $P/sh
echo "dataset cores impl wall_s rss_KB ids"
for ds in english multilingual code mix; do
    tok=results_enc_ho3/vocab/${ds}_bytelevel_v131072_none/tokenizer.json; T=$P/${ds}10.txt
    [ -f $T ] || cp $SCRATCH/tok_bench/bench_server/data/${ds}_10GB.txt $T
    for t in 16 64; do
        c=64-$((63+t))
        /usr/bin/time -f "%e %M" -o $P/tm taskset -c $c ../hfenc-thr/release/hfenc $tok $T bytelevel $P/h.u32 > /dev/null 2>&1
        echo "$ds $t hfmain $(cat $P/tm) ref"
        HB_SHARDS=1 HB_THREADS=$t /usr/bin/time -f "%e %M" -o $P/tm taskset -c $c ../union-enc/target/release/unienc $tok $T bytelevel $P/sh/o > /dev/null 2>&1
        python3 join_shards.py $P/sh/o; cmp -s $P/sh/o.joined $P/h.u32 && r=same || r=DIFFERENT
        echo "$ds $t union $(cat $P/tm) $r"
        rm -f $P/sh/* $P/h.u32
    done
    [ $ds = english ] || rm -f $T
done
echo done
