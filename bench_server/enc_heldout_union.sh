#!/bin/bash
# enc_heldout_union.sh: the union encoder on the held-out settings the sweep did not cover (threads 2/4/8/16 default,
# vocab 32k/64k/256k, headline llama3 32k at 2 and 16 threads), 4 datasets, ids vs HF main, cores 64+.
# Appends rows in results_enc_ho3/results.tsv format to results_enc_ho3/union.tsv (impl = union).
cd $SCRATCH/tok_bench_v3/bench_server; P=/dev/shm/prof_$USER/eu; mkdir -p $P; O=results_enc_ho3/union.tsv
head -1 results_enc_ho3/results.tsv > $O
one() {  # one <ds> <axis> <threads> <pretok> <vocab> <norm>
    local tok=results_enc_ho3/vocab/${1}_${4}_v${5}_${6}/tokenizer.json T=data_small/${1}_test.txt c=64-$((63 + $3))
    taskset -c $c ../hfenc-thr/release/hfenc $tok $T $4 $P/h.u32 > /dev/null 2>&1
    for r in 1 2; do
        HB_THREADS=$3 taskset -c $c /usr/bin/time -f "%e %M" -o $P/tm ../union-enc2/target/release/unienc $tok $T $4 $P/u.u32 > /dev/null 2>&1
        read w m < <(tail -1 $P/tm); n=$(( $(stat -c%s $P/u.u32) / 4 ))
        cmp -s $P/u.u32 $P/h.u32 && [ $n -gt 0 ] && v="same ($n ids)" || v="DIFFERENT"
        echo -e "$1\t$2\t$3\t$4\t$5\t$6\tunion\t$r\t$w\t$((m / 1024))\t$n\t$v" >> $O
    done
}
for ds in english multilingual code mix; do
    one $ds default 8 bytelevel 131072 none
    for t in 2 4 16; do one $ds threads $t bytelevel 131072 none; done
    for v in 32768 65536 262144; do one $ds vocab 8 bytelevel $v none; done
    for pt in llama3 gpt4o whitespace whitespace_split metaspace bert punctuation digits unicode_scripts char_delimiter; do one $ds pretok 8 $pt 131072 none; done
    for nm in nfc nfd nfkc nfkd lowercase strip_accents nfkc_lowercase replace bert; do one $ds norm 8 bytelevel 131072 $nm; done
    for t in 2 16; do one $ds headline $t llama3 32768 none; done
done
echo done
