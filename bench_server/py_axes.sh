#!/bin/bash
# py_axes.sh <out.tsv>: default setting (1 GB, 8 threads, GPT-2 byte-level, vocab 131072), Rust-only vs Python binding
cd $SCRATCH/tok_bench_v3/bench_server
R=$1; CL=${CL:-56-63}; P=/dev/shm/ibadanin_py; mkdir -p $P
[ -f $R ] || echo -e "task\tdataset\timpl\tapi\trun\twall_s\tpeak_rss_KB\tout" > $R
run() { # task ds impl api run cmd...
  grep -q "^$1	$2	$3	$4	$5	" $R && return
  local t=$1 d=$2 i=$3 a=$4 r=$5; shift 5
  /usr/bin/time -f "%e %M" -o $P/tm taskset -c $CL "$@" > $P/log 2>&1
  local out=$(grep -oE "(ids|tokens|merges)=[0-9]+" $P/log | head -1)
  [ $t = train ] && [ -f $P/o/merges.txt ] && out="merges=$(wc -l < $P/o/merges.txt)"
  echo -e "$t\t$d\t$i\t$a\t$r\t$(tr ' ' '\t' < $P/tm)\t$out" | tee -a $R
}
for ds in english multilingual code mix; do
  tok=results_enc_ho3/vocab/${ds}_bytelevel_v131072_none/tokenizer.json; T=data_small/${ds}_test.txt; C=data_small/${ds}_1GB.txt
  for r in 1 2; do
    run encode $ds hfmain rust $r ../hfenc-thr/release/hfenc $tok $T bytelevel $P/a.u32
    run encode $ds hfmain python $r $SCRATCH/hfmain-venv/bin/python py_bench.py encode_batch $tok $T
    run encode $ds ours rust $r env HB_THREADS=8 ../union-enc2/target/release/unienc $tok $T bytelevel $P/b.u32
    run encode $ds ours python $r $SCRATCH/fork-venv/bin/python py_bench.py encode_whole $tok $T
    rm -rf $P/o; run train $ds ours rust $r env HB_THREADS=8 MIMALLOC_PURGE_DELAY=0 ../union-pr2501/target/release/trunion $C 131072 $P/o
    run train $ds ours python $r $SCRATCH/fork-venv/bin/python py_bench.py train 131072 $C 8
  done
done
echo done
