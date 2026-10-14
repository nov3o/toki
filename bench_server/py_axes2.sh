#!/bin/bash
# py_axes2.sh <out.tsv>: Python-binding overhead on a subset of axes (threads, pre-tokenizer, normalizer, vocab)
cd $SCRATCH/tok_bench_v3/bench_server
R=$1; P=/dev/shm/ibadanin_py2; mkdir -p $P
[ -f $R ] || echo -e "dataset\tsetting\timpl\tapi\trun\twall_s\tpeak_rss_KB\tout" > $R
run() { # ds setting impl api run cmd...
  grep -q "^$1	$2	$3	$4	$5	" $R && return
  local d=$1 s=$2 i=$3 a=$4 r=$5; shift 5
  /usr/bin/time -f "%e %M" -o $P/tm taskset -c $CL "$@" > $P/log 2>&1
  local out=$(grep -oE "(ids|tokens)=[0-9]+" $P/log | head -1)
  echo -e "$d\t$s\t$i\t$a\t$r\t$(tr ' ' '\t' < $P/tm)\t$out" | tee -a $R
}
# setting: threads pretok vocab normalizer
for st in "2 bytelevel 131072 none" "16 bytelevel 131072 none" "8 llama3 131072 none" "8 bert 131072 none" \
          "8 metaspace 131072 none" "8 bytelevel 131072 nfkc_lowercase" "8 bytelevel 32768 none"; do
  set -- $st; t=$1 pt=$2 v=$3 n=$4; name=t${t}_${pt}_v${v}_${n}
  CL=80-$((79 + t))
  for ds in english multilingual; do
    tok=results_enc_ho3/vocab/${ds}_${pt}_v${v}_${n}/tokenizer.json; T=data_small/${ds}_test.txt
    for r in 1 2; do
      run $ds $name hfmain rust $r ../hfenc-thr/release/hfenc $tok $T $pt $P/a.u32
      run $ds $name hfmain python $r $SCRATCH/hfmain-venv/bin/python py_bench.py encode_batch $tok $T
      run $ds $name ours rust $r env HB_THREADS=$t ../union-enc2/target/release/unienc $tok $T $pt $P/b.u32
      run $ds $name ours rust_purge $r env HB_THREADS=$t MIMALLOC_PURGE_DELAY=0 ../union-enc2/target/release/unienc $tok $T $pt $P/b.u32
      run $ds $name ours python_stream $r $SCRATCH/fork-venv/bin/python py_bench.py encode_file $tok $T $P/c.u32 $t
      [ $r = 1 ] && { cmp -s $P/b.u32 $P/c.u32 && echo "$ds $name IDENTICAL" || echo "$ds $name DIFF"; } >> $P/check
    done
  done
done
echo done
