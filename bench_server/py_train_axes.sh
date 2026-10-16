#!/bin/bash
# py_train_axes.sh <out.tsv>: Python-binding overhead of training, PR #2501 (hfbench / train_bpe_pr2501) and ours
# (trunion / train_bpe), Rust vs Python, on the default setting and a few axes. Merges written and compared.
cd $SCRATCH/tok_bench_v3/bench_server
R=$1; P=/dev/shm/ibadanin_pyt; mkdir -p $P; PYF=$SCRATCH/fork-venv/bin/python
[ -f $R ] || echo -e "dataset\tsetting\timpl\tapi\trun\twall_s\tpeak_rss_KB\tout" > $R
run() { # ds setting impl api run merges_file cmd...
  grep -q "^$1	$2	$3	$4	$5	" $R && return
  local d=$1 s=$2 i=$3 a=$4 r=$5 m=$6; shift 6; rm -rf $P/o; mkdir -p $P/o; rm -f $m
  HB_NORMALIZER="$SPEC" RAYON_NUM_THREADS=$T /usr/bin/time -f "%e %M" -o $P/tm taskset -c $CL "$@" > $P/log 2>&1
  [ -f $P/o/merges.txt ] && cp $P/o/merges.txt $m
  local out=$(grep -oE "merges=[0-9]+" $P/log | head -1); [ -f $m ] && out="merges=$(wc -l < $m)"
  echo -e "$d\t$s\t$i\t$a\t$r\t$(tr ' ' '\t' < $P/tm)\t$out" | tee -a $R
}
# setting: threads pretok normalizer datasets
for st in "8 bytelevel none english,multilingual,code,mix" "2 bytelevel none english,multilingual" \
          "16 bytelevel none english,multilingual" "8 llama3 none english,multilingual" \
          "8 bert none english,multilingual" "8 bytelevel nfkc_lowercase english,multilingual"; do
  set -- $st; T=$1 pt=$2 n=$3; name=t${T}_${pt}_${n}; CL=96-$((95 + T))
  SPEC=$(venv-x86_64/bin/python -c "import normalizers; print(normalizers.spec('$n'))")
  norms=$( [ $n = none ] && echo "" || echo ${n/_/,} ); bl=0; case $pt in bytelevel|llama3|gpt4o) bl=1;; esac
  for ds in ${4//,/ }; do
    tok=results_enc_ho3/vocab/${ds}_${pt}_v131072_${n}/tokenizer.json; C=data_small/${ds}_1GB.txt
    for r in 1 2; do
      run $ds $name pr2501 rust $r $P/a.m hfbench2-pr2501/target/release/hfbench $tok $C 131072 $P/o $bl
      run $ds $name pr2501 python $r $P/b.m $PYF py_bench.py train_pr2501 131072 $C $tok $bl $P/b.m
      run $ds $name ours rust $r $P/c.m env HB_THREADS=$T HB_PRETOK=$pt MIMALLOC_PURGE_DELAY=0 ../union-pr2501/target/release/trunion $C 131072 $P/o
      run $ds $name ours python $r $P/d.m $PYF py_bench.py train 131072 $C $T $pt "$norms" $P/d.m
      [ $r = 1 ] && for f in b c d; do cmp -s <(cut -d' ' -f1,2 $P/a.m) <(cut -d' ' -f1,2 $P/$f.m) && echo "$ds $name $f same" || echo "$ds $name $f DIFF"; done >> $P/check
    done
  done
done
echo done
