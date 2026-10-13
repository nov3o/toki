#!/bin/bash
# train_sweep_union.sh: training, HF main vs PR #2501 vs union, 1 GB x 4 datasets, one axis at a time (pre-tokenizer,
# normalizer), vocab 131072, pinned to cores 0-15. HF runs use HF's own normalizers / pre-tokenizer (hfbench2, config
# from the held-out tokenizer.json of that setting). Merges compared byte by byte with PR #2501's.
cd $SCRATCH/tok_bench_v3/bench_server; P=/dev/shm/prof_$USER/trs2; mkdir -p $P; PY=venv-x86_64/bin/python
echo "dataset axis pretok norm impl wall_s rss_KB feed_s train_s merges_vs_pr2501"
run() {  # run <ds> <axis> <pretok> <norm>
    local ds=$1 pt=$3 nm=$4; local C=data_small/${ds}_1GB.txt tok=results_enc_ho3/vocab/${ds}_${3}_v131072_${4}/tokenizer.json
    local bl=0 sp="[UNK]"; case $pt in bytelevel|llama3|gpt4o) bl=1; sp="";; esac
    local spec; spec=$($PY -c "import normalizers; print(normalizers.spec('$nm'))")
    for impl in pr2501 hfmain union; do
        mkdir -p $P/$impl
        case $impl in
            pr2501) cmd="hfbench2-pr2501/target/release/hfbench $tok $C 131072 $P/$impl $bl $sp" ;;
            hfmain) cmd="hfbench2-main/target/release/hfbench $tok $C 131072 $P/$impl $bl $sp" ;;
            union)  cmd="env HB_PRETOK=$pt HB_THREADS=16 ../union-pr2501/target/release/trunion $C 131072 $P/$impl" ;;
        esac
        HB_NORMALIZER="$spec" timeout 3600 taskset -c 0-15 /usr/bin/time -f "%e %M" -o $P/tm $cmd > $P/log 2>&1
        f=$(grep -o "feed_s=[0-9.]*" $P/log | cut -d= -f2); t=$(grep -o "train_s=[0-9.]*" $P/log | cut -d= -f2)
        if [ $impl = pr2501 ]; then r=ref; elif cmp -s $P/$impl/merges.txt $P/pr2501/merges.txt; then r=same; else r=DIFFERENT; fi
        echo "$ds $2 $pt $nm $impl $(tail -1 $P/tm) ${f:-NA} ${t:-NA} $r"
    done
}
for ds in english multilingual code mix; do
    for pt in bytelevel llama3 gpt4o whitespace whitespace_split metaspace bert punctuation digits unicode_scripts char_delimiter; do run $ds pretok $pt none; done
    for nm in nfc nfd nfkc nfkd lowercase strip_accents nfkc_lowercase replace bert; do run $ds norm bytelevel $nm; done
done
echo done
