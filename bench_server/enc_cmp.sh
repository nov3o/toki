#!/bin/bash
# enc_cmp.sh <results_dir> [pretoks] [threads]: encoding only, HF 0.22 vs HF main vs ours, 1 GB, 4 datasets,
# vocab = the one HF 0.22 trained for that setting (vocab 128k). Order per setting: hf022 hfmain ours ours hfmain
# (main and ours twice, alternating). Ids compared token by token with HF 0.22's (verify.py).
RES=$1; PRETOKS=${2:-"bytelevel llama3 gpt4o whitespace whitespace_split metaspace bert punctuation digits unicode_scripts char_delimiter"}
THREADS=${3:-"8 16"}; OLD=${OLD:-$SCRATCH/tok_bench/bench_server}; SHM=/dev/shm/ec_$USER; mkdir -p $RES $SHM
TSV=$RES/results.tsv; [ -f $TSV ] || echo -e "dataset\tpretok\tthreads\timpl\trun\twall_s\tpeak_rss_MB\ttokens\tids_vs_hf022" > $TSV
one() {  # one <impl> <dataset> <pretok> <threads> <run>
    local tok=$OLD/results_pretok/${2}_1GB/t8_${3}_v131072/train_hf/tokenizer.json d=$RES/$2/$3/t$4/$1_r$5; mkdir -p $d
    [ -f $d/ok ] && return
    case $1 in
        hf022)  cmd="env RAYON_NUM_THREADS=$4 TOKENIZERS_PARALLELISM=true venv-x86_64/bin/python encode_hf.py $tok data_small/${2}_1GB.txt $d $3" ;;
        hfmain) cmd="env RAYON_NUM_THREADS=$4 TOKENIZERS_PARALLELISM=true venv-hfmain/bin/python encode_hf.py $tok data_small/${2}_1GB.txt $d $3" ;;
        ours)   cmd="env HB_THREADS=$4 PYTHONPATH=../rust_bpe_py/build/x86_64/simd venv-x86_64/bin/python encode_ours.py $tok data_small/${2}_1GB.txt $d $3" ;;
    esac
    IDS_PATH=$SHM/$1.u32 timeout 3600 /usr/bin/time -f "%e %M" -o $d/time.txt $cmd > $d/log.txt 2>&1
    read w m < <(tail -1 $d/time.txt)
    local n=$(grep -o "tokens=[0-9]*" $d/log.txt | cut -d= -f2) v=ref
    [ $1 = hf022 ] && mv $SHM/hf022.u32 $SHM/ref.u32 || { v=$(venv-x86_64/bin/python verify.py $SHM/ref.u32 $SHM/$1.u32); rm -f $SHM/$1.u32; }
    echo -e "$2\t$3\t$4\t$1\t$5\t$w\t$((m / 1024))\t$n\t$v" >> $TSV; echo "$(date +%T) $2 $3 t=$4 $1 r$5: ${w}s $v"
    touch $d/ok
}
for p in $PRETOKS; do for t in $THREADS; do for ds in english multilingual code mix; do
    one hf022 $ds $p $t 1; one hfmain $ds $p $t 1; one ours $ds $p $t 1; one ours $ds $p $t 2; one hfmain $ds $p $t 2
    rm -f $SHM/ref.u32
done; done; done
