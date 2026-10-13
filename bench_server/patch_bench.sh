#!/bin/bash
# patch_bench.sh <impls> <results_dir>: before vs after a patch vs ours, 1 GB english, encode only, one step at a time.
# impls: space-separated name=venv pairs for tiktoken builds, e.g. "tik_before=venv-tikbase tik_after=venv-tikpatch".
# Order per setting: A B ours ours B A (two runs each, drift cancels). Ids are checked against the first impl.
IMPLS=$1; RES=$2; mkdir -p $RES; TSV=$RES/results.tsv
OLD=${OLD:-$SCRATCH/tok_bench/bench_server}; C=data_small/english_1GB.txt; SHM=/dev/shm/pb_$USER; mkdir -p $SHM
[ -f $TSV ] || echo -e "impl\tpretok\tthreads\trun\twall_s\tpeak_rss_MB\tids_sha256" > $TSV
one() {  # one <impl> <venv> <pretok> <threads> <run>
    local tok=$OLD/results_pretok/english_1GB/t8_${3}_v131072/train_hf/tokenizer.json d=$RES/$1_$3_t$4_r$5; mkdir -p $d
    if [ $1 = ours ]; then
        HB_THREADS=$4 PYTHONPATH=../rust_bpe_py/build/x86_64/simd IDS_PATH=$SHM/ids.u32 /usr/bin/time -f "%e %M" -o $d/time.txt \
            venv-x86_64/bin/python encode_ours.py $tok $C $d $3 > $d/log.txt 2>&1
    else
        IDS_PATH=$SHM/ids.u32 /usr/bin/time -f "%e %M" -o $d/time.txt $2/bin/python encode_tiktoken.py $tok $C $3 $4 $d > $d/log.txt 2>&1
    fi
    read w m < <(tail -1 $d/time.txt); echo -e "$1\t$3\t$4\t$5\t$w\t$((m / 1024))\t$(sha256sum < $SHM/ids.u32 | cut -c1-16)" >> $TSV; rm -f $SHM/ids.u32
    echo "$(date +%T) $1 $3 t=$4 run=$5: ${w}s"
}
for p in bytelevel llama3 gpt4o; do for t in 8 16; do
    set -- $IMPLS; a=${1%%=*}; av=${1#*=}; b=${2%%=*}; bv=${2#*=}
    one $a $av $p $t 1; one $b $bv $p $t 1; one ours - $p $t 1; one ours - $p $t 2; one $b $bv $p $t 2; one $a $av $p $t 2
done; done
