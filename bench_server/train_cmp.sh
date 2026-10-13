#!/bin/bash
# train_cmp.sh <results_dir>: BPE training, HF 0.22 (Python) vs HF main vs HF PR #2501 (Rust, hfbench) vs ours,
# GPT-2 byte-level, 8 threads, 4 datasets; PR #2501's setting (256 MiB, vocab 100k) and ours (1 GB, vocab 128k).
# One step at a time; merges and vocab of every run compared with HF 0.22's.
RES=$1; mkdir -p $RES; TSV=$RES/results.tsv; T=8; PY=venv-x86_64/bin/python
[ -f $TSV ] || echo -e "setting\tdataset\timpl\twall_s\tpeak_rss_MB\tfeed_s\ttrain_s\tmerges_vs_hf022\tvocab_vs_hf022" > $TSV
for setting in "256MiB 100000" "1GB 131072"; do
    set -- $setting; sz=$1; V=$2
    for ds in english code multilingual mix; do
        c=data_small/${ds}_1GB.txt
        if [ $sz = 256MiB ]; then c=/dev/shm/${ds}_256MiB.txt; head -c 268435456 data_small/${ds}_1GB.txt > $c; fi
        for impl in hf022 hfmain hfpr2501 ours; do
            d=$RES/${sz}_$ds/$impl; mkdir -p $d
            case $impl in
                hf022) cmd="env RAYON_NUM_THREADS=$T $PY train_hf.py $c bytelevel $V $d none" ;;
                ours)  cmd="env BB_THREADS=$T PYTHONPATH=../rust_bpe_py/build/x86_64/simd $PY train_ours.py $c bytelevel $V $d none" ;;
                *)     cmd="env RAYON_NUM_THREADS=$T hfbench-${impl#hf}/target/release/hfbench $c $V $d" ;;
            esac
            echo "$(date +%T) $impl $sz $ds"
            timeout 7200 /usr/bin/time -f "%e %M" -o $d/time.txt $cmd > $d/log.txt 2>&1
            read w m < <(tail -1 $d/time.txt)
            f=$(grep -o "feed_s=[0-9.]*" $d/log.txt | cut -d= -f2); t=$(grep -o "train_s=[0-9.]*" $d/log.txt | cut -d= -f2)
            vm=$($PY verify.py $RES/${sz}_$ds/hf022/merges.txt $d/merges.txt 2>&1 | head -c 60); vv=$($PY verify.py $RES/${sz}_$ds/hf022/vocab.txt $d/vocab.txt 2>&1 | head -c 60)
            echo -e "$sz\t$ds\t$impl\t$w\t$((m / 1024))\t$f\t$t\t$vm\t$vv" >> $TSV
        done
        rm -f /dev/shm/${ds}_256MiB.txt
    done
done
