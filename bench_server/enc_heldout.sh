#!/bin/bash
# enc_heldout.sh <results_dir>: encoding benchmark on held-out text (toki-encoder).
# Each 1 GB sample (documents in random order) is split at a line boundary into train (first 0.5 GB) and test (rest).
# The vocab of every setting is trained on that dataset's train half (our trainer: same merges and vocab as HF);
# HF main, tiktoken (pip; same speed and ids as its main) and ours then encode the test half. Settings: one axis at a
# time around the default (8 threads, GPT-2 byte-level, vocab 128k, no normalizer), 4 datasets.
# Each run is pinned to exactly <threads> cores (taskset). Order per setting: hfmain ours tiktoken ours hfmain (ours and HF twice). Ids compared token by token with HF main's.
RES=$1; DATASETS=${DATASETS:-english multilingual code mix}; BASE=${BASE:-0}   # BASE: first core of this worker
mkdir -p $RES/vocab; SHM=/dev/shm/eh_${USER}_${DATASETS// /_}; mkdir -p $SHM; PY=venv-x86_64/bin/python
OURS=../rust_bpe_py/build/x86_64/nosimd
TSV=$RES/results.tsv; [ -f $TSV ] || echo -e "dataset\taxis\tthreads\tpretok\tvocab\tnorm\timpl\trun\twall_s\tpeak_rss_MB\ttokens\tids_vs_hfmain" > $TSV
for ds in $DATASETS; do
    [ -f data_small/${ds}_test.txt ] || $PY - <<P
import os
b = open("data_small/${ds}_1GB.txt", "rb").read(); c = b.index(b"\n", len(b) // 2) + 1
open("data_small/${ds}_train.txt", "wb").write(b[:c]); open("data_small/${ds}_test.txt", "wb").write(b[c:])
P
done
vocab() {  # vocab <dataset> <pretok> <vocab> <norm> -> path of tokenizer.json trained on the train half
    local d=$RES/vocab/${1}_${2}_v${3}_${4}
    if [ ! -f $d/tokenizer.json ]; then
        mkdir -p $d; BB_THREADS=16 PYTHONPATH=$OURS taskset -c $BASE-$(($BASE + 15)) $PY train_ours.py data_small/${1}_train.txt $2 $3 $d $4 > $d.log 2>&1
        # byte-level: every byte gets a token, as in real byte-level vocabs (test text may have bytes train text lacked);
        # missing ones are appended in byte order, after the trained vocab, so no trained id changes
        case $2 in bytelevel|llama3|gpt4o) $PY - $d/tokenizer.json <<'P'
import json, sys
p = sys.argv[1]; j = json.load(open(p, encoding="utf-8")); v = j["model"]["vocab"]
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); ch = {}; n = 0
for b in range(256):
    ch[b] = chr(b) if b in bs else chr(256 + (n := n + 1) - 1)
nxt = max(v.values()) + 1
for b in range(256):
    if ch[b] not in v: v[ch[b]] = nxt; nxt += 1
json.dump(j, open(p, "w", encoding="utf-8"), ensure_ascii=False)
P
        ;; esac
    fi
    echo $d/tokenizer.json
}
one() {  # one <impl> <dataset> <axis> <threads> <pretok> <vocab> <norm> <run>
    local tok; tok=$(vocab $2 $5 $6 $7); local d=$RES/$2/$3/t$4_$5_v$6_$7/$1_r$8; mkdir -p $d
    [ -f $d/ok ] && return
    case $1 in
        hfmain)   cmd="hfenc-main/target/release/hfenc $tok data_small/${2}_test.txt $5 $SHM/hfmain.u32" ;;   # HF main from Rust (its Python layer was the bottleneck)
        ours)     cmd="env HB_THREADS=$4 PYTHONPATH=$OURS $PY encode_ours.py $tok data_small/${2}_test.txt $d $5" ;;
        tiktoken) cmd="$PY encode_tiktoken.py $tok data_small/${2}_test.txt $5 $4 $d" ;;
    esac
    # every run is pinned to exactly <threads> cores: HF main ignores RAYON_NUM_THREADS and uses all the cores it may run on
    : > $SHM/$1.u32; IDS_PATH=$SHM/$1.u32 timeout 3600 taskset -c $BASE-$(($BASE + $4 - 1)) /usr/bin/time -f "%e %M" -o $d/time.txt $cmd > $d/log.txt 2>&1
    read w m < <(tail -1 $d/time.txt)
    local n=$(grep -o "tokens=[0-9]*" $d/log.txt | cut -d= -f2) v=ref
    if [ $1 = hfmain ] && [ ! -f $SHM/ref.u32 ]; then mv $SHM/hfmain.u32 $SHM/ref.u32
    else v=$($PY verify.py $SHM/ref.u32 $SHM/$1.u32); rm -f $SHM/$1.u32; fi
    echo -e "$2\t$3\t$4\t$5\t$6\t$7\t$1\t$8\t$w\t$((m / 1024))\t$n\t$v" >> $TSV; echo "$(date +%T) $2 $3 t=$4 $5 v$6 $7 $1 r$8: ${w}s $v"
    [ -n "$n" ] && [ "$n" != 0 ] && touch $d/ok      # only a run that produced ids counts as done
}
setting() {  # setting <axis> <threads> <pretok> <vocab> <norm>
    for ds in $DATASETS; do
        rm -f $SHM/ref.u32
        one hfmain $ds "$@" 1; one ours $ds "$@" 1
        case $3 in bytelevel|llama3|gpt4o) [ $5 = none ] && one tiktoken $ds "$@" 1 ;; esac
        one ours $ds "$@" 2; one hfmain $ds "$@" 2
    done
}
setting default 8 bytelevel 131072 none
for t in 2 4 16; do setting threads $t bytelevel 131072 none; done
for v in 32768 65536 262144; do setting vocab 8 bytelevel $v none; done
for p in llama3 gpt4o whitespace whitespace_split metaspace bert punctuation digits unicode_scripts char_delimiter; do setting pretok 8 $p 131072 none; done
for n in nfc nfd nfkc nfkd lowercase strip_accents nfkc_lowercase replace bert; do setting norm 8 bytelevel 131072 $n; done
for t in 2 16; do setting headline $t llama3 32768 none; done
echo "done"
