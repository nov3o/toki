#!/bin/bash
# other.sh <data_dir> <results_dir>: SentencePiece (BPE, Unigram) and HF WordPiece / Unigram at the default setting
# (1 GB, 8 threads, vocab 131072), one step at a time, for speed/memory context next to run.sh's BPE numbers.
DATA=$1; RES=$2; mkdir -p "$RES"; PY=venv-$(uname -m)/bin/python; T=8; V=${V:-131072}
TSV=$RES/results.tsv; [ -f "$TSV" ] || echo -e "dataset\tmethod\tstep\tstatus\twall_s\tinner_s\tpeak_rss_MB\ttokens" > "$TSV"
for ds in ${DATASETS:-english multilingual code mix}; do
  for m in ${METHODS:-sp_bpe sp_unigram hf_wordpiece hf_unigram}; do
    for st in train encode; do
      dir=$RES/${ds}/${st}_$m; mkdir -p "$dir"; [ -f "$dir/ok" ] && continue
      [ $st = encode ] && [ ! -f "$RES/${ds}/train_$m/ok" ] && continue
      echo "$(date +%T) $m $st $ds"
      timeout ${STEP_TIMEOUT:-3600} /usr/bin/time -f "%e %M" -o "$dir/time.txt" $PY bench_other.py $m $st "$DATA/${ds}_1GB.txt" $V $T "$dir" > "$dir/log.txt" 2>&1
      rc=$?; status=ok; [ $rc -ne 0 ] && status=failed_rc$rc; [ $rc -eq 124 ] && status=timeout && echo "${STEP_TIMEOUT:-3600} 0" > "$dir/time.txt"
      read wall rss < <(tail -1 "$dir/time.txt")
      inner=$(grep -o "${st}_s=[0-9.]*" "$dir/log.txt" | cut -d= -f2); tokens=$(grep -o "tokens=[0-9]*" "$dir/log.txt" | cut -d= -f2)
      echo -e "$ds\t$m\t$st\t$status\t$wall\t$inner\t$((rss / 1024))\t$tokens" >> "$TSV"; echo "    -> $status wall=${wall}s peak=$((rss / 1024))MB"
      [ $rc -eq 0 ] && touch "$dir/ok"
    done
  done
done
