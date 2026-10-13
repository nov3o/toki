#!/bin/bash
# Overnight queue on bristen: everything one step at a time. Code: $SCRATCH/tok_bench_v3. Log: logs/queue.log
set -u
V3=$SCRATCH/tok_bench_v3
OLD=$SCRATCH/tok_bench/bench_server
cd $V3/bench_server
export PATH=$SCRATCH/rust-x86_64/cargo/bin:$PATH CARGO_HOME=$SCRATCH/rust-x86_64/cargo RUSTUP_HOME=$SCRATCH/rust-x86_64/rustup
PY=venv-x86_64/bin/python
log() { echo "$(date '+%F %T') $*"; }
log "start on $(hostname); $(uptime)"

# 0. build: binding (simd + nosimd) and two trainer CLIs for the A/B
PYO3_PYTHON=$PWD/venv-x86_64/bin/python ../rust_bpe_py/build.sh 2>&1 | grep -E "error|Finished"
(cd ../rust_bpe && RUSTFLAGS="-C target-cpu=native" cargo build --release --features simd --target-dir target/x86_64-new 2>&1 | grep -E "^error|Finished")
(cd ../ab_old/rust_bpe && RUSTFLAGS="-C target-cpu=native" cargo build --release --features simd --target-dir target/x86_64-old 2>&1 | grep -E "^error|Finished")

# 1. pre-tokenizer check at 1 GB (resumes $OLD/results_pretok); re-run ours wherever ours failed; HF capped at 60 min/step
for d in $(awk -F'\t' '$3 ~ /^ours/ && $8 ~ /failed/ {print $13}' $OLD/results_pretok/results.tsv | xargs -r -n1 dirname | sort -u); do
  rm -f $d/done; log "repair: $d"
done
log "pre-tokenizer check"
STEP_TIMEOUT=3600 ALL_THREADS= ALL_VOCABS= ALL_NORMS= SIZES=1 ./run.sh data_small $OLD/results_pretok > logs/pretok2.log 2>&1
# 1c. second repair pass (paths in the old results.tsv are relative to $OLD)
for d in $(awk -F'	' '$3 ~ /^ours/ && $8 ~ /failed/ {print $13}' $OLD/results_pretok/results.tsv | xargs -r -n1 dirname | sort -u); do
  case $d in /*) ;; *) d=$OLD/$d;; esac
  if [ -f $d/done ] && [ ! -f $d/train_ours_simd/ok ]; then rm -f $d/done; log "repair: $d"; fi
done
STEP_TIMEOUT=3600 ALL_THREADS= ALL_VOCABS= ALL_NORMS= SIZES=1 ./run.sh data_small $OLD/results_pretok > logs/pretok3.log 2>&1
log "pretok verify: $($PY verify.py $OLD/results_pretok | tail -1)"

# 2. A/B trainer (previous commit vs now), multilingual 1 GB, 8 threads, alternating
D=data_small/multilingual_1GB.txt
for r in 1 2 3; do
  for v in old new; do
    if [ $v = old ]; then B=../ab_old/rust_bpe/target/x86_64-old/release/bucket_bpe; else B=../rust_bpe/target/x86_64-new/release/bucket_bpe; fi
    o=$(BB_THREADS=8 BB_PRETOK=bytelevel $B $D 0 131072 /dev/shm/ab_${USER}_$v.merges | grep -o "count_s=[0-9.]*\|init_s=[0-9.]*\|merge_s=[0-9.]*\|total_s=[0-9.]*\|peak_rss_mb=[0-9.]*" | tr "\n" " ")
    log "AB run=$r $v $o"
  done
done
cmp /dev/shm/ab_${USER}_old.merges /dev/shm/ab_${USER}_new.merges && log "AB merges identical"; rm -f /dev/shm/ab_${USER}_*.merges

# 3. 1 GB: threads + vocab axes
log "1GB threads+vocab"; STEP_TIMEOUT=3600 ALL_PRETOKS= ALL_NORMS= SIZES=1 ./run.sh data_small results_axes > logs/axes.log 2>&1
log "axes verify: $($PY verify.py results_axes | tail -1)"
# 4. 1 GB: normalizer axis
log "1GB normalizers"; STEP_TIMEOUT=3600 ALL_THREADS= ALL_PRETOKS= ALL_VOCABS= SIZES=1 ./run.sh data_small results_norm > logs/norm.log 2>&1
log "norm verify: $($PY verify.py results_norm | tail -1)"

# 5. where HF timed out at 1 GB: check exactness at 100 MB (first 100 MB of the same sample, cut at a line end)
TP=$(awk -F'\t' '$3 == "hf" && $8 ~ /timeout|failed/ {print $5}' $OLD/results_pretok/results.tsv | sort -u | tr "\n" " ")
if [ -n "$TP" ]; then
  log "HF timed out at 1 GB for: $TP -> 100 MB check"
  for ds in english multilingual code mix; do
    f=data_small/${ds}_0.1GB.txt
    [ -f $f ] || $PY -c "
b=open('data_small/${ds}_1GB.txt','rb').read(100_000_000); open('$f','wb').write(b[:b.rfind(b'\n')+1])"
  done
  STEP_TIMEOUT=3600 ALL_THREADS= ALL_VOCABS= ALL_NORMS= ALL_PRETOKS="$TP" SIZES=0.1 ./run.sh data_small results_pretok_small > logs/pretok_small.log 2>&1
  log "pretok 100MB verify: $($PY verify.py results_pretok_small | tail -1)"
fi

# 6. draw the 1 GB / 10 GB samples from the 100 GB pools (pools already built; existing samples are kept)
log "sampling 1/10 GB from the 100 GB pools"; $PY prepare_data.py data 100 1 10 2>&1 | grep -v Warning | tail -12
# 7. 10 GB baseline
log "10GB baseline"; STEP_TIMEOUT=10800 ALL_THREADS= ALL_PRETOKS= ALL_VOCABS= ALL_NORMS= SIZES=10 ./run.sh data results_10g > logs/10g.log 2>&1
log "10g verify: $($PY verify.py results_10g | tail -1)"
log "queue done"
