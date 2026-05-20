#!/usr/bin/env bash
set -euo pipefail

# Benchmark runner for the MPI heat solver.
# Requirement: run p=1,2,4,8,... and take average of 3 consecutive runs.
# This script writes the *raw* run data into a CSV; use analyze.py for averages + Karp–Flatt.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROGRAM="${ROOT_DIR}/scripts/heat2d_mpi.py"

N=512
ITERS=100000
BC_TOP=100
BC_LEFT=0
BC_BOTTOM=0
BC_RIGHT=0
CSV_OUT="${ROOT_DIR}/results/raw_times.csv"
MAXP="$(nproc)"
MPIRUN="mpirun"

usage() {
  cat <<EOF
Usage: $0 [--N N] [--iters I] [--bc TOP LEFT BOTTOM RIGHT] [--maxp MAXP] [--mpirun MPIRUN] [--csv PATH]

Example:
  bash scripts/bench.sh --N 512 --iters 100000 --maxp 16 --csv results/raw_times.csv
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --N) N="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --bc) BC_TOP="$2"; BC_LEFT="$3"; BC_BOTTOM="$4"; BC_RIGHT="$5"; shift 5;;
    --maxp) MAXP="$2"; shift 2;;
    --mpirun) MPIRUN="$2"; shift 2;;
    --csv) CSV_OUT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

mkdir -p "$(dirname "$CSV_OUT")"

# Build p list: powers of two up to MAXP.
P_LIST=(1)
p=2
while [[ $p -le $MAXP ]]; do
  P_LIST+=("$p")
  p=$((p*2))
done

# CSV header (always start fresh to avoid mixing configurations)
echo "p,run,T_total,iters,N,dims" > "$CSV_OUT"

  echo "Benchmarking: N=$N iters=$ITERS maxp=$MAXP bc=[$BC_TOP,$BC_LEFT,$BC_BOTTOM,$BC_RIGHT]"

  if [[ "$(id -u)" -eq 0 ]]; then
    default_cmd=($MPIRUN --allow-run-as-root)
  else
    default_cmd=($MPIRUN)
  fi

for p in "${P_LIST[@]}"; do
  if [[ $p -lt 1 ]]; then continue; fi
  if [[ $p -gt $MAXP ]]; then continue; fi

  for run in 1 2 3; do
    # Solver prints a single summary line; no file I/O is performed.
    out_line=$("${default_cmd[@]}" -np "$p" python3 "$PROGRAM" \
      --N "$N" --iters "$ITERS" --bc "$BC_TOP" "$BC_LEFT" "$BC_BOTTOM" "$BC_RIGHT" \
      | tail -n 1)

    # Expected line format from rank 0:
    #   T_total=0.001234 iters=50 N=32 p=2 dims=[2, 1]
    T_total=$(echo "$out_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i ~ /^T_total$/) print $(i+1)}')
    iters_done=$(echo "$out_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i ~ /^iters$/) print $(i+1)}')
    N_seen=$(echo "$out_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i ~ /^N$/) print $(i+1)}')
    dims=$(echo "$out_line" | awk -F'dims=' '{print $2}')

    echo "$p,$run,$T_total,$iters_done,$N_seen,\"$dims\"" >> "$CSV_OUT"
    echo "p=$p run=$run T_total=$T_total"
  done
done

echo "Raw results written to: $CSV_OUT"
