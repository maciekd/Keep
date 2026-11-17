#!/bin/bash
# Runs the unit suite on a machine made busy: busy loops beside it, LOOPS_PER_CORE for every core,
# so that a test depending on real time (an assertion on elapsed time, two sleeps racing, a thread
# expected somewhere within a timeout) fails here the way it fails by day beside other work and on
# a CI runner — an idle machine passes such a test every time. The nightly test health measures
# flaky tests with it.
#
# Six loops per core is the load under which the suite takes as long here as on GitHub's 3-core
# macOS runner, which runs its 8 workers on those 3 cores (measured 2026-11-17: 128-153 s on the
# development machine, 127-168 s on the runner). At one loop per core a test that failed on the
# runner that day — a loop whose thread lowers its own priority — passed here every time; at
# three per core it failed.
#
#   scripts/test_under_load.sh <runs> <output dir> [pytest arguments…]
#                           <runs> runs one after another, the whole output of run N kept in
#                           <output dir>/run_N.txt; the arguments replace `tests/unit` (the
#                           tests a fix touched, say)
#
# The suite reads the config the caller's FLOXY_CONFIG names (config.yaml without it): the nightly
# run adds one pass with config.example.yaml, the config the CI runners and a fresh clone have.
#
# The loops and the suite both run niced: they contend with each other as equals, and whatever
# else the machine does (the server) keeps its normal priority and gets the CPU when it wants it.
# LOAD_PROCESSES overrides the number of loops. A loop ends by itself once this script is gone,
# however the script went.
cd "$(dirname "$0")/.."

NICENESS=10
LOOPS_PER_CORE=6

if ! [[ "$1" =~ ^[1-9][0-9]*$ ]] || [ -z "$2" ]; then
    echo "usage: scripts/test_under_load.sh <runs> <output dir> [pytest arguments…]" >&2
    exit 2
fi
runs=$1
mkdir -p "$2" || exit 1
out=$(cd "$2" && pwd)
shift 2

loops=${LOAD_PROCESSES:-$(($(getconf _NPROCESSORS_ONLN) * LOOPS_PER_CORE))}
pids=()
for _ in $(seq 1 "$loops"); do
    nice -n $NICENESS bash -c 'while kill -0 "$0" 2>/dev/null; do for ((i = 0; i < 200000; i++)); do :; done; done' $$ &
    pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' EXIT
trap 'exit 130' INT TERM
echo "$loops busy loops and the suite at niceness $NICENESS, $runs runs, output in $out"

source .venv/bin/activate
for run in $(seq 1 "$runs"); do
    nice -n $NICENESS python -m pytest "${@:-tests/unit}" --override-ini="log_cli=false" -q --tb=short \
        > "$out/run_$run.txt" 2>&1
    echo "run $run: $(grep -E '^=+ .*(passed|failed|error)' "$out/run_$run.txt" | tr -d '=')"
done
