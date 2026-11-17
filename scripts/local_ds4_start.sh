#!/bin/bash
# Starts the local ds4-server for Keep: DeepSeek V4.1 Flash (the IQ2_XXS/Q2_K imatrix GGUF) on
# this machine's GPU, served OpenAI-style on port 8000 on every interface, so a Keep on another
# machine of the local network reaches it as http://<this host>:8000/v1 (`ds4.base_url`).
# The ds4 checkout is expected beside this repository as ../ds4 (DS4_DIR points elsewhere), cloned from
# the keep branch of our fork — `git clone -b keep https://github.com/maciekd/ds4`, then make: upstream
# github.com/antirez/ds4 plus the disk KV cache changes below (antirez/ds4#1322, #1323, #1347).
#
# The disk KV cache flags matter: ds4-server keeps one live KV context and reuses it only
# for a prompt that extends it, so without them every switch between two prompts (another
# tier, the autonomous check, a Keep restart) prefills the whole prompt again — ~12 s for
# 16k tokens instead of 0.2 s. With them the server saves prompt prefixes to disk and loads
# the longest one that fits a request. (Keep's src/assistant/llm/Ds4LLM.py docstring has the
# measurements.) The budget is 64 GB because the server saves the evicted live state on every
# switch between prompts — ~253 MiB per entry at Keep's 40k-token prompt — and, once the budget
# is full, evicts the entry with no hits yet, which is the cold checkpoint a new prompt just
# stored: from then on every request of that prompt prefills the whole 40k from zero (31 s).
# 16 GB held ~60 switches; 64 GB holds ~250. Measured 2026-10-21 on the prompt test suite,
# where a reasoning effort other than none renders the history differently and is a new prompt.
#
# The server runs detached and appends to ds4-server.log in the ds4 folder (Keep's prompt tests
# read it for the real prompt and the cache decisions; a restart keeps the earlier runs' lines,
# so a problem seen before the restart can still be read). The script waits until the API answers,
# which takes a minute or two while the model is mapped. Stop it with `pkill ds4-server`.
#
# Images: the server refuses a request with an image unless it was started with the matching
# vision encoder (`./download_model.sh ds41f-vision`, 0.9 GiB, ~1 GiB resident; an image costs
# up to 1024 context tokens). Keep's Photos indexer sends images, so with `local_backend: ds4`
# routing its tiers here the encoder must be on. The script passes it when the file is present;
# VISION= drops it, VISION=path names another one.
#
# Usage: scripts/local_ds4_start.sh [model.gguf]
# Overrides: DS4_DIR, MODEL (GGUF path, same as the argument), VISION (encoder GGUF path), CTX (context tokens), KV_DIR, KV_MB,
# COLD_MAX, COLD_ANCHOR, HOST, PORT.
set -euo pipefail
DS4_DIR="${DS4_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/ds4}"
cd "$DS4_DIR" 2>/dev/null || { echo "no ds4 checkout at $DS4_DIR — git clone -b keep https://github.com/maciekd/ds4 beside this repository (then make) or set DS4_DIR"; exit 1; }

# The default is what ds4's ./download_model.sh ds41f-q2 fetches; another model is the first argument
# (or MODEL=), relative to the ds4 checkout or absolute.
MODEL="${1:-${MODEL:-gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf}}"
VISION="${VISION-gguf/DeepSeek-V4.1-Flash-Vision.gguf}"
# 196k: the memory curator's nightly prompt reached 98k tokens on 2026-10-22 and failed at the
# 64k window with a 400 (the first-run bootstrap chunks hit the same wall), and the test-suite
# prompt is ~16.5k plus one round's command results (the server log tail alone is ~8.5k). The
# model is trained to 1M positions. The window is cheap: the context buffers are ~8 GiB of
# prefill scratch that does not grow with it plus ~6.4 KB a token of compressed KV, so 196k
# costs ~2.5 GiB more than 64k (the server's startup "memory:" line) next to the 152 GiB model.
CTX="${CTX:-196608}"
KV_DIR="${KV_DIR:-$HOME/.ds4/server-kv}"
KV_MB="${KV_MB:-65536}"
# The disk cache's "cold" checkpoint (the prompt cut at its first user turn, what a sibling
# request loads) is only written for a prompt up to this many tokens; the server's default
# of 30000 left the 40k prompt-test prompt with no reusable entry at all — every test
# prefilled from zero (32 s). With the ceiling at the context size a test loads the system
# prompt (11k) and prefills the rest (19 s). Measured 2026-10-21.
COLD_MAX="${COLD_MAX:-$CTX}"
# Where that checkpoint is cut: ds4's default is before the first task (the last user turn
# before the first assistant turn), which for Keep's prompt tests is the system prompt alone
# (11k of 40k). `last-user` cuts before the prompt's last user turn — the test's query, which
# the test server sends as a turn of its own on ds4 (Ds4LLM.prefers_query_as_own_turn) — so a
# test loads everything but its query from disk. The fork's keep branch has
# the flag; an upstream ds4 without it refuses to start with it, so set COLD_ANCHOR= to drop it.
COLD_ANCHOR="${COLD_ANCHOR-last-user}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG=ds4-server.log

[ -x ./ds4-server ] || { echo "no ds4-server binary in $DS4_DIR — run make there first"; exit 1; }
if [ ! -f "$MODEL" ]; then
    echo "You do not have the model $MODEL in $DS4_DIR."
    echo "Download the default one with ./download_model.sh ds41f-q2 there, or pass the model you want to run:"
    echo "  $0 gguf/<model>.gguf"
    others=$(ls gguf/*.gguf 2>/dev/null | grep -v -- '-Vision\.gguf$' || true)
    [ -n "$others" ] && echo "Models in $DS4_DIR/gguf:" && echo "$others" | sed 's/^/  /'
    exit 1
fi
if [ -n "$VISION" ] && [ ! -f "$VISION" ]; then
    echo "no vision encoder at $VISION — starting without images (./download_model.sh ds41f-vision adds it)"
    VISION=
fi

busy=$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$busy" ]; then
    echo "port $PORT is already taken by pid(s) $busy — stop that server first (pkill ds4-server)"
    exit 1
fi

mkdir -p "$KV_DIR"

# macOS unwires an idle process's GPU memory after about a second, and ds4's keepalive kernel
# runs once a second by default — it loses that race every ~6 s, and a request arriving then
# waits 2.5–11 s while 209 GB is wired again (measured 2026-11-11 on the M5 Ultra, macOS 27;
# at 500 ms and at 250 ms: no unwiring in 60 s idle, every request at its normal latency).
# The fork's keep branch has 250 ms as its default (antirez/ds4#1347); this covers an upstream build.
export DS4_METAL_QUEUE_KEEPALIVE_MS="${DS4_METAL_QUEUE_KEEPALIVE_MS:-250}"

echo "starting ds4-server: $MODEL, ${VISION:+vision $VISION, }ctx $CTX, disk KV cache $KV_DIR ($KV_MB MB), $HOST:$PORT"
nohup ./ds4-server \
    --model "$MODEL" \
    ${VISION:+--vision "$VISION"} \
    --ctx "$CTX" \
    --host "$HOST" \
    --port "$PORT" \
    --kv-disk-dir "$KV_DIR" \
    --kv-disk-space-mb "$KV_MB" \
    --kv-cache-cold-max-tokens "$COLD_MAX" \
    ${COLD_ANCHOR:+--kv-cache-cold-anchor "$COLD_ANCHOR"} \
    >> "$LOG" 2>&1 < /dev/null &
pid=$!
echo "pid $pid, log $PWD/$LOG"

# Mapping 340 GB takes a while; give up only if the process dies or 10 minutes pass.
for i in $(seq 1 600); do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ds4-server exited during startup — the last lines of $LOG:"
        tail -n 20 "$LOG"
        exit 1
    fi
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "ds4-server is up after ${i}s: $(curl -s "http://127.0.0.1:$PORT/v1/models")"
        echo "follow it with: tail -f $PWD/$LOG"
        exit 0
    fi
    sleep 1
done
echo "ds4-server did not answer on port $PORT within 10 minutes — it is still running as pid $pid; see $LOG"
exit 1
