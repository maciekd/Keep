#!/bin/bash
# Guards a PR trial (src/assistant/agents/PrTrial.py): waits for the marker the server writes
# once it is up on the PR's code; when it never comes, checks the previous branch out again
# and restarts. The server runs a copy of this script taken BEFORE it checked the PR out
# (cache/trial/watchdog.sh), so a PR cannot break its own way back. Runs with the repository
# root as its working directory.
#
#   trial_watchdog.sh <previous branch> [timeout seconds]

prev="$1"
timeout="${2:-80}"
dir=cache/trial

rlog() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] trial watchdog: $*" >> log/restart.log; }

if [ -z "$prev" ] || [ ! -f start.sh ]; then
    echo "usage: trial_watchdog.sh <previous branch> [timeout], from the repository root" >&2
    exit 2
fi

rlog "waiting up to ${timeout}s for the server to come up (way back: $prev)"
for i in $(seq 1 "$timeout"); do
    if [ -f "$dir/started" ]; then
        rlog "server is up after ${i}s — nothing to do"
        exit 0
    fi
    if [ ! -f "$dir/state.json" ]; then
        rlog "the trial state is gone — nothing to guard"
        exit 0
    fi
    sleep 1
done

rlog "server not up after ${timeout}s — going back to $prev"
tail -n 30 log/raw-floxy.log > "$dir/rolled_back" 2>/dev/null
[ -s "$dir/rolled_back" ] || echo "(the server wrote no output)" > "$dir/rolled_back"
if ! git checkout "$prev" >> log/restart.log 2>&1; then
    rlog "ERROR: git checkout $prev failed — the server stays down on the PR's code"
    exit 1
fi
rlog "on $prev again — restarting"
bash start.sh
