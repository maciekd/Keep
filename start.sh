#!/bin/bash
cd "$(dirname "$0")"

RLOG=log/restart.log
rlog() {
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$line" >> "$RLOG"
  # Dimmed on the terminal: it's bookkeeping, the server log that follows is what matters
  [ -t 1 ] && printf '\033[2;90m%s\033[0m\n' "$line" || echo "$line"
}

# What this install runs — the Keep server or an iMessage satellite — with its ports, and the
# lookups that see only this user's processes of this checkout (scripts/role.sh says why).
. scripts/role.sh || { echo "start.sh: scripts/role.sh is missing — this checkout is incomplete" >&2; exit 1; }

rlog "Start script invoked (caller-supplied old pid: ${1:-none}, role: $role)"

# A role or a port the process would refuse to start on stops the script here, before anything
# is killed: the old process keeps running, as it does after any failed update.
if [ -n "$role_problem" ]; then
    rlog "ERROR: $role_problem — nothing stopped, nothing started"
    exit 1
fi

# Old server pids from three sources: the pid handed to us by the restarting
# server itself, the owners of the server's listen ports, and an argv match
# (anchored with $ to exclude testsuite instances). pgrep -f alone proved
# unreliable when start.sh is spawned from inside the server — it missed the
# running process and the replacement then died on a port collision.
# A satellite is found the same three ways, on its own port (Updater.relaunch hands
# its pid over exactly as the server's restart does). The two lookups never return
# another macOS user's process or another checkout's.
old_pids="${1:-} $(own_listeners $listen_ports) $(own_processes "$KEEP_PROCESS")"
old_pids=$(printf '%s\n' $old_pids | grep -E '^[0-9]+$' | sort -un)

if [ -n "$old_pids" ]; then
    rlog "Stopping old $what pid(s): $(echo $old_pids)"
    kill $old_pids 2>/dev/null
    # Wait until they're actually gone (up to 10s)
    alive=""
    for i in $(seq 1 20); do
        alive=""
        for p in $old_pids; do
            kill -0 "$p" 2>/dev/null && alive="$alive $p"
        done
        [ -z "$alive" ] && break
        sleep 0.5
    done
    if [ -n "$alive" ]; then
        rlog "survivor(s) after 10s, force-killing:$alive"
        kill -9 $alive 2>/dev/null
        sleep 1
    fi
else
    rlog "no old $what process found — nothing to stop"
fi

# Never launch into occupied ports — the new server would spend ~25s loading
# models and then die on bind (Errno 48).
busy=""
for i in $(seq 1 10); do
    busy=$(own_listeners $listen_ports)
    [ -z "$busy" ] && break
    sleep 0.5
done
if [ -n "$busy" ]; then
    rlog "ERROR: $what port(s) still held by pid(s) $(echo $busy) — aborting launch"
    exit 1
fi

rlog "Ports free — launching new $what via Terminal (so the Mac's grants apply to it)..."
# Remember where the log ends before the launch: the server can fail within
# milliseconds of starting (dependency check, bad config), and a `tail -n 0`
# attached after the sleep below would begin past those lines and show nothing.
# On the very first start the log does not exist yet and tail would give up on it;
# the server opens it in append mode, so creating it empty here is harmless.
mkdir -p log && touch log/verbose-floxy.out
log_offset=$(/usr/bin/stat -f %z log/verbose-floxy.out 2>/dev/null || echo 0)  # BSD stat even with GNU coreutils on PATH
rm -f log/cli_start.tty
open -a Terminal "$PWD/scripts/cli_start.sh"
# cli_start.sh reports its tty once the server has left that window's session. Close
# that one window (or tab, if Terminal opened it as one) and nothing else of the
# user's; it would otherwise linger as "[Process completed]" after every restart.
for i in $(seq 1 40); do
    [ -s log/cli_start.tty ] && break
    sleep 0.25
done
if [ -s log/cli_start.tty ]; then
    rlog "closing the launch window on $(cat log/cli_start.tty)"
    osascript - "$(cat log/cli_start.tty)" >/dev/null 2>&1 <<'APPLESCRIPT'
on run argv
    tell application "Terminal"
        repeat with w in windows
            repeat with t in tabs of w
                if tty of t is (item 1 of argv) then
                    if (count of tabs of w) is 1 then
                        close w
                    else
                        close t
                    end if
                end if
            end repeat
        end repeat
    end tell
end run
APPLESCRIPT
else
    rlog "cli_start.sh did not report its tty within 10s — leaving its window open"
fi

# A PR trial (src/assistant/agents/PullRequestTrial.py) hands over the branch to fall back to as $2:
# the server just launched runs a pull request's code, and if it is not listening within
# TRIAL_TIMEOUT seconds — or its process is gone before that — the previous branch is checked
# out again and the server relaunched on it, so a PR that cannot boot never leaves a dead server.
# The note left in log/ is what the relaunched server tells the user.
fallback="${2:-}"
if [ -n "$fallback" ]; then
    timeout="${TRIAL_TIMEOUT:-120}"
    rlog "PR trial: waiting up to ${timeout}s for the server to listen, falling back to $fallback otherwise"
    up=""
    for i in $(seq 1 "$timeout"); do
        [ -n "$(own_listeners $up_port)" ] && up=1 && break
        if [ "$i" -gt 15 ] && [ -z "$(own_processes "$KEEP_PROCESS")" ]; then
            rlog "PR trial: the server process is gone after ${i}s"
            break
        fi
        sleep 1
    done
    if [ -n "$up" ]; then
        rlog "PR trial: the server is up after ${i}s"
    else
        rlog "PR trial: the server did not come up — going back to $fallback"
        {
            echo "The server did not come up on the pull request's code within ${i}s, so it is back on $fallback. The last lines it wrote:"
            tail -n 30 log/raw-floxy.log 2>/dev/null || echo "(no output)"
        } > log/trial_rollback.txt
        if git checkout "$fallback" >> "$RLOG" 2>&1; then
            exec bash "$0"
        fi
        rlog "ERROR: git checkout $fallback failed — the server stays down on the pull request's code"
        exit 1
    fi
fi

# Only interactively — when spawned detached (auto-restart), a tail here would
# leak as an orphan process on every restart. Show the startup log until the
# server's TUI port is listening (at least 5s, at most 60s), then hand the
# terminal to the TUI; tui.py reconnects by itself if the server isn't up yet.
if [ -t 1 ]; then
    tail -c +$((log_offset + 1)) -f log/verbose-floxy.out &
    tail_pid=$!
    # A backgrounded job ignores SIGINT, so Ctrl-C would kill the script and
    # leave the tail running in the terminal — take it down on any exit.
    trap 'kill $tail_pid 2>/dev/null' EXIT
    trap 'rlog "interrupted — leaving the $what running"; exit 130' INT TERM
    # A satellite has no console to hand the terminal to. The startup log is shown until its
    # port listens (at most 60s) or its process is gone again — a refused config ends it within
    # a second or two — and then the terminal gets one plain verdict and its prompt back; the
    # satellite runs on without it, like a server whose console was closed.
    if [ -z "$console_port" ]; then
        up=""; gone=""
        for i in $(seq 1 120); do
            [ -n "$(own_listeners $up_port)" ] && up=1 && break
            [ "$i" -gt 10 ] && [ -z "$(own_processes "$KEEP_PROCESS")" ] && gone=1 && break
            sleep 0.5
        done
        kill $tail_pid 2>/dev/null
        if [ -n "$up" ]; then
            rlog "the satellite listens on port $up_port after ${i}x0.5s — no console to hand over to"
            echo
            echo "The Keep iMessage satellite is up on port $up_port and keeps running without this window."
            echo "Log: $PWD/log/floxy.out    Stop: $PWD/stop.sh"
            exit 0
        fi
        rlog "the satellite is not listening on port $up_port after ${i}x0.5s${gone:+ and its process is gone}"
        echo
        if [ -n "$gone" ]; then
            echo "The Keep iMessage satellite stopped while starting — the lines above say why."
        else
            echo "The Keep iMessage satellite is not answering on port $up_port after 60s. It may be waiting for a"
            echo "macOS dialog on this Mac's screen; $PWD/log/floxy.out says where it is."
        fi
        exit 1
    fi
    for i in $(seq 1 120); do
        [ "$i" -gt 10 ] && [ -n "$(own_listeners $console_port)" ] && break
        sleep 0.5
    done
    kill $tail_pid 2>/dev/null
    rlog "handing the terminal to tui.py after ${i}x0.5s of tail"
    exec .venv/bin/python3 tui.py localhost
fi
