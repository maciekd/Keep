#!/bin/bash
# Sourced by start.sh on a Mac, after scripts/role.sh and the role check: the old process is
# stopped, the ports are waited for, and the new one is launched through Terminal — so the Mac's
# privacy grants apply to it — in a window that is closed again. Linux has scripts/launch_linux.sh
# in its place. Leaves `log_offset`, where the startup log ended before the launch, for start.sh.

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
