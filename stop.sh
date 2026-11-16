#!/bin/bash
# Stops what this install runs, the Keep server or an iMessage satellite: every floxy_ai_server
# process of this user that runs from this checkout (the test suite's instances included) gets
# SIGTERM, which both roles answer with a clean shutdown — and so does whatever of this user's
# listens on the role's own ports, unless it is seen to run from another checkout: a process
# whose working directory cannot be read is found by its port alone. Another macOS user's Keep
# on the same Mac, and another checkout of this user's, are not this install's and are left
# alone (scripts/role.sh). It only stops: with `watchdog: true` in config.yaml launchd starts
# the process again within five minutes.
# On Linux the server is the systemd user unit keep.service (scripts/service.sh): the unit is
# stopped first — when there is one and it is this checkout's — so systemd does not take the
# server's end for a crash, and what is left of this checkout (the test suite's instances, a
# server started by hand) is stopped as on the Mac. The unit stays enabled: it starts again at
# the next boot, and at the next ./start.sh.
cd "$(dirname "$0")"
. scripts/role.sh || { echo "stop.sh: scripts/role.sh is missing — this checkout is incomplete" >&2; exit 1; }

RLOG=log/restart.log
rlog() { mkdir -p log; echo "[$(date '+%Y-%m-%d %H:%M:%S')] stop.sh: $*" >> "$RLOG"; echo "$*"; }

# port_holders: this user's listeners on the role's ports that are this install's as far as can
# be told — running from this checkout, or from a directory lsof cannot read.
port_holders() {
    local pid cwd
    for pid in $(own_listeners $listen_ports); do
        cwd=$(cwd_of "$pid")
        if [ -z "$cwd" ] || [ "$cwd" = "$here" ]; then echo "$pid"; fi
    done
}

unit_stopped=""
if [ "$os" = Linux ] && [ -f scripts/service.sh ]; then
    . scripts/service.sh
    if ! unit_is_ours; then
        rlog "no $UNIT of this checkout ($unit_path) — no unit to stop"
    elif out=$(systemctl --user stop "$UNIT" 2>&1); then
        unit_stopped=1
        rlog "stopped $UNIT — it stays down until ./start.sh or the next boot ('systemctl --user disable $UNIT' keeps it from starting at boot)"
    else
        rlog "'systemctl --user stop $UNIT' failed: $(echo $out)"
    fi
fi

pids=$( { own_processes 'floxy_ai_server'; port_holders; } | sort -un)
if [ -z "$pids" ]; then
    [ -n "$unit_stopped" ] || rlog "no Keep $what of this install is running — nothing to stop"
    exit 0
fi
rlog "stopping the Keep $what, pid(s): $(echo $pids)"
kill $pids 2>/dev/null
[ "$os" != Linux ] || exit 0     # the watchdog is the Mac's; on Linux systemd leaves a stopped unit down
case "$(config_value watchdog | tr '[:upper:]' '[:lower:]')" in
    true|yes|on) echo "config.yaml says watchdog: true — launchd starts it again within 5 minutes. To keep it down, set" \
                      "watchdog: false and run ./start.sh once (a start applies the key), then stop again." ;;
esac
