#!/bin/bash
# Sourced by start.sh on Linux, after scripts/role.sh and the role check, where the Mac has
# scripts/launch_mac.sh: the server is the systemd user unit keep.service (scripts/service.sh),
# so a start is `systemctl --user restart` — systemd stops the old process and starts the new
# one, at boot and after a crash too. Leaves `log_offset`, where the startup log ended before
# the launch, for start.sh.

[ -f scripts/service.sh ] || { rlog "ERROR: scripts/service.sh is missing — this checkout is incomplete, nothing stopped, nothing started"; exit 1; }
. scripts/service.sh

# Without ss no listener is ever seen: every port would pass for free and every start for one
# that never came up.
for tool in systemctl ss; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        rlog "ERROR: $tool is not installed (apt: systemd, iproute2) — nothing stopped, nothing started"
        exit 1
    fi
done

# The server restarts itself by spawning this script (common/Updater.relaunch), which puts it in
# the unit's control group — and the restart below stops everything in there, this script
# included, before the PR-trial watch further down has run. So a start that finds itself inside
# the unit runs again in a scope of its own, which the unit's stop does not reach. The copy left
# behind only waits for that one and ends with the unit. KEEP_START_SCOPED keeps a scope that
# did not take the script out of the unit from being tried twice.
if in_unit; then
    if [ -n "${KEEP_START_SCOPED:-}" ]; then
        rlog "WARNING: still inside $UNIT after systemd-run — going on here; the restart ends this script too, so a PR trial is not watched"
    elif systemd-run --user --scope --quiet -- true >/dev/null 2>&1; then
        rlog "running inside $UNIT — going on in a scope of its own, which the restart leaves alone"
        KEEP_START_SCOPED=1 systemd-run --user --scope --quiet -- bash "$PWD/$(basename "$0")" "$@"
        exit $?
    else
        rlog "WARNING: running inside $UNIT and systemd-run gives no scope — going on here; the restart ends this script too, so a PR trial is not watched"
    fi
fi

if [ -n "${1:-}" ]; then
    rlog "old pid $1 is not killed here: it is the unit's process, and the restart stops it"
fi

# A listener the unit does not run is a server started by hand (`python floxy_ai_server.py` in a
# terminal). systemd would start a second one beside it, which loads its models and dies on the
# port — five times, until the start limit. Whose process that is, is not this script's to guess.
foreign=""
for p in $(own_listeners $listen_ports); do
    in_unit "$p" || foreign="$foreign $p"
done
if [ -n "$foreign" ]; then
    rlog "ERROR: $what port(s) $listen_ports held by pid(s)$foreign, which $UNIT does not run — a server started by hand? Stop it (./stop.sh) and start again. Nothing stopped, nothing started"
    exit 1
fi

if ! ensure_unit; then
    rlog "ERROR: $unit_problem — nothing stopped, nothing started"
    exit 1
fi

# As on the Mac: the server can fail within milliseconds of starting, so where the log ends is
# remembered before the launch. The unit appends the process's own output to log/raw-floxy.log,
# and systemd creates that file but not its folder.
mkdir -p log && touch log/verbose-floxy.out
log_offset=$(stat -c %s log/verbose-floxy.out 2>/dev/null || echo 0)

# A unit that used up its start limit (a config the server refused five times) refuses a start by
# hand as well until the limit is reset.
systemctl --user reset-failed "$UNIT" >/dev/null 2>&1
rlog "restarting $UNIT — systemd stops the old $what and starts the new one..."
if ! out=$(systemctl --user restart "$UNIT" 2>&1); then
    rlog "ERROR: 'systemctl --user restart $UNIT' failed: $(echo $out) — see 'systemctl --user status $UNIT'"
    exit 1
fi
rlog "$UNIT restarted (status: systemctl --user status $UNIT, output: log/raw-floxy.log)"
