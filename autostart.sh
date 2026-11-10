#!/bin/bash
# The login item's script. "Keep Autostart.app" in this folder (common/Autostart.py builds it and
# registers it under System Settings > General > Login Items once the first-time setup sees Full
# Disk Access granted) runs it at login with no terminal attached. It starts the server the way it
# always starts — start.sh, through Terminal, so the Mac's grants apply — and opens no console;
# the Desktop shortcut (keep.sh) is for that. The watchdog (common/Watchdog.py, a launchd agent)
# runs the same applet every few minutes. A server that is already up is left alone, and so is a
# start.sh in flight: between its kill of the old server and the launch of the new one neither
# the port nor the process exists, and a second start.sh there would collide with it.
# An iMessage satellite install is started the same way — its process registers the applet itself,
# common/imessage/Satellite.stay_up, having no first-time setup — and "up" is then its own port: what is
# running is asked of this user's processes of this checkout only (scripts/role.sh), so the Keep
# server of the Mac's other user never passes for this install's process — the satellite of a
# second macOS user would otherwise not start at that user's login.
cd "$(dirname "$0")"

RLOG=log/restart.log
rlog() { mkdir -p log; echo "[$(date '+%Y-%m-%d %H:%M:%S')] autostart.sh: $*" >> "$RLOG"; }

. scripts/role.sh || { rlog "scripts/role.sh is missing — this checkout is incomplete, nothing started"; exit 1; }

if [ -n "$(own_listeners $up_port)" ] || [ -n "$(own_processes "$KEEP_PROCESS")" ]; then
    rlog "a $what is already running — nothing to start"
    exit 0
fi
if [ -n "$(own_processes '(^|[ /])start\.sh( |$)')" ]; then
    rlog "a start.sh is already running — nothing to start"
    exit 0
fi
rlog "no $what running — starting it"
exec ./start.sh
