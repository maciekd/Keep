#!/bin/bash
# Runs inside the Terminal window that start.sh opens: Terminal is then the process macOS
# holds responsible for the server, so its microphone / Photos / Full Disk Access grants
# apply. The server leaves the window's session (setsid) before it starts, so start.sh can
# close the window without Terminal asking about running processes and without hanging
# up the server; the tty is reported only once that has happened.
# macOS only: on Linux the systemd user unit runs the server (scripts/service.sh).
if [ "${KEEP_OS:-$(uname -s)}" = Linux ]; then      # as scripts/role.sh tells the system
    echo "cli_start.sh is for macOS. On Linux run ./start.sh — the server is the systemd user unit keep.service." >&2
    exit 1
fi
cd "$(dirname "$0")/.."
source .venv/bin/activate
python3 -c 'import os, sys; os.setsid(); open(sys.argv[1], "w").write(sys.argv[2]); os.execv(sys.executable, [sys.executable] + sys.argv[3:])' \
    log/cli_start.tty "$(tty)" floxy_ai_server.py < /dev/null >> log/raw-floxy.log 2>&1 &
