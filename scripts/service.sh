#!/bin/bash
# Sourced on Linux by start.sh (scripts/launch_linux.sh) and stop.sh, from the repository root and
# after scripts/role.sh: the systemd user unit that runs the Keep server there. One unit stands
# where the Mac has Terminal, the login applet and the launchd watchdog — it starts the server at
# boot (with `loginctl enable-linger`, which the installer runs once), restarts it when it dies
# and keeps it down after a stop:
#   systemctl --user status keep      journalctl --user -u keep
# It is plain bash and awk: start.sh writes the unit before any Python runs.
#
# ensure_unit says why it could not do its work in `unit_problem` and returns 1, for the caller
# to log and stop on; what it did, it logs through the caller's `rlog`.

UNIT=keep.service
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_path="$unit_dir/$UNIT"
unit_problem=""

# unit_text <checkout>: the unit, for the checkout at that path.
# PATH: a unit gets the user manager's, which has no ~/.local/bin — where the Claude CLI the nightly
# code-quality run calls is installed, and uv, and whatever else the user put there — so the line
# names the whole of it: the user's two folders, the system's as systemd has them by default, and
# /snap/bin, which Ubuntu adds for a login and where its Chromium lives. %h is the home folder.
unit_text() {
    cat <<UNIT_TEXT
# Written by $1/scripts/service.sh at every start — edits are overwritten.
[Unit]
Description=Keep
After=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
WorkingDirectory=$1
ExecStart=$1/.venv/bin/python floxy_ai_server.py
StandardOutput=append:$1/log/raw-floxy.log
StandardError=inherit
Environment=PYTHONUTF8=1
Environment="PATH=%h/.local/bin:%h/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
UNIT_TEXT
}

# unit_checkout: the folder the installed unit runs in, as its WorkingDirectory line has it;
# nothing when there is no unit or it names none.
unit_checkout() {
    awk '/^WorkingDirectory=/ { dir = substr($0, 18) } END { if (dir != "") print dir }' "$unit_path" 2>/dev/null
}

# unit_is_ours: there is a unit and it runs this checkout — the same folder by device and inode,
# as role.sh tells every folder, never by name.
unit_is_ours() {
    local dir
    dir=$(unit_checkout)
    [ -n "$dir" ] && [ -n "$here" ] && [ "$(stat -L -c '%d %i' "$dir" 2>/dev/null)" = "$here" ]
}

# cgroup_of <pid|self>: the control groups of the process, as /proc has them.
cgroup_of() { cat "/proc/$1/cgroup" 2>/dev/null; }

# in_unit [<pid>]: the process — this shell when none is named — runs in the unit's control group:
# it is the server systemd started, or something that server spawned.
in_unit() { cgroup_of "${1:-self}" | grep -Eq "/$UNIT(/|\$)"; }

# ensure_unit: the unit is installed for this checkout, known to systemd and enabled. The file is
# written only when its text differs, and systemd re-reads it only then.
ensure_unit() {
    local checkout other text out
    # A space, a quote, a backslash, $ and % each mean something in a unit file, and differently
    # in ExecStart than in the two path settings. Such a folder is refused rather than escaped
    # three ways: a unit that starts another program than the one meant is worse than no unit.
    local unsafe="[[:space:][:cntrl:]\"'\\\$%]"
    unit_problem=""
    checkout=$(pwd -P)
    if [[ "$checkout" =~ $unsafe ]]; then
        unit_problem="systemd cannot be given this folder: '$checkout' has a space, a quote, a backslash, \$ or % in its path — move the Keep folder to a path without them (~/Keep)"
        return 1
    fi
    if [ -e "$unit_path" ] && ! unit_is_ours; then
        other=$(unit_checkout)
        if [ -z "$other" ]; then
            unit_problem="$unit_path exists and is not a unit Keep wrote (it names no WorkingDirectory), so it is not replaced with one for $checkout — move it away and start again"
            return 1
        fi
        if [ -d "$other" ]; then
            unit_problem="$unit_path already runs the Keep in $other, and this one is $checkout — one Keep per user. To move the service here: 'systemctl --user disable --now $UNIT', remove that file and start again"
            return 1
        fi
        rlog "$unit_path points at $other, which is gone — rewriting it for $checkout"
    fi
    text=$(unit_text "$checkout")
    if [ -f "$unit_path" ] && [ "$(cat "$unit_path")" = "$text" ]; then
        rlog "$unit_path is up to date"
    else
        if ! { mkdir -p "$unit_dir" && printf '%s\n' "$text" > "$unit_path"; } 2>/dev/null; then
            unit_problem="$unit_path could not be written"
            return 1
        fi
        rlog "wrote $unit_path — telling systemd to read it"
        if ! out=$(systemctl --user daemon-reload 2>&1); then
            unit_problem="'systemctl --user daemon-reload' failed ($(echo $out)) — Keep needs a systemd user session; over ssh, log in as the user rather than through su or sudo"
            return 1
        fi
    fi
    if ! out=$(systemctl --user enable --quiet "$UNIT" 2>&1); then
        unit_problem="'systemctl --user enable $UNIT' failed ($(echo $out)) — Keep needs a systemd user session; over ssh, log in as the user rather than through su or sudo"
        return 1
    fi
    return 0
}
