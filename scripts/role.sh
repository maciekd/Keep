#!/bin/bash
# Sourced by start.sh, stop.sh, autostart.sh and keep.sh from the repository root: what this
# install runs, the ports that go with it, and which processes on the Mac are its own.
#
# An install is a Keep server or an iMessage satellite (common/Role.py). The root key `role` of
# config.yaml decides, and a satellite listens on imessage_satellite.port alone — never on the
# server's 8888 and 43523 or the console's 5151. Both keys are read here with awk, the way
# install.sh reads the same file: these scripts answer in milliseconds and before any Python runs.
# The reading follows the Python side, so the two never disagree about what is running here:
#   - role: absent or empty is `server`; compared in lower case with the quotes off, as
#     Role.current() does. Anything but `server` and `imessage_satellite` is no role.
#   - imessage_satellite.port: absent or empty is 8898; otherwise a plain decimal 1-65535, what
#     yaml hands Satellite.py as an int (a quoted number is refused there, and so here).
#   - only the block style install.sh writes is understood: `key: value` lines from the first
#     column, a section's keys indented under it, a key in quotes or not. YAML has more ways to
#     write the same file — the whole of it in braces or indented, a value on the lines under its
#     key, a section in braces — and those are not read as "the key is absent": that would make a
#     server here of what Python starts as a satellite, or watch another port than the one bound.
#     They are a `role_problem` (config_trouble).
# What the process itself would refuse to start on is `role_problem` here, and start.sh stops on
# it before it kills anything: taking an unreadable role for "server" would take down the owners
# of the server's ports on a Mac that was meant to run a satellite.
#
# A satellite usually lives under a second macOS user of the Mac that runs the server, and a
# process list is the whole Mac's. A script that took the other user's process for "the old
# server" could not kill it: it would wait, force-kill in vain and then refuse to launch into
# ports it believed busy — and autostart.sh would never start a satellite at login, since "a
# server is already running". So every lookup is narrowed three ways: to the caller's own
# processes (-u), for a port lookup to the ports of the install's own role, and for a match by
# name to the processes running from this checkout (their working directory) — two installs under
# one user, a server and a satellite or a second checkout, are both `python floxy_ai_server.py`
# and must not stop each other.

me=$(id -u)
# This checkout as lsof tells a working directory with -FDi: the device and the inode of the
# folder, never its name. A name does not compare — lsof escapes what the caller's locale cannot
# print, and the login applet and launchd run without one (~/Kęp is K\xc4\x99p there), as it
# escapes a tab or a backslash in any locale, while pwd prints the bytes. The pair is the same
# however the folder was reached: through a symlink, in another case.
here=$(/usr/bin/stat -f '0x%Xd %i' . 2>/dev/null)

# The reader of config.yaml, for config_value and config_trouble. install.sh carries the same
# program between the same two marker lines (it runs before a checkout exists, so it cannot
# source this file); tests/unit/test_install_sh.py fails when the two differ. It runs in the C
# locale whatever the caller has, so a file reads the same from Terminal and from launchd.
# >>> config reader
CONFIG_READER='
    # The key a `key: value` line opens, without its quotes ("" when it opens none); what
    # follows the colon is left in `rest`.
    function key_of(line,    k) {
        rest = ""
        sub(/^[ \t]+/, "", line)
        if (line ~ /^-([ \t]|$)/) return ""
        if (!match(line, /^("[^"]*"|\047[^\047]*\047|[^][ \t:#"\047{},&*!|>%@?][^ \t:]*)[ \t]*:([ \t]|$)/)) return ""
        k = substr(line, 1, RLENGTH); rest = substr(line, RLENGTH + 1)
        sub(/[ \t]*:[ \t]?$/, "", k); gsub(/["\047]/, "", k)
        return k
    }
    function value_of(text) {
        sub(/^[ \t]+/, "", text); sub(/(^|[ \t]+)#.*$/, "", text); sub(/[ \t]+$/, "", text)
        return text
    }
    # The key was written without a value: one on the lines under it is not read here.
    function expect_nothing_under(name, indent) { waiting = name; waiting_indent = indent }
    NR == 1 { sub(/^\357\273\277/, "") }     # a byte order mark
    /^[ \t]*(#.*)?$/ { next }
    !started && /^(---|%)/ { next }
    {
        match($0, /^[ \t]*/); indent = RLENGTH
        key = key_of($0); value = value_of(rest)
        if (!started) {
            started = 1
            if (indent > 0 || key == "") shape = "it does not open with a `key: value` line in the first column"
        }
        if (waiting != "") {
            if (indent > waiting_indent || (indent == waiting_indent && $0 ~ /^[ \t]*-([ \t]|$)/))
                key_trouble = "the value of " waiting " stands on the lines under the key"
            waiting = ""
        }
        if (indent == 0) {
            section = key; child = -1
            if (key != top) next
            section_trouble = key_trouble = ""
            if (inner == "") { found = value; seen = 1; if (value == "") expect_nothing_under(top, 0) }
            else {
                seen = 0; found = ""
                if (value !~ /^(null|Null|NULL|~)?$/) section_trouble = top " is not a section of indented lines"
            }
            next
        }
        if (inner == "" || section != top) next
        if (child < 0) child = indent
        if (indent != child) next
        if (key == inner) {
            found = value; seen = 1; key_trouble = ""
            if (value == "") expect_nothing_under(top "." inner, indent)
        } else if (key == "<<") section_trouble = top " takes its keys from another section (<<)"
    }
    END {
        if (ask != "trouble") { if (seen) print found; exit }
        if (shape != "") print shape
        else if (section_trouble != "") print section_trouble
        else if (key_trouble != "") print key_trouble
    }
'
# <<< config reader

# config_value <root key> [<key under it>]: the value as written, without its comment — the last
# one when the key stands twice, as yaml reads it; nothing when config.yaml or the key is missing.
# A key under a section is one of the section's own lines, not of a section nested in it.
config_value() {
    LC_ALL=C awk -v top="$1" -v inner="${2:-}" -v ask=value "$CONFIG_READER" config.yaml 2>/dev/null
}

# config_trouble <root key> [<key under it>]: why config_value cannot be trusted for the key —
# config.yaml is written in a way this reader does not follow — and nothing when it can.
config_trouble() {
    LC_ALL=C awk -v top="$1" -v inner="${2:-}" -v ask=trouble "$CONFIG_READER" config.yaml 2>/dev/null
}

# An empty value and yaml's null are no role (a server); "null" in quotes is a word, as to Python.
role=$(config_value role)
case "$role" in
    ""|null|Null|NULL|"~"|'""'|"''") role=server ;;
    *) role=$(printf '%s\n' "$role" | awk '{ gsub(/["\047]/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); print tolower($0) }') ;;
esac
unreadable=$(config_trouble role)
[ -z "$unreadable" ] || role=unreadable

role_problem=""
case "$role" in
    server)
        what=server
        listen_ports="8888 43523"   # everything the process binds; a start needs them all free
        up_port=8888                # the one that says it is up
        console_port=5151 ;;
    imessage_satellite)
        what=satellite
        console_port=""             # no console: nothing to hand a terminal to
        up_port=$(config_value imessage_satellite port)
        unreadable=$(config_trouble imessage_satellite port)
        case "$up_port" in ""|null|"~") up_port=8898 ;; esac
        if [ -n "$unreadable" ]; then
            role_problem="config.yaml: $unreadable — write imessage_satellite.port as install.sh does, 'port: 8898' on a line indented under 'imessage_satellite:'"
            up_port=""
        elif ! [[ "$up_port" =~ ^[1-9][0-9]{0,4}$ ]] || [ "$up_port" -gt 65535 ]; then
            role_problem="imessage_satellite.port in config.yaml is not a port number ($up_port)"
            up_port=""
        fi
        listen_ports=$up_port ;;
    *)
        what=process
        listen_ports=""; up_port=""; console_port=""
        if [ -n "$unreadable" ]; then
            role_problem="config.yaml: $unreadable — whether this is a Keep server or an iMessage satellite cannot be told. Write the keys as lines of their own from the first column ('role: imessage_satellite'), as install.sh does"
        else
            role_problem="role: $role in config.yaml is not a role — remove the key for a Keep server, or set it to imessage_satellite"
        fi ;;
esac

# own_listeners <port>...: the pids of this user's processes listening on any of the TCP ports.
# (lsof exits 1 as soon as one of several ports has no listener, so callers read the output.)
own_listeners() {
    [ $# -gt 0 ] || return 0
    # shellcheck disable=SC2046
    lsof -a -u "$me" -t $(printf -- '-iTCP:%s ' "$@") -sTCP:LISTEN 2>/dev/null
    return 0
}

# cwd_of <pid>: the device and the inode of the process's working directory, as `here` has this
# checkout's; nothing when lsof cannot read it.
cwd_of() {
    lsof -a -p "$1" -d cwd -FDi 2>/dev/null |
        awk '/^D/ { device = substr($0, 2) } /^i/ { inode = substr($0, 2) }
             END { if (device != "" && inode != "") print device, inode }'
}

# own_processes <pgrep -f pattern>: the pids of this user's matching processes that run from this
# checkout. A process whose working directory cannot be read is not ours: of the two mistakes,
# not stopping an old process (the port lookup and the pid a restart hands over still find it) is
# the cheaper one than stopping another install's.
own_processes() {
    local pid
    [ -n "$here" ] || return 0
    for pid in $(pgrep -u "$me" -f "$1" 2>/dev/null); do
        if [ "$(cwd_of "$pid")" = "$here" ]; then
            echo "$pid"
        fi
    done
    return 0
}

# The process an install runs, either role: `python floxy_ai_server.py` with no argument (the
# $ keeps the test suite's instances, which carry a port and "testsuite", out).
KEEP_PROCESS='floxy_ai_server\.py$'
