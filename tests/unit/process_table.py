"""A machine's process table as the start and stop scripts ask about it: the tools they look
things up with as bash functions (sourced through $BASH_ENV, see test_start_sh.py for why
functions) answering from files a test writes — `lsof` and `pgrep` for the Mac's branch; `ss`,
GNU `stat` on /proc, `pgrep`, `systemctl`, `systemd-run` and `loginctl` for Linux's. Either
branch runs on either system (`KEEP_OS`), so nothing here may need the real tool of one of them.

The stubs honour what the scripts pass — the user (`-u`), the ports (`-iTCP:`) and a process's
working directory (`-p <pid> -d cwd`, told in the fields `-F` names: the device and the inode for
`D` and `i`, the name for `n`) — so a script that drops the narrowing to its own user, asks
for another role's port or matches a process of another checkout by name gets the wrong answer
here exactly as it would on a Mac with two users. A pid the table does not hold has no readable
working directory. On Linux `ss` names every listener's pid, as it does for root: a script that
does not check whose the pid is takes the other user's. Every call is recorded.

The table's files are read with bash's own `read` (`row`), never awk or grep: a Linux start asks
these doubles some thirty questions, and a process for each was most of such a test's time."""
import os
from pathlib import Path

OTHER_USER = os.getuid() + 1          # the machine's second account
OTHER_CHECKOUT = "/Users/marcus/KeepElsewhere"
ELSEWHERE_DEVICE, ELSEWHERE_INODE = "D0x2f", "i47"      # how lsof -FDi tells a folder that is not on this disk
UNIT_CGROUP = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/keep.service"
SESSION_CGROUP = "0::/user.slice/user-1000.slice/session-3.scope"
SCOPE_CGROUP = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/run-r47.scope"

_BSD = os.uname().sysname == 'Darwin'
# The device, the inode and the size of a file from the system's real stat, whichever it is.
# ($follow is GNU stat's -L where the caller passed it: a folder named through a symlink is the folder.)
_REAL_STAT = "command /usr/bin/stat $follow -f '%d %i %z'" if _BSD else "command /usr/bin/stat $follow -c '%d %i %s'"
# role.sh's Mac branch asks /usr/bin/stat by its path, in BSD's words (`-f <format> <file>`).
# Where that file is GNU's, a function of that name answers for it — bash looks a function up
# before the path, slashes or not — in the formats the scripts and the lsof doubles use.
_BSD_STAT_SHIM = f"""
/usr/bin/stat() {{
    local device inode size follow=""
    read -r device inode size <<< "$({_REAL_STAT} "$3" 2>/dev/null)"
    [ -n "$device" ] || return 1
    case "$2" in
        '0x%Xd %i') printf '0x%x %s\\n' "$device" "$inode" ;;
        'D0x%Xd') printf 'D0x%x\\n' "$device" ;;
        'i%i') printf 'i%s\\n' "$inode" ;;
        'D0x%Xd%ni%i') printf 'D0x%x\\ni%s\\n' "$device" "$inode" ;;
        '%z') printf '%s\\n' "$size" ;;
        *) echo "the BSD stat double does not know the format $2" >&2; return 1 ;;
    esac
}}
"""
BSD_STAT = '' if _BSD else _BSD_STAT_SHIM


def stubs(table: Path) -> str:
    """The bash functions; `table` is the directory the table's files and the call records live in."""
    return BSD_STAT + f"""
# row <file> <column> <value>: the first line of the table's file whose column (1 or 2) holds the
# value, left in `row_1`, `row_2` and `row_3` (the rest of the line); 1 when there is none.
row() {{
    row_1=""; row_2=""; row_3=""
    local a b c
    [ -f "{table}/$1" ] || return 1
    while read -r a b c; do
        if {{ [ "$2" = 1 ] && [ "$a" = "$3" ]; }} || {{ [ "$2" = 2 ] && [ "$b" = "$3" ]; }}; then
            row_1=$a; row_2=$b; row_3=$c
            return 0
        fi
    done < "{table}/$1"
    return 1
}}
lsof() {{
    echo "$*" >> "{table}/lsof_calls"
    local uid="" pid="" cwd="" fields="" ports=" " prev="" a
    for a in "$@"; do
        case "$prev" in -u) uid=$a ;; -p) pid=$a ;; -d) [ "$a" = cwd ] && cwd=1 ;; esac
        case "$a" in -iTCP:*) ports="$ports${{a#-iTCP:}} " ;; -F*) fields=${{a#-F}} ;; esac
        prev=$a
    done
    if [ -n "$cwd" ]; then
        local dir
        row processes 1 "$pid"; dir=$row_3
        [ -n "$dir" ] || return 1
        printf 'p%s\\nfcwd\\n' "$pid"
        [ "$dir" != - ] || dir=$(pwd -P)
        if [ -d "$dir" ]; then
            case "$fields" in *D*) /usr/bin/stat -f 'D0x%Xd' "$dir" ;; esac
            case "$fields" in *i*) /usr/bin/stat -f 'i%i' "$dir" ;; esac
        else
            case "$fields" in *D*) echo "{ELSEWHERE_DEVICE}" ;; esac
            case "$fields" in *i*) echo "{ELSEWHERE_INODE}" ;; esac
        fi
        case "$fields" in *n*) echo "n$dir" ;; esac
        return 0
    fi
    local port lpid luid found=1
    while read -r port lpid luid; do
        case "$ports" in *" $port "*) ;; *) continue ;; esac
        [ -z "$uid" ] || [ "$uid" = "$luid" ] || continue
        echo "$lpid"; found=0
    done < "{table}/listeners"
    return $found
}}
pgrep() {{
    echo "$*" >> "{table}/pgrep_calls"
    local uid="" prev="" a ppid puid dir found=1
    for a in "$@"; do [ "$prev" = -u ] && uid=$a; prev=$a; done
    while read -r ppid puid dir; do
        [ -z "$uid" ] || [ "$uid" = "$puid" ] || continue
        echo "$ppid"; found=0
    done < "{table}/processes"
    return $found
}}
# Linux. `ss -Hltnp`: every listening socket of the machine, each with its pid.
ss() {{
    echo "$*" >> "{table}/ss_calls"
    local port lpid luid
    while read -r port lpid luid; do
        printf 'LISTEN 0      128          0.0.0.0:%s       0.0.0.0:*    users:(("python",pid=%s,fd=7))\\n' "$port" "$lpid"
    done < "{table}/listeners"
    echo 'LISTEN 0      4096   127.0.0.53%lo:53         0.0.0.0:*'
    echo 'LISTEN 0      128             [::]:22            [::]:*'
}}
# GNU stat as role.sh's Linux branch asks it: `-c '%d %i'` of a folder or of /proc/<pid>/cwd —
# which only the process's own user can follow —, `-c %u` of /proc/<pid>, `-c %s` of a file.
stat() {{
    echo "$*" >> "{table}/stat_calls"
    local format="" target="" prev="" follow="" a pid owner dir device inode size
    for a in "$@"; do [ "$prev" = -c ] && format=$a; [ "$a" != -L ] || follow=-L; target=$a; prev=$a; done
    case "$target" in
        /proc/*)
            pid=${{target#/proc/}}; pid=${{pid%%/*}}
            row processes 1 "$pid"; owner=$row_2; dir=$row_3
            case "$target" in
                */cwd)
                    [ -n "$dir" ] && [ "$owner" = "$UID" ] || return 1
                    [ "$dir" != - ] || dir=.
                    [ -d "$dir" ] || {{ echo "47 47"; return 0; }}
                    target=$dir ;;
                *)
                    [ -n "$owner" ] || {{ row listeners 2 "$pid"; owner=$row_3; }}
                    [ -n "$owner" ] || return 1
                    echo "$owner"; return 0 ;;
            esac ;;
    esac
    read -r device inode size <<< "$({_REAL_STAT} "$target" 2>/dev/null)"
    [ -n "$device" ] || return 1
    case "$format" in
        '%d %i') echo "$device $inode" ;;
        '%s') echo "$size" ;;
        *) echo "the GNU stat double does not know the format $format" >&2; return 1 ;;
    esac
}}
# /proc/<pid>/cgroup, where the scripts read whether a process is the unit's; any other file is cat's own.
cat() {{
    case "$1" in
        /proc/*/cgroup)
            local pid=${{1#/proc/}} line
            pid=${{pid%/cgroup}}
            if [ "$pid" = self ] && [ -n "$KEEP_TEST_IN_SCOPE" ]; then echo "{SCOPE_CGROUP}"; return 0; fi
            row cgroups 1 "$pid"; line=$row_2
            echo "${{line:-{SESSION_CGROUP}}}" ;;
        *) command cat "$@" ;;
    esac
}}
# The verbs named in systemctl_fails fail; `restart` runs on_restart, where a test says what the
# started server does.
systemctl() {{
    echo "$*" >> "{table}/systemctl_calls"
    local a
    for a in "$@"; do
        if row systemctl_fails 1 "$a"; then
            echo "Failed to connect to bus: No medium found" >&2
            return 1
        fi
    done
    case " $* " in *" restart "*) [ ! -f "{table}/on_restart" ] || . "{table}/on_restart" ;; esac
    return 0
}}
# `systemd-run --user --scope -- <command>`: the command, run where the unit's stop does not reach.
systemd-run() {{
    echo "$*" >> "{table}/systemd_run_calls"
    [ ! -f "{table}/no_scope" ] || return 1
    while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done
    shift
    if [ -f "{table}/scope_stays_in_unit" ]; then "$@"; else KEEP_TEST_IN_SCOPE=1 "$@"; fi
}}
loginctl() {{ echo "$*" >> "{table}/loginctl_calls"; }}
"""


class ProcessTable:
    """What runs on the machine: Keep processes (`python floxy_ai_server.py`, whoever owns them and
    whichever checkout they run from), the ports that are listened on and, on Linux, which
    processes are the systemd unit's."""

    def __init__(self, directory: Path):
        self.dir = directory
        for name in ('listeners', 'processes', 'cgroups'):
            (directory / name).write_text('')

    def listener(self, port: int, pid: int, user: int | None = None):
        """`pid` listens on `port`; `user` defaults to the one running the script."""
        with (self.dir / 'listeners').open('a') as f:
            f.write(f"{port} {pid} {os.getuid() if user is None else user}\n")

    def process(self, pid: int, user: int | None = None, checkout: str | None = None):
        """A Keep process; by default the script's own user's, running from the script's checkout."""
        with (self.dir / 'processes').open('a') as f:
            f.write(f"{pid} {os.getuid() if user is None else user} {checkout or '-'}\n")

    def in_unit(self, pid: int | str = 'self'):
        """`pid` runs in keep.service's control group; without one, the script itself does."""
        with (self.dir / 'cgroups').open('a') as f:
            f.write(f"{pid} {UNIT_CGROUP}\n")

    def systemctl_fails(self, verb: str):
        with (self.dir / 'systemctl_fails').open('a') as f:
            f.write(verb + '\n')

    def on_restart(self, bash: str):
        """What `systemctl --user restart` does besides being recorded."""
        (self.dir / 'on_restart').write_text(bash + '\n')

    def _calls(self, name: str) -> list[str]:
        path = self.dir / f'{name}_calls'
        return path.read_text().splitlines() if path.exists() else []

    @property
    def ports_asked(self) -> set[int]:
        """Every TCP port any lsof call named."""
        return {int(word.removeprefix('-iTCP:')) for call in self._calls('lsof')
                for word in call.split() if word.startswith('-iTCP:')}

    @property
    def lookups(self) -> list[str]:
        """Every lsof and pgrep call, as its argument line."""
        return self._calls('lsof') + self._calls('pgrep')

    @property
    def systemctl_calls(self) -> list[str]:
        return self._calls('systemctl')

    @property
    def systemd_run_calls(self) -> list[str]:
        return self._calls('systemd_run')
