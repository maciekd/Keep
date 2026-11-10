"""A Mac's process table as the start and stop scripts ask about it: `lsof` and `pgrep` as bash
functions (sourced through $BASH_ENV, see test_start_sh.py for why functions) answering from two
files a test writes.

The stubs honour what the scripts pass — the user (`-u`), the ports (`-iTCP:`) and a process's
working directory (`-p <pid> -d cwd`, told in the fields `-F` names: the device and the inode for
`D` and `i`, the name for `n`) — so a script that drops the narrowing to its own user, asks
for another role's port or matches a process of another checkout by name gets the wrong answer
here exactly as it would on a Mac with two users. A pid the table does not hold has no readable
working directory. Every call is recorded."""
import os
from pathlib import Path

OTHER_USER = os.getuid() + 1          # the Mac's second account
OTHER_CHECKOUT = "/Users/marcus/KeepElsewhere"
ELSEWHERE_DEVICE, ELSEWHERE_INODE = "D0x2f", "i47"      # how lsof -FDi tells a folder that is not on this disk


def stubs(table: Path) -> str:
    """The bash functions; `table` is the directory the two files and the call records live in."""
    return f'''
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
        dir=$(awk -v p="$pid" '$1 == p {{ print $3 }}' "{table}/processes")
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
'''


class ProcessTable:
    """What runs on the Mac: Keep processes (`python floxy_ai_server.py`, whoever owns them and
    whichever checkout they run from) and the ports that are listened on."""

    def __init__(self, directory: Path):
        self.dir = directory
        (directory / 'listeners').write_text('')
        (directory / 'processes').write_text('')

    def listener(self, port: int, pid: int, user: int | None = None):
        """`pid` listens on `port`; `user` defaults to the one running the script."""
        with (self.dir / 'listeners').open('a') as f:
            f.write(f"{port} {pid} {os.getuid() if user is None else user}\n")

    def process(self, pid: int, user: int | None = None, checkout: str | None = None):
        """A Keep process; by default the script's own user's, running from the script's checkout."""
        with (self.dir / 'processes').open('a') as f:
            f.write(f"{pid} {os.getuid() if user is None else user} {checkout or '-'}\n")

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
