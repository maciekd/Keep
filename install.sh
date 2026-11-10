#!/bin/bash
#
# Keep installer. From the readme:
#
#   curl -fsSL https://raw.githubusercontent.com/maciekd/Keep/release/install.sh | bash
#
# Checks the Mac, finds or installs Python 3.12 without touching anything else on the
# machine, clones the repository, builds the venv, asks for the Anthropic API key (and the
# workspace it acts in, when the key is not tied to one), writes
# config.yaml and hands the terminal to start.sh. Every step is idempotent: re-run the same
# command to resume after a failure or to upgrade an existing install.
#
# Environment overrides (all optional):
#   KEEP_DIR           where to install            (default: ~/Keep); the variable must
#                      reach bash, not curl:  curl ... | KEEP_DIR=~/somewhere bash
#   ANTHROPIC_API_KEY  skips the API key prompt
#   ANTHROPIC_WORKSPACE_ID  the workspace (wrkspc_...) a key not tied to one workspace acts in;
#                      skips that prompt. A key created for one workspace needs none.
#   KEEP_NO_START=1    stop after installing instead of launching start.sh
#   KEEP_REPO          repository URL              (default: https://github.com/maciekd/Keep.git)
#   KEEP_BRANCH        branch to install           (default: release — what release.sh publishes;
#                      an existing checkout keeps the branch it is on unless this is set)
#   KEEP_NO_TELEMETRY=1  send no install events to PostHog and write telemetry.enabled: false
#                      into the new config.yaml (see the telemetry section below)
#   KEEP_ROLE=imessage_satellite  install an iMessage satellite instead of a Keep server: the same
#                      checkout and venv, but no assistant runs there — floxy_ai_server.py serves
#                      the Messages account of this macOS user to a Keep server elsewhere (another
#                      user of this Mac, or another Mac). The value is the one config.yaml's `role`
#                      key takes. No API key is asked for, config.yaml holds the role, a generated
#                      token and little else, and the summary prints what the server needs to pair.
#                      An existing install keeps the role its config.yaml names: a re-run upgrades
#                      it with or without the variable, and a variable that names the other role
#                      is refused rather than turning a server into a satellite.
#   KEEP_LAN_ACCESS=1|0  satellite only: whether it answers the network (1: the Keep server is on
#                      another Mac) or this Mac alone (0: the server is another user of this Mac).
#                      Skips that prompt; without a terminal to ask on, 0 is assumed.
#
# Written for the bash 3.2 that ships with macOS: no associative arrays, mapfile or ${var,,}.
# Everything sits inside main() so a truncated download runs nothing at all.

set -eEuo pipefail

main() {
  REPO_URL="${KEEP_REPO:-https://github.com/maciekd/Keep.git}"
  BRANCH="${KEEP_BRANCH:-release}"
  KEEP_DIR="${KEEP_DIR:-$HOME/Keep}"
  MIN_MACOS=14         # mlx needs Sonoma or newer
  MIN_FREE_GB=10       # the voice stack (torch, spacy, mlx) is close to 3 GB installed
  PIP_EXPECTED_PACKAGES=380   # packages a full install collects (pip freeze | wc -l); scales the progress bar
  TOTAL_STEPS=8
  ROLE=server          # server | imessage_satellite; settled by resolve_role
  STEP=0
  STEP_NAME="startup"
  SPINNER_PID=""
  SPINNER_STATUS=""    # a command the spinner evaluates each tick for extra text (see pip_status)
  CHILD_PID=""
  PY=""
  PY_SOURCE=""         # found | brew | uv: where the Python came from, for the install report
  TELEMETRY_ID=""
  API_KEY=""
  WORKSPACE_ID=""
  VERIFY_CODE=""
  ANTHROPIC_SAID=""

  init_ui
  trap on_error ERR
  trap on_exit EXIT
  trap on_interrupt INT TERM
  open_log
  resolve_role

  printf '\n  %sKeep%s - personal AI done right\n' "$C_BOLD" "$C_RESET"
  if satellite; then
    TOTAL_STEPS=7      # no API key step: no LLM runs on a satellite
    dim "Installing an iMessage satellite into $KEEP_DIR  (override with KEEP_DIR=...)"
    dim "No assistant runs here: it lends this Mac user's Messages to your Keep server."
  else
    dim "Installing into $KEEP_DIR  (override with KEEP_DIR=...)"
  fi
  dim "Install log: $LOG"
  init_telemetry
  telemetry_started

  step "Checking this Mac"
  check_mac

  step "Xcode Command Line Tools"
  ensure_clt

  step "Python 3.12"
  ensure_python

  step "Keep source"
  ensure_repo

  step "Virtual environment"
  ensure_venv

  step "Python packages"
  install_requirements

  if satellite; then
    step "Configuration"
    if [ -f "$KEEP_DIR/config.yaml" ]; then
      ok "config.yaml kept as it is"
    else
      write_satellite_config
    fi
  elif [ -f "$KEEP_DIR/config.yaml" ]; then
    step "Anthropic API key"
    ok "config.yaml already exists; its key is kept"
    step "Configuration"
    ok "config.yaml kept as it is"
  else
    step "Anthropic API key"
    obtain_api_key
    step "Configuration"
    write_config
  fi

  summary
  # shellcheck disable=SC2046
  telemetry_event install_finished elapsed_s="$SECONDS" python="$PY_SOURCE" \
    commit="$(git -C "$KEEP_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)" $(telemetry_role)
  launch
}

# ------------------------------------------------------------- the role -----

satellite() { [ "${ROLE:-server}" = imessage_satellite ]; }

# The role comes from KEEP_ROLE on a fresh install and from config.yaml on an existing one — the
# file is what floxy_ai_server.py and start.sh read (common/Role.py, scripts/role.sh), so it is
# what the install is. Settled before anything is checked or installed: a value that is no role
# stops here, and so does a variable asking for the other role than the folder's install has.
resolve_role() {
  local wanted=${KEEP_ROLE:-} existing unreadable
  case "$wanted" in
    ""|server|imessage_satellite) ;;
    *) fail "KEEP_ROLE=$wanted is not a role." \
         "Leave it out for a Keep server, or set KEEP_ROLE=imessage_satellite for an iMessage satellite." ;;
  esac
  if [ ! -f "$KEEP_DIR/config.yaml" ]; then
    ROLE=${wanted:-server}
    log "role: $ROLE (${wanted:+KEEP_ROLE, }a fresh install)"
    return
  fi
  unreadable=$(config_trouble role "")
  [ -z "$unreadable" ] || fail "$KEEP_DIR/config.yaml: $unreadable." \
    "Whether this install is a Keep server or an iMessage satellite cannot be told, and is not guessed." \
    "Write the keys as lines of their own from the first column (role: imessage_satellite), then re-run the installer."
  # An empty value and yaml's null are no role (a server); "null" in quotes is a word, as to Python.
  existing=$(config_raw role "")
  case "$existing" in
    ""|null|Null|NULL|"~"|'""'|"''") existing=server ;;
    *) existing=$(printf '%s\n' "$existing" | awk '{ gsub(/["\047]/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); print tolower($0) }') ;;
  esac
  case "$existing" in
    server|imessage_satellite) ;;
    *) fail "$KEEP_DIR/config.yaml names role: $existing, which is not a role." \
         "Remove the key for a Keep server, or set it to imessage_satellite, and re-run the installer." ;;
  esac
  if [ "$existing" = imessage_satellite ]; then
    unreadable=$(config_trouble imessage_satellite port)
    [ -z "$unreadable" ] || fail "$KEEP_DIR/config.yaml: $unreadable." \
      "start.sh would not know which port the satellite listens on. Write it as the installer does:" \
      "'port: 8898' on a line indented under 'imessage_satellite:', then re-run the installer."
  fi
  if [ -n "$wanted" ] && [ "$wanted" != "$existing" ]; then
    if [ "$wanted" = imessage_satellite ]; then
      fail "$KEEP_DIR holds a Keep server's install; a satellite is an install of its own." \
        "It belongs where the Messages account it serves is signed in: under another macOS user of this Mac," \
        "or on another Mac. Run the same command there, in that user's Terminal." \
        "A second folder under this user (KEEP_DIR=...) is not the way: the watchdog that relaunches a dead" \
        "process is one per macOS user, and the server and the satellite would each point it at themselves."
    fi
    fail "$KEEP_DIR holds an iMessage satellite's install; a Keep server is an install of its own." \
      "Install it under the macOS user it is to run as, or on another Mac. In another folder of this user" \
      "(curl ... | KEEP_DIR=~/somewhere bash) it shares this user's one watchdog with the satellite:" \
      "each start points it at its own install."
  fi
  ROLE=$existing
  log "role: $ROLE (from the existing config.yaml)"
}

# ---------------------------------------------------------------- UI --------

init_ui() {
  if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    IS_TTY=1
    C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'; C_RESET=$'\033[0m'
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_CYAN=$'\033[36m'
  else
    IS_TTY=""
    C_BOLD=""; C_DIM=""; C_RESET=""; C_GREEN=""; C_RED=""; C_YELLOW=""; C_CYAN=""
  fi
  # Prompts go through the terminal directly: stdin may be a pipe when the script is
  # fed to bash by curl, and that pipe must never be what `read` waits on. Whether there is a
  # terminal is told by opening it: /dev/tty exists, readable and writable, for a process that
  # has none (ssh host 'curl ... | bash', launchd), and only the open fails there.
  if (exec </dev/tty >/dev/tty) 2>/dev/null; then TTY=/dev/tty; else TTY=""; fi
}

# The log lives in its final place, $KEEP_DIR/log/install.log, from the first line. Only a
# directory the clone will refuse anyway (non-empty, not a Keep checkout) gets a temp log
# instead, so nothing is left behind in it.
open_log() {
  if [ ! -e "$KEEP_DIR" ] || [ -d "$KEEP_DIR/.git" ] || holds_only_our_log "$KEEP_DIR"; then
    mkdir -p "$KEEP_DIR/log"
    LOG="$KEEP_DIR/log/install.log"
  else
    LOG=$(/usr/bin/mktemp -t keep-install)
  fi
  log ""
  log "Keep installer started $(date '+%Y-%m-%d %H:%M:%S') by $(id -un)"
}

# A directory that holds nothing but log/install.log is one an earlier run created and then
# failed in before the clone (no network, interrupted Command Line Tools wait, ...). It
# counts as empty: the re-run appends to that log and clones over it.
holds_only_our_log() {
  [ -z "$(ls -A "$1" 2>/dev/null | grep -v '^log$' || true)" ] &&
    [ -z "$(ls -A "$1/log" 2>/dev/null | grep -v '^install\.log$' || true)" ]
}

step() {
  STEP=$((STEP + 1))
  STEP_NAME="$1"
  printf '\n  %s[%d/%d]%s %s%s%s\n' "$C_CYAN" "$STEP" "$TOTAL_STEPS" "$C_RESET" "$C_BOLD" "$1" "$C_RESET"
  log ""
  log "=== [$STEP/$TOTAL_STEPS] $1"
}

ok()   { printf '  %s✔%s %s\n' "$C_GREEN" "$C_RESET" "$1"; log "ok: $1"; }
warn() { printf '  %s!%s %s\n' "$C_YELLOW" "$C_RESET" "$1"; log "warn: $1"; }
dim()  { printf '  %s%s%s\n' "$C_DIM" "$1" "$C_RESET"; log "$1"; }
note() { printf '    %s\n' "$1"; }
log()  { if [ -n "${LOG:-}" ]; then printf '%s\n' "$*" >>"$LOG"; fi; }

fail() {
  local reason=$1
  printf '\n  %s✖ %s%s\n' "$C_RED" "$1" "$C_RESET"
  shift
  local line
  for line in "$@"; do note "$line"; done
  log "FAILED in step: $STEP_NAME"
  telemetry_event install_failed step="$STEP_NAME" reason="$reason" elapsed_s="$SECONDS"
  if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then printf '\n  %sLog: %s%s\n' "$C_DIM" "$LOG" "$C_RESET"; fi
  printf '  %sRe-run the same command to resume; finished steps are skipped.%s\n\n' "$C_DIM" "$C_RESET"
  exit 1
}

on_error() {
  stop_spinner
  fail "Unexpected error in step \"$STEP_NAME\" (install.sh line ${BASH_LINENO[0]:-?})."
}

on_exit() { stop_spinner; }

on_interrupt() {
  stop_spinner
  if [ -n "$CHILD_PID" ]; then kill "$CHILD_PID" 2>/dev/null || true; fi
  printf '\n  %sInterrupted.%s Re-run the same command to resume.\n\n' "$C_YELLOW" "$C_RESET"
  telemetry_event install_failed step="$STEP_NAME" reason=interrupted elapsed_s="$SECONDS"
  exit 130
}

fmt_elapsed() {
  local s=$1
  if [ "$s" -ge 60 ]; then printf '%dm %02ds' $((s / 60)) $((s % 60)); else printf '%ds' "$s"; fi
}

start_spinner() {
  if [ -z "$IS_TTY" ]; then printf '  … %s\n' "$1"; return; fi
  local full_label=$1 start=$SECONDS
  printf '\033[?25l'
  (
    frames='⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏'
    i=0
    status=""
    cols=80
    while :; do
      # shellcheck disable=SC2086
      set -- $frames
      shift $((i % 10))
      label=$full_label
      if [ -n "$SPINNER_STATUS" ] && [ $((i % 5)) -eq 0 ]; then
        status=" $(eval "$SPINNER_STATUS" 2>/dev/null || true)"
      fi
      if [ $((i % 10)) -eq 0 ]; then cols=$(tput cols 2>/dev/null || echo 80); fi
      elapsed=$(fmt_elapsed $((SECONDS - start)))
      # The line is redrawn in place with \r, which only works while it fits on one row:
      # once the terminal wraps it, every tick leaves the previous row behind. Clip to the
      # width, status text (the long wheel names) first, the label after it.
      room=$((cols - 1 - 2 - 2 - ${#label} - 1 - ${#elapsed} - 2))
      if [ "$room" -lt "${#status}" ]; then
        if [ "$room" -gt 3 ]; then status="${status:0:$((room - 1))}…"; else status=""; fi
        room=$((cols - 1 - 2 - 2 - 1 - ${#elapsed} - 2))
        if [ "$room" -lt "${#label}" ] && [ "$room" -gt 3 ]; then label="${label:0:$((room - 1))}…"; fi
      fi
      printf '\r  %s%s%s %s%s %s(%s)%s\033[K' "$C_CYAN" "$1" "$C_RESET" "$label" "$status" "$C_DIM" "$elapsed" "$C_RESET"
      sleep 0.1
      i=$((i + 1))
    done
  ) &
  SPINNER_PID=$!
}

stop_spinner() {
  if [ -n "${SPINNER_PID:-}" ]; then
    kill "$SPINNER_PID" 2>/dev/null || true
    wait "$SPINNER_PID" 2>/dev/null || true
    SPINNER_PID=""
    printf '\r\033[K\033[?25h'
  fi
}

# run <label> <command...>: the command's output goes to the log, the screen shows a
# spinner with the elapsed time; on failure the log's tail is shown and the install stops.
run() {
  local label=$1; shift
  local start=$SECONDS status=0
  log ""
  log "--- $label: $*"
  start_spinner "$label"
  "$@" >>"$LOG" 2>&1 &
  CHILD_PID=$!
  wait "$CHILD_PID" || status=$?
  CHILD_PID=""
  stop_spinner
  if [ "$status" -eq 0 ]; then
    ok "$label $C_DIM($(fmt_elapsed $((SECONDS - start))))$C_RESET"
    return 0
  fi
  printf '\n  %s✖ %s failed (exit %d). Last lines of the log:%s\n\n' "$C_RED" "$label" "$status" "$C_RESET"
  tail -n 25 "$LOG" | sed 's/^/    │ /'
  fail "$label did not complete."
}

# ask <prompt> <variable> [secret]: reads from the terminal, never from stdin.
ask() {
  local prompt=$1 var=$2 secret=${3:-}
  [ -n "$TTY" ] || fail "No terminal to ask on." \
    "Run the installer from Terminal, or set the value in the environment (see the top of install.sh)."
  printf '  %s%s%s ' "$C_BOLD" "$prompt" "$C_RESET" >"$TTY"
  if [ -n "$secret" ]; then
    IFS= read -r -s "$var" <"$TTY"
    printf '\n' >"$TTY"
  else
    IFS= read -r "$var" <"$TTY"
  fi
}

# ---------------------------------------------------------- 1. the Mac ------

check_mac() {
  [ "$(id -u)" -ne 0 ] || fail "Please run the installer as your normal user, not as root." \
    "Everything is installed under your home directory; nothing needs sudo."
  [ "$(uname -s)" = "Darwin" ] || fail "Keep runs on macOS only (this is $(uname -s))."

  # hw.optional.arm64 is true even inside a Rosetta shell, where uname -m says x86_64.
  local arm chip
  arm=$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)
  chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown CPU")
  [ "$arm" = "1" ] || fail "Keep needs an Apple Silicon Mac (M1 or newer); this one reports: $chip." \
    "The on-device speech models run on the GPU via MLX."
  ok "Apple Silicon: $chip"
  if [ "$(uname -m)" != "arm64" ]; then
    warn "This shell runs under Rosetta (x86_64). Keep still installs natively, but launch it from a native Terminal."
  fi

  local macos major
  macos=$(sw_vers -productVersion)
  major=${macos%%.*}
  [ "$major" -ge "$MIN_MACOS" ] 2>/dev/null || fail "macOS $macos is too old; Keep needs macOS $MIN_MACOS (Sonoma) or newer."
  ok "macOS $macos"

  # The console is a full-screen TUI; Terminal.app renders it noticeably worse than
  # Ghostty or iTerm2. TERM_PROGRAM is what every macOS terminal sets to identify itself.
  # (A satellite has no console, so any terminal will do.)
  if ! satellite; then
    case "${TERM_PROGRAM:-}" in
      ghostty)   ok "Terminal: Ghostty" ;;
      iTerm.app) ok "Terminal: iTerm2" ;;
      *) warn "Ghostty terminal is recommended for the best experience - download here: https://ghostty.org/" ;;
    esac
  fi

  # System binaries by absolute path: with Homebrew's GNU coreutils first on PATH, df -g,
  # mktemp -t <prefix> and stat -f are not the BSD commands this script was written for.
  local free
  free=$(/bin/df -Pk "$HOME" | tail -n 1 | awk '{ print int($4 / 1048576) }')
  [ "${free:-0}" -ge "$MIN_FREE_GB" ] 2>/dev/null || fail "Only ${free:-?} GB free on the disk holding $HOME; Keep needs about $MIN_FREE_GB GB." \
    "The speech and speaker-identification models are the bulk of it."
  ok "Disk: ${free} GB free"

  start_spinner "Testing network"
  reachable https://github.com || fail "Cannot reach github.com." "Check the network connection and try again."
  if satellite; then   # it talks to GitHub (the code) and to its Keep server, never to Anthropic
    stop_spinner
    ok "Network: github.com reachable"
    return
  fi
  reachable https://api.anthropic.com || fail "Cannot reach api.anthropic.com." "Check the network connection and try again."
  stop_spinner
  ok "Network: github.com and api.anthropic.com reachable"
}

reachable() {
  local code
  code=$(curl -sS -o /dev/null --max-time 15 -w '%{http_code}' "$1" 2>>"$LOG" || true)
  [ -n "$code" ] && [ "$code" != "000" ]
}

# --------------------------------------------------------- 2. Xcode CLT -----

ensure_clt() {
  # git on a fresh Mac is a stub that pops the Command Line Tools installer, so this is
  # settled before git is ever called.
  if xcode-select -p >/dev/null 2>&1; then
    ok "Installed at $(xcode-select -p)"
    return
  fi
  warn "Not installed yet. macOS opens its installer window now: click Install and let it finish."
  xcode-select --install >>"$LOG" 2>&1 || true
  start_spinner "Waiting for the Command Line Tools installer"
  local waited=0
  until xcode-select -p >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge 1800 ]; then
      stop_spinner
      fail "The Command Line Tools did not appear within 30 minutes." \
        "Install them with: xcode-select --install   and re-run this installer."
    fi
  done
  stop_spinner
  ok "Installed at $(xcode-select -p)"
}

# ------------------------------------------------------------ 3. Python -----

# A candidate must be CPython 3.12 built for arm64 (a Rosetta 3.12 would pull x86 wheels
# and MLX would never install) with the ssl, sqlite3 and venv modules present.
python_ok() {
  [ -n "$1" ] && [ -x "$1" ] || return 1
  "$1" - <<'EOF' >>"$LOG" 2>&1
import platform, sys
assert sys.version_info[:2] == (3, 12), sys.version
assert platform.machine() == "arm64", platform.machine()
import ssl, sqlite3, venv  # noqa: F401
EOF
}

python_version() { "$1" -c 'import platform; print(platform.python_version())' 2>/dev/null; }

find_python() {
  shopt -s nullglob
  local c
  for c in \
      "$(command -v python3.12 2>/dev/null || true)" \
      /opt/homebrew/opt/python@3.12/bin/python3.12 \
      /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
      "$HOME"/.local/share/uv/python/cpython-3.12.*-macos-aarch64-none/bin/python3.12 \
      "$HOME"/.pyenv/versions/3.12.*/bin/python3.12; do
    [ -n "$c" ] || continue
    if python_ok "$c"; then PY="$c"; shopt -u nullglob; return 0; fi
    log "rejected python candidate: $c"
  done
  shopt -u nullglob
  return 1
}

ensure_python() {
  if find_python; then
    PY_SOURCE=found
    ok "Python $(python_version "$PY") at $PY"
    return
  fi
  dim "No usable Python 3.12 found (need an arm64 CPython 3.12 with ssl, sqlite3 and venv)."

  if command -v brew >/dev/null 2>&1; then
    # Homebrew's python@3.12 is versioned: it adds python3.12 next to whatever python3 the
    # machine already has and changes nothing else.
    note "Installing python@3.12 with Homebrew (adds the versioned python3.12 only)."
    run "brew install python@3.12" brew install python@3.12
    PY="$(brew --prefix python@3.12 2>/dev/null || true)/bin/python3.12"
    PY_SOURCE=brew
  else
    # No Homebrew: uv keeps its interpreters under ~/.local/share/uv and edits no shell
    # profile, so this is the least intrusive way to get a 3.12 without an admin password.
    local uv
    uv=$(command -v uv 2>/dev/null || true)
    if [ -z "$uv" ] && [ -x "$HOME/.local/bin/uv" ]; then uv="$HOME/.local/bin/uv"; fi
    if [ -z "$uv" ]; then
      note "Installing uv into ~/.local/bin (a standalone Python manager; your shell profile is not modified)."
      run "Install uv" env UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh -c \
        'curl -LsSf https://astral.sh/uv/install.sh | sh'
      uv="$HOME/.local/bin/uv"
      [ -x "$uv" ] || fail "uv did not install to $uv."
    fi
    run "uv python install 3.12" "$uv" python install 3.12
    PY=$("$uv" python find 3.12 2>>"$LOG" || true)
    PY_SOURCE=uv
  fi

  python_ok "$PY" || fail "Installed Python at '$PY' but it does not pass the checks." \
    "It must be CPython 3.12 for arm64 with ssl, sqlite3 and venv. See the log for details."
  ok "Python $(python_version "$PY") at $PY"
}

# --------------------------------------------------------- 4. the source ----

ensure_repo() {
  export GIT_TERMINAL_PROMPT=0
  if [ -d "$KEEP_DIR/.git" ]; then
    local origin
    origin=$(git -C "$KEEP_DIR" remote get-url origin 2>/dev/null || true)
    if [ "${origin%.git}" != "${REPO_URL%.git}" ]; then
      fail "$KEEP_DIR is a git checkout of '$origin', not of $REPO_URL." \
        "Choose another directory: curl ... | KEEP_DIR=~/somewhere bash"
    fi
    dim "Existing checkout found, updating it."
    if [ -n "$(git -C "$KEEP_DIR" status --porcelain 2>/dev/null)" ]; then
      warn "Local changes in $KEEP_DIR are kept; only a fast-forward update is applied."
    fi
    # An install follows the branch it was made from (release, or master on a development
    # checkout); only an explicit KEEP_BRANCH moves it.
    local current
    current=$(git -C "$KEEP_DIR" symbolic-ref --short -q HEAD 2>/dev/null || true)
    if [ -z "${KEEP_BRANCH:-}" ] && [ -n "$current" ]; then
      BRANCH="$current"
    elif [ "$current" != "$BRANCH" ]; then
      run "git fetch $BRANCH" git -C "$KEEP_DIR" fetch origin "$BRANCH"
      run "git checkout $BRANCH" git -C "$KEEP_DIR" checkout "$BRANCH"
    fi
    run "git pull --ff-only ($BRANCH)" git -C "$KEEP_DIR" pull --ff-only origin "$BRANCH"
  elif [ -e "$KEEP_DIR" ] && ! holds_only_our_log "$KEEP_DIR"; then
    fail "$KEEP_DIR exists and is not empty, and it is not a Keep checkout." \
      "Move it away, or install elsewhere: curl ... | KEEP_DIR=~/somewhere bash"
  else
    # git clone insists on an empty directory and the log already lives in it, so the log
    # is parked outside for the clone and moved back into the fresh checkout afterwards.
    # Anything else in log/ would make the clone fail, which holds_only_our_log rules out.
    # A full clone, not a shallow one: the assistant's self-improvement agent works with
    # branches and history in this very checkout.
    local final="$KEEP_DIR/log/install.log" parked
    parked=$(/usr/bin/mktemp -t keep-install)
    mv "$LOG" "$parked"
    rmdir "$KEEP_DIR/log"
    LOG="$parked"
    run "git clone ($BRANCH)" git clone --branch "$BRANCH" "$REPO_URL" "$KEEP_DIR"
    mkdir -p "$KEEP_DIR/log"
    mv "$parked" "$final"
    LOG="$final"
  fi
  local rev
  rev=$(git -C "$KEEP_DIR" log -1 --format='%h  %s' 2>/dev/null || echo "?")
  ok "At $rev"
}

# ------------------------------------------------------------- 5. venv -----

ensure_venv() {
  local venv="$KEEP_DIR/.venv"
  if python_ok "$venv/bin/python"; then
    ok "Existing venv is a healthy Python $(python_version "$venv/bin/python")"
    return
  fi
  if [ -e "$venv" ]; then
    warn "Existing .venv is not an arm64 Python 3.12; recreating it."
    rm -rf "$venv"
  fi
  run "python3.12 -m venv .venv" "$PY" -m venv "$venv"
  python_ok "$venv/bin/python" || fail "The new venv's Python does not pass the checks."
}

# ---------------------------------------------------------- 6. packages -----

install_requirements() {
  local py="$KEEP_DIR/.venv/bin/python"
  export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off
  run "Upgrade pip" "$py" -m pip install --upgrade pip
  # requirements-voice.txt includes requirements.txt. Both groups go in even though voice
  # is off by default: the server verifies both at every start so voice can be enabled
  # from the chat later without a reinstall.
  note "Downloading Python packages (including speech-identification and language models); takes 1-10 minutes."
  local offset
  offset=$(($(wc -c <"$LOG") + 1))
  SPINNER_STATUS="pip_status $offset $PIP_EXPECTED_PACKAGES"
  run "pip install" "$py" -m pip install -r "$KEEP_DIR/requirements-voice.txt"
  SPINNER_STATUS=""
}

# pip_status <log offset> <expected package count>: progress read off pip's own output.
# Pip works in passes: it resolves first, printing "Collecting <pkg>" per package while it
# fetches metadata, then downloads every wheel ("Downloading <file> (<size>)", unindented),
# builds the few source-only packages and finally unpacks. The bar spans the first two
# passes, which is where the time goes on a fresh Mac, scaled by the package count of a
# full install since pip only reveals the total at the end; the text says what it is doing.
pip_status() {
  local out n_collect n_down n_total total text filled bar
  out=$(tail -c +"$1" "$LOG" 2>/dev/null || true)
  case "$out" in
    *"Installing collected packages"*)
      n_total=$(printf '%s\n' "$out" | grep '^Installing collected packages' | tr ',' '\n' | grep -c . || true)
      printf 'unpacking %d packages' "$n_total"; return ;;
    *"Building wheels for collected packages"*)
      printf 'building source packages'; return ;;
  esac
  n_collect=$(printf '%s\n' "$out" | grep -c '^Collecting ' || true)
  n_down=$(printf '%s\n' "$out" | grep -cE '^(Downloading|Using cached) ' || true)
  if [ "$n_down" -gt 0 ]; then
    text=$(printf '%s\n' "$out" | grep -E '^(Downloading|Using cached) ' | tail -n 1 | awk '{
      sub(/^(Downloading|Using cached) /, ""); split($1, a, "-")
      size = ""; if (match($0, /\([^()]*\)$/)) size = " " substr($0, RSTART, RLENGTH)
      print "downloading " a[1] " " a[2] size }')
  else
    text="resolving $n_collect packages"
  fi
  total=$(($2 * 2))
  n_total=$((n_collect + n_down))
  [ "$n_total" -le "$total" ] || n_total=$total
  filled=$((n_total * 24 / total))
  bar=$(printf '%*s' "$filled" '' | sed 's/ /█/g')$(printf '%*s' $((24 - filled)) '')
  printf '[%s] %s' "$bar" "$text"
}

# ------------------------------------------------------------ 7. API key ----

KEYS_URL="https://platform.claude.com/settings/keys"
WORKSPACES_URL="https://platform.claude.com/settings/workspaces"
BILLING_URL="https://platform.claude.com/settings/billing"

verify_api_key() {
  # Leaves in VERIFY_CODE what Anthropic made of the key: an HTTP status, or "workspace" when
  # the key is not tied to one workspace and Anthropic wants to be told which to use, or
  # "bad-workspace" when the id given for it is not one the key can act in, or "credit" when
  # the key is good but its organization has no credit; Anthropic's own words in ANTHROPIC_SAID.
  # A real one-token message on Haiku, a fraction of a cent: the free count_tokens endpoint
  # is not documented to check the balance, and an unfunded key that passes here leaves the
  # first chat silent.
  local body="$LOG.verify" code
  local -a workspace_header=()
  [ -z "$WORKSPACE_ID" ] || workspace_header=(-H "anthropic-workspace-id: $WORKSPACE_ID")
  code=$(curl -sS -o "$body" --max-time 20 -w '%{http_code}' \
    -H "x-api-key: $API_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" \
    ${workspace_header[@]+"${workspace_header[@]}"} \
    -d '{"model":"claude-haiku-5-5","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
    https://api.anthropic.com/v1/messages 2>>"$LOG" || true)
  ANTHROPIC_SAID=$(sed -n 's/.*"message":"\([^"]*\)".*/\1/p' "$body" 2>/dev/null | head -n 1 || true)
  log "key check: HTTP ${code:-none} ${ANTHROPIC_SAID:+- $ANTHROPIC_SAID}"
  case "$code" in
    400) case "$ANTHROPIC_SAID" in
           *anthropic-workspace-id*required*) code=workspace ;;
           *[Cc]redit\ balance*) code=credit ;;
           *workspace*) [ -z "$WORKSPACE_ID" ] || code=bad-workspace ;;
         esac ;;
    404) case "$ANTHROPIC_SAID" in *[Ww]orkspace*) [ -z "$WORKSPACE_ID" ] || code=bad-workspace ;; esac ;;
  esac
  VERIFY_CODE=$code
}

explain_workspace() {
  warn "This key's Scope is Organization, so Anthropic needs to be told which workspace to use."
  note "The easy fix: create a new key at ${C_BOLD}$KEYS_URL${C_RESET} with Scope set to"
  note "Default workspace (and Expires to Never). Then paste the new key here."
  note "Or keep this key and paste a workspace id instead: it starts with wrkspc_ and is in the"
  note "ID column at ${C_BOLD}$WORKSPACES_URL${C_RESET}."
  printf '\n'
}

explain_credit() {
  note "Keep uses your organization's API credit. On a Claude Max, Ultra or Team plan you get"
  note "\$100–500 of it a month for free: claim it at claude.ai → Settings → Billing → API credits"
  note "(link the Console organization you create this key in). Otherwise add credits at"
  note "${C_BOLD}$BILLING_URL${C_RESET}."
}

# The key is good but its organization has no credit: wait while the user claims or buys some,
# then check again. Returns 0 to check again, 1 to go on without credit.
wait_for_credit() {
  warn "The key works, but its organization has no API credit yet."
  note "On a Claude Max, Ultra or Team plan, claim the plan's monthly credit at claude.ai → Settings →"
  note "Billing → API credits and link this key's Console organization (a new subscription can claim"
  note "it after 7 days on the plan). Otherwise add credits at ${C_BOLD}$BILLING_URL${C_RESET}."
  printf '\n'
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    warn "Keeping ANTHROPIC_API_KEY from the environment. Keep cannot answer until the organization has credit."
    return 1
  fi
  local answer=""
  ask "Press Enter to check again, or type skip to finish without credit:" answer
  case "$(printf '%s' "$answer" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')" in
    skip) warn "Keeping the key. Keep cannot answer until the organization has credit."; return 1 ;;
  esac
  dim "Checking again..."
  return 0
}

obtain_api_key() {
  API_KEY="${ANTHROPIC_API_KEY:-}"
  WORKSPACE_ID="${ANTHROPIC_WORKSPACE_ID:-}"
  if [ -n "$API_KEY" ]; then
    dim "Using ANTHROPIC_API_KEY from the environment."
  else
    note "Keep runs on Claude. An API key is the only way to use it without breaking Anthropic's terms of service."
    note "Create one at ${C_BOLD}$KEYS_URL${C_RESET} (it starts with sk-ant-). In the dialog:"
    note "  Expires: Never"
    note "  Scope:   Default workspace"
    note "The key is stored only in $KEEP_DIR/config.yaml. Typing is hidden."
    printf '\n'
    explain_credit
    printf '\n'
  fi
  local attempt=0 workspace_attempt=0
  while :; do
    if [ -z "$API_KEY" ]; then
      attempt=$((attempt + 1))
      [ "$attempt" -le 3 ] || fail "No valid API key after 3 attempts." \
        "Re-run the installer when you have one, or pass it as ANTHROPIC_API_KEY=... in the environment."
      ask "Anthropic API key:" API_KEY secret
      API_KEY=$(printf '%s' "$API_KEY" | tr -d '[:space:]')
    fi
    case "$API_KEY" in
      sk-ant-*) ;;
      "")  warn "Nothing entered."; continue ;;
      *)   warn "That does not look like an Anthropic key (they start with sk-ant-)."; API_KEY=""; continue ;;
    esac
    verify_api_key
    case "$VERIFY_CODE" in
      200) ok "Key verified with the Anthropic API${WORKSPACE_ID:+ (workspace $WORKSPACE_ID)}"; return ;;
      credit) if wait_for_credit; then continue; fi; return ;;
      401|403) warn "Anthropic rejected that key (HTTP $VERIFY_CODE). Check for a typo or create a new one."; API_KEY="" ;;
      workspace|bad-workspace)
        if [ "$VERIFY_CODE" = workspace ]; then explain_workspace
        else warn "Anthropic does not know workspace $WORKSPACE_ID, or this key cannot use it. Check the ID column at $WORKSPACES_URL."; fi
        workspace_attempt=$((workspace_attempt + 1))
        [ "$workspace_attempt" -le 3 ] || fail "No usable workspace id after 3 attempts." \
          "Create a key with Scope set to Default workspace at $KEYS_URL and re-run the installer, or pass ANTHROPIC_WORKSPACE_ID=wrkspc_... in the environment."
        ask "Workspace id (or press Enter to paste a new key instead):" WORKSPACE_ID
        WORKSPACE_ID=$(printf '%s' "$WORKSPACE_ID" | tr -d '[:space:]')
        case "$WORKSPACE_ID" in
          wrkspc_*) continue ;;
          "") [ -z "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY from the environment is not tied to a workspace." \
                "Pass ANTHROPIC_WORKSPACE_ID=wrkspc_... as well, or create a key with Scope set to Default workspace at $KEYS_URL."
              API_KEY=""; continue ;;
          *) warn "That does not look like a workspace id (they start with wrkspc_)."; WORKSPACE_ID=""; continue ;;
        esac ;;
      *) warn "Could not verify the key right now (HTTP ${VERIFY_CODE:-none}${ANTHROPIC_SAID:+: $ANTHROPIC_SAID}); keeping it, the server will tell you if it is wrong."; return ;;
    esac
    [ -z "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY from the environment was rejected."
  done
}

# -------------------------------------------------------------- 8. config ---

random_secret() {
  # head reads a fixed amount, so nothing upstream dies of SIGPIPE under pipefail.
  head -c 96 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | cut -c1-32
}

write_config() {
  local vault_key
  vault_key=$(random_secret)
  [ "${#vault_key}" -eq 32 ] || fail "Could not generate a vault passphrase."
  (
    umask 077
    cat >"$KEEP_DIR/config.yaml" <<EOF
# Keep configuration, written by install.sh on $(date '+%Y-%m-%d'). config.example.yaml documents
# every setting; the assistant sets up the optional integrations itself once it runs.
api_keys:
  anthropic: $API_KEY
${WORKSPACE_ID:+  anthropic_workspace: $WORKSPACE_ID   # the key is not tied to one workspace, so every request names this one}
$(if [ "$BRANCH" = master ]; then echo "developer_release: true   # installed from master, the development line; restart_server keeps following it"
   else echo "developer_release: false  # restart_server follows the release branch; true: master, the development line"; fi)
vault:
  # Encrypts vault.bin, the secrets the assistant stores for you. Lose it and the vault is
  # lost; change it and the existing vault.bin can no longer be opened.
  key: $vault_key
telemetry:
  # Beta usage telemetry; config.example.yaml says what goes out. enabled: false turns it off.
$(if [ -n "${KEEP_NO_TELEMETRY:-}" ]; then echo "  enabled: false"
   else echo "  enabled: true"; echo "  id: $TELEMETRY_ID"; fi)
EOF
  )
  ok "Wrote config.yaml (readable by you only)"
  note "A random vault passphrase was generated for you. It lives in config.yaml: back that file up."
}

# The config of an iMessage satellite: the role, the service's section and what keeps the
# process running and up to date. No API key and no vault — nothing on a satellite uses either,
# and floxy_ai_server.py asks for neither under this role. The file has to be there before the
# first start: without one ConfigManager writes the server's skeleton.
SATELLITE_PORT=8898   # imessage_satellite.port: not the server's 8888, since a second user shares the Mac's ports
LAN_ACCESS=false      # imessage_satellite.lan_access, settled by choose_lan_access

write_satellite_config() {
  local token
  token=$(random_secret)
  [ "${#token}" -eq 32 ] || fail "Could not generate the satellite's token."
  choose_lan_access
  (
    umask 077
    cat >"$KEEP_DIR/config.yaml" <<EOF
# Keep configuration of an iMessage satellite, written by install.sh on $(date '+%Y-%m-%d'). No assistant
# runs on this install: it serves this Mac user's Messages to a Keep server, which is given the
# address and the token below — as imessage.host and imessage.host_token when the account is the
# owner's own, as imessage_chat.host and imessage_chat.host_token when it is the assistant's own
# Apple ID. config.example.yaml documents the keys.
role: imessage_satellite
imessage_satellite:
  port: $SATELLITE_PORT
  # What the Keep server must present. A secret: whoever holds it and reaches the port reads and
  # sends this account's messages. A new value here must be entered on the server as well.
  token: $token
  # false: only this Mac reaches the satellite (the Keep server is another user of this Mac).
  # true: the network does (the server is on another Mac); the token and the messages then cross
  # it unencrypted, so only on a network you trust. Read at start.
  lan_access: $LAN_ACCESS
# Where the install came from. A running satellite follows the Keep server that calls it: it
# moves itself to that server's line (release or master) and version.
$(if [ "$BRANCH" = master ]; then echo "developer_release: true   # installed from master, the development line"
   else echo "developer_release: false  # installed from the release branch"; fi)
# Nobody sits at a satellite and no assistant is there to restart it, so launchd starts it again
# within 5 minutes of dying. Applied at each start: to keep a stopped satellite down, set this to
# false and run start.sh once more, then stop.sh.
watchdog: true
telemetry:
  # The installer's own events (started, finished, or the step it failed in); a running satellite
  # sends nothing. config.example.yaml says what goes out. enabled: false turns it off.
$(if [ -n "${KEEP_NO_TELEMETRY:-}" ]; then echo "  enabled: false"
   else echo "  enabled: true"; echo "  id: $TELEMETRY_ID"; fi)
EOF
  )
  ok "Wrote config.yaml (readable by you only)"
}

# Whether the satellite answers the network: KEEP_LAN_ACCESS, else one question, else — with no
# terminal to ask on — this Mac only, the answer that opens nothing.
choose_lan_access() {
  LAN_ACCESS=false
  case "$(printf '%s' "${KEEP_LAN_ACCESS:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)  LAN_ACCESS=true
                    dim "KEEP_LAN_ACCESS is set: the satellite will answer the network (a Keep server on another Mac)."
                    return ;;
    0|false|no|off) dim "KEEP_LAN_ACCESS is off: the satellite will answer this Mac only (a Keep server under another user here)."
                    return ;;
    "") ;;
    *) fail "KEEP_LAN_ACCESS=${KEEP_LAN_ACCESS:-} is neither 1 nor 0." ;;
  esac
  if [ -z "$TTY" ]; then
    dim "No terminal to ask on: the satellite will answer this Mac only (imessage_satellite.lan_access: false)."
    dim "For a Keep server on another Mac, set it to true in config.yaml and run start.sh again."
    return
  fi
  note "Where does your Keep server run?"
  note "  1  on this Mac, under another user: the satellite answers this Mac only"
  note "  2  on another Mac: the satellite answers the network, and the token and the messages"
  note "     cross it unencrypted — only on a network you trust"
  local answer="" attempt=0
  while :; do
    attempt=$((attempt + 1))
    ask "1 or 2 (Enter for 1):" answer
    case "$(printf '%s' "$answer" | tr -d '[:space:]')" in
      ""|1) ok "This Mac only (lan_access: false)"; return ;;
      2)    LAN_ACCESS=true; ok "The network (lan_access: true)"; return ;;
    esac
    if [ "$attempt" -ge 3 ]; then
      warn "No 1 or 2 after 3 attempts: this Mac only. imessage_satellite.lan_access in config.yaml changes it."
      return
    fi
    warn "Please answer 1 or 2."
  done
}

# --------------------------------------------------------- telemetry --------

# The install reports to the PostHog project the server reports to (assistant/Telemetry.py
# says what goes out from there and why): started, finished, or the step it failed in and
# why — a tester whose install died is otherwise never heard of. The key is the project's
# write-only one, meant to ship in clients. The install id is minted here and written into
# config.yaml, so the server's events carry the same one and the install joins the turns
# that follow it. A re-run reads the id, and the opt-out, from the existing config.yaml.
POSTHOG_HOST="https://eu.i.posthog.com"
POSTHOG_PROJECT_KEY="phc_nvbSSUXHLCDv5oDic5ULNwQXTJT2Xk4SrdMSQbMGjyCk"

init_telemetry() {
  TELEMETRY_ID=""
  if [ -n "${KEEP_NO_TELEMETRY:-}" ]; then log "telemetry: off (KEEP_NO_TELEMETRY is set)"; return; fi
  if [ -f "$KEEP_DIR/config.yaml" ]; then
    if [ "$(config_telemetry_value enabled)" = false ]; then
      log "telemetry: off (telemetry.enabled is false in config.yaml)"; return
    fi
    TELEMETRY_ID=$(config_telemetry_value id)
  fi
  [ -n "$TELEMETRY_ID" ] || TELEMETRY_ID=$(uuidgen 2>/dev/null | tr 'A-Z' 'a-z' || true)
  log "telemetry: install id ${TELEMETRY_ID:-none}"
}

# One value of config.yaml's telemetry section: the key's line among those indented under
# `telemetry:`, up to the next top-level key, with the comment and any quotes stripped.
config_telemetry_value() { config_value telemetry "$1"; }

# The reader of config.yaml, the same program scripts/role.sh has between the same two marker
# lines — the installer cannot source that file, it runs before the checkout exists — so that an
# existing install is what start.sh will take it for (tests/unit/test_install_sh.py compares them).
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

# config_raw <section> <key>: one value as written, without its comment; with an empty key, the
# value of the top-level key <section> itself (`role`). The last one when the key stands twice,
# as yaml reads it; nothing when the file or the key is missing.
config_raw() {
  LC_ALL=C awk -v top="$1" -v inner="${2:-}" -v ask=value "$CONFIG_READER" "$KEEP_DIR/config.yaml" 2>/dev/null || true
}

# config_value <section> <key>: the same, with any quotes off.
config_value() { config_raw "$1" "${2:-}" | tr -d "\"'"; }

# config_trouble <section> <key>: why the key cannot be read from config.yaml — the file is
# written in a way the reader does not follow — and nothing when it can.
config_trouble() {
  LC_ALL=C awk -v top="$1" -v inner="${2:-}" -v ask=trouble "$CONFIG_READER" "$KEEP_DIR/config.yaml" 2>/dev/null || true
}

# The role as an event property — for a satellite only, so a server's events stay as they were.
telemetry_role() { if satellite; then printf 'role=%s' "$ROLE"; fi; }

telemetry_started() {
  local fresh=true
  [ ! -f "$KEEP_DIR/config.yaml" ] || fresh=false
  telemetry_event install_started fresh=$fresh branch="$BRANCH" terminal="${TERM_PROGRAM:-unknown}" \
    macos="$(sw_vers -productVersion 2>/dev/null || echo unknown)" \
    chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)" \
    default_dir=$([ "$KEEP_DIR" = "$HOME/Keep" ] && echo true || echo false) $(telemetry_role)
}

# telemetry_event <event> [name=value ...]: one event, posted right away. A number or
# true/false goes as one, anything else as a string; the home directory in a value becomes
# ~, since the path holds the user's name. Never holds the install up: five seconds at most,
# and the outcome goes to the log only.
telemetry_event() {
  [ -n "${TELEMETRY_ID:-}" ] || return 0
  local event=$1 props="" kv name value code
  shift
  for kv in "$@"; do
    name=${kv%%=*}
    value=${kv#*=}
    value=${value//"$HOME"/"~"}
    case "$value" in
      true|false) ;;
      ''|*[!0-9]*) value=${value//\\/\\\\}; value=${value//\"/\\\"}; value="\"$value\"" ;;
    esac
    props="$props,\"$name\":$value"
  done
  local json="{\"api_key\":\"$POSTHOG_PROJECT_KEY\",\"event\":\"$event\",\"distinct_id\":\"$TELEMETRY_ID\",\"properties\":{\"\$lib\":\"keep-install\"$props}}"
  code=$(curl -sS -o /dev/null --max-time 5 -w '%{http_code}' -H 'content-type: application/json' \
    -d "$json" "$POSTHOG_HOST/capture/" 2>>"${LOG:-/dev/null}" || true)
  log "telemetry: $event -> HTTP ${code:-none}"
}

# ----------------------------------------------------------- summary --------

summary() {
  if satellite; then satellite_summary; return; fi
  printf '\n  %s✔ Keep is installed%s\n\n' "$C_GREEN$C_BOLD" "$C_RESET"
  note "Folder     $KEEP_DIR"
  note "Config     $KEEP_DIR/config.yaml"
  note "Log        $LOG"
  note "Start      $KEEP_DIR/start.sh      (server + console)"
  note "Stop       $KEEP_DIR/stop.sh"
  note "Upgrade    re-run the install command"
  printf '\n'
}

# What a satellite's owner needs next, read back from config.yaml as it stands — so a re-run on an
# existing satellite shows the same lines again, with the port and the token that install really
# has. This is the one place the token is shown: on the screen (note), never in the install log.
satellite_summary() {
  local port token lan name address hint
  port=$(config_value imessage_satellite port)
  port=${port:-$SATELLITE_PORT}
  token=$(config_value imessage_satellite token)
  lan=$(config_value imessage_satellite lan_access | tr '[:upper:]' '[:lower:]')
  printf '\n  %s✔ The Keep iMessage satellite is installed%s\n\n' "$C_GREEN$C_BOLD" "$C_RESET"
  note "Folder     $KEEP_DIR"
  note "Config     $KEEP_DIR/config.yaml"
  note "Log        $LOG"
  note "Start      $KEEP_DIR/start.sh      (the satellite alone: no assistant, no console)"
  note "Stop       $KEEP_DIR/stop.sh       (with watchdog: true in config.yaml it is started again within 5 minutes)"
  note "Upgrade    by itself: it follows the version of the Keep server that calls it"
  printf '\n'
  note "${C_BOLD}Pair it with your Keep server.${C_RESET} The server needs two values, under the keys that fit whose"
  note "Messages this Mac user is signed in to. Ask the server's assistant to set it up: it lets you type"
  note "them where it does not see them (config_request)."
  printf '\n'
  hint=""
  if [ "$lan" = true ]; then
    name=$(scutil --get LocalHostName 2>/dev/null || true)
    address="http://<this Mac's address>:$port"
    [ -z "$name" ] || address="http://$name.local:$port"
  else
    address="http://127.0.0.1:$port"
    hint=local
  fi
  note "Your own Messages — the assistant reads and sends your iMessages (the iMessage integration):"
  note "  imessage.host        $address"
  [ -z "$token" ] || note "  imessage.host_token  $token"
  printf '\n'
  note "The assistant's own Apple ID — people text it and add it to groups (the iMessage chat):"
  note "  imessage_chat.host        $address"
  [ -z "$token" ] || note "  imessage_chat.host_token  $token"
  printf '\n'
  if [ -n "$hint" ]; then
    note "The address is this Mac's own: the server runs on this Mac, under another user. For a server on"
    note "another Mac: imessage_satellite.lan_access: true in config.yaml, then start.sh."
  else
    note "The address is this Mac's name or IP address on your network, and the port."
  fi
  if [ -z "$token" ]; then
    warn "config.yaml holds no imessage_satellite.token: the satellite does not start without one (16+ random characters)."
  fi
  printf '\n'
  note "The token is a secret: whoever holds it and reaches the port reads and sends this account's"
  note "messages. It is shown here only, and stays in $KEEP_DIR/config.yaml."
  printf '\n'
  note "On this Mac, under this user ($(id -un)), the satellite needs:"
  note "  - Messages signed in to iMessage with the Apple ID it serves"
  note "  - Full Disk Access for Terminal: System Settings > Privacy & Security > Full Disk Access."
  note "    macOS applies it to a relaunched Terminal: quit Terminal, open it again and run start.sh"
  note "  - Automation: allow Terminal to control Messages when macOS asks, at the first message sent,"
  note "    and System Events, asked once Messages is readable — that makes the satellite start at login"
  note "  - Optional, for typing dots while Keep answers as an iMessage contact of its own: Accessibility"
  note "    for Terminal (System Settings > Privacy & Security > Accessibility). The dots show only while"
  note "    this user is logged in on the Mac's screen, the screen is unlocked and Messages is open"
  printf '\n'
}

launch() {
  local what=server
  if satellite; then what=satellite; fi
  if [ -n "${KEEP_NO_START:-}" ]; then
    dim "KEEP_NO_START is set; not starting the $what."
    exit 0
  fi
  if [ -z "$TTY" ] || [ -z "$IS_TTY" ]; then
    dim "Not an interactive terminal; start Keep with: $KEEP_DIR/start.sh"
    exit 0
  fi
  if satellite; then
    dim "Starting the satellite. Its start is shown here; it then runs on without this window."
    note "macOS may ask to let Terminal control Terminal."
  else
    dim "Starting Keep. The first start loads models for a while; the console takes over when the server is up."
    note "macOS may ask to let Terminal control Terminal and, once voice is enabled, to use the microphone."
  fi
  printf '\n'
  trap - ERR EXIT INT TERM
  cd "$KEEP_DIR"
  exec ./start.sh <"$TTY"
}

main "$@"
