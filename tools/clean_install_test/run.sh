#!/bin/bash
# Installs Keep on a clean Mac: a fresh macOS VM on another machine, made from a golden image for
# every run, with the installer run in it as a new user would run it. For whoever changes
# install.sh or src/common/install — the unit tests run them over fakes, this runs them for real.
#
#   tools/clean_install_test/run.sh [options] [KEY=VALUE ...]
#
# KEY=VALUE are the installer's own variables (the top of install.sh): KEEP_AI=local,
# ANTHROPIC_API_KEY=..., KEEP_NO_START=1, KEEP_ROLE=imessage_satellite ...
# KEEP_NO_TELEMETRY=1 is set for you, so a test sends no install events; KEEP_NO_TELEMETRY= (empty)
# turns that off.
#
#   --host <ssh host>     the machine the VMs run on (default: $KEEP_TEST_HOST). An Apple Silicon
#                         Mac with tart (brew install openai/tools/tart) and room: a run takes the
#                         memory it is given and up to its disk size.
#   --worktree            install this checkout as it stands, uncommitted changes included (default)
#   --branch <name>       install a branch from GitHub instead, as the readme's command would
#   --macos <name>        the guest's macOS: sequoia (15, default), tahoe (26) — an image of
#                         github.com/cirruslabs/macos-image-templates, `macos-<name>-vanilla`
#   --memory <GB>         the guest's memory (default 32); --disk <GB> (120); --cpus <n> (8)
#   --unattended          nobody at the keyboard: sudo asks for no password in the guest, so give
#                         every answer as a variable. Without it the run is yours to type into —
#                         the installer's questions, and the password (admin) where sudo asks.
#   --keep always|failure|never   when the VM is left running for a look (default: failure)
#
#   tools/clean_install_test/run.sh list | shell <vm> | delete <vm>|all | golden
#   tools/clean_install_test/run.sh again <vm>     the installer once more in a VM that was kept
#
# What a run leaves here: log/clean_install/<vm>/ with screen.log (what the user saw), install.log,
# config.yaml without its secrets and machine.txt (Homebrew, oMLX, its settings and log).
#
# A VM has the host's GPU through a virtual device; whether a local model runs on it is the
# guest's affair, and a run that reaches the model's first answer is what shows it.
set -euo pipefail
cd "$(dirname "$0")/../.."

HOST="${KEEP_TEST_HOST:-}"
SOURCE=worktree
MACOS=sequoia; MEMORY_GB=32; DISK_GB=120; CPUS=8; KEEP_VM=failure; UNATTENDED=0
REMOTE_DIR=keep-install-tests
VARS=()
COMMAND=run

die() { printf 'clean-install test: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST=$2; shift 2 ;;
    --worktree) SOURCE=worktree; shift ;;
    --branch) SOURCE="branch:$2"; shift 2 ;;
    --macos) MACOS=$2; shift 2 ;;
    --memory) MEMORY_GB=$2; shift 2 ;;
    --disk) DISK_GB=$2; shift 2 ;;
    --cpus) CPUS=$2; shift 2 ;;
    --keep) KEEP_VM=$2; shift 2 ;;
    --unattended) UNATTENDED=1; shift ;;
    -h|--help) sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    list|golden) COMMAND=$1; shift ;;
    shell|delete|again) COMMAND="$1 ${2:?$1 needs a VM}"; shift 2 ;;
    *=*) VARS+=("$1"); shift ;;
    *) die "unknown argument '$1' (--help)" ;;
  esac
done
[ -n "$HOST" ] || die "name the VM host: --host <ssh host>, or KEEP_TEST_HOST in the environment"

ssh "$HOST" "mkdir -p $REMOTE_DIR"
scp -q tools/clean_install_test/host.sh tools/clean_install_test/free_disk.py "$HOST:$REMOTE_DIR/"
TTY_FLAG=-t; [ -t 0 ] || TTY_FLAG=-T      # without a terminal here there is none to pass on
remote() { ssh "$TTY_FLAG" "$HOST" "cd $REMOTE_DIR && chmod +x host.sh && $*"; }

# fetch <vm>: what the run left on the host, into log/clean_install/<vm>/ here.
fetch() {
  [ -n "$1" ] || return 0
  mkdir -p "log/clean_install/$1"
  scp -q "$HOST:$REMOTE_DIR/runs/$1/*" "log/clean_install/$1/" 2>/dev/null || true
  echo "What the run left: log/clean_install/$1/"
}

if [ "$COMMAND" != run ]; then
  status=0
  # shellcheck disable=SC2086
  remote "MACOS=$MACOS ./host.sh $COMMAND" || status=$?
  case "$COMMAND" in again\ *) fetch "${COMMAND#again }" ;; esac
  exit "$status"
fi

# The checkout as it stands, as a repository the installer can clone: a commit of the working tree
# (the tracked and the untracked files, never the ignored ones — config.yaml, the logs) made in a
# throwaway clone, so this checkout's index, branches and stash are not touched.
bundle_worktree() {
  local work index tree commit
  work=$(mktemp -d)
  index="$work/index"
  cp "$(git rev-parse --git-path index)" "$index"
  GIT_INDEX_FILE="$index" git add -A
  tree=$(GIT_INDEX_FILE="$index" git write-tree)
  commit=$(git commit-tree "$tree" -p HEAD -m "The working tree, for a clean-install test")
  git clone -q --shared --no-checkout . "$work/repo"
  git -C "$work/repo" update-ref refs/heads/keep-test "$commit"
  git -C "$work/repo" bundle create -q "$work/keep-src.bundle" keep-test
  cp install.sh "$work/keep-src.install.sh"
  scp -q "$work/keep-src.bundle" "$work/keep-src.install.sh" "$HOST:$REMOTE_DIR/"
  rm -rf "$work"
}

if [ "$SOURCE" = worktree ]; then
  echo "Sending this checkout ($(git rev-parse --short HEAD) and its uncommitted changes) to $HOST..."
  bundle_worktree
  SOURCE="bundle:\$HOME/$REMOTE_DIR/keep-src.bundle"
fi

quoted=""
for pair in ${VARS[@]+"${VARS[@]}"}; do quoted+=" $(printf '%q' "$pair")"; done
screen=$(mktemp)
status=0
remote "MACOS=$MACOS MEMORY_GB=$MEMORY_GB DISK_GB=$DISK_GB CPUS=$CPUS KEEP_VM=$KEEP_VM UNATTENDED=$UNATTENDED SOURCE=$SOURCE ./host.sh run$quoted" \
  | tee "$screen" || status=${PIPESTATUS[0]}

vm=$(sed -n 's/^RESULT .* vm=\([^ ]*\) .*/\1/p' "$screen" | tr -d '\r' | tail -n 1)
rm -f "$screen"
fetch "$vm"
exit "$status"
