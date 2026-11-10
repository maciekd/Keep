#!/bin/bash
#
# Moves a Keep install to another Mac. Everything the user owns sits inside the checkout
# (the code and the venv are rebuilt by the installer), so a migration is: pack that state
# here, install Keep there, unpack it over the fresh checkout, start.
#
#   scripts/migrate.sh pack              writes ~/keep-migration-<date>.zip  (default)
#   scripts/migrate.sh unpack <zip>      on the new Mac, from the fresh checkout, before the first start
#
# Carried: config.yaml (keys, accounts, the vault passphrase), vault.bin, floxy.db with its
# SQLite sidecars, memory/, first_run.json, voice_id/, the signed-in sessions (cache/ and the
# WhatsApp and Telegram feed session files in the root). Left behind: .venv, log/, the wipe
# backups and Chrome's own disk caches, which are rebuilt. macOS permissions (microphone,
# Full Disk Access, Photos) do not travel; the new Mac asks for them again.
#
# Written for the bash 3.2 that ships with macOS. zip and unzip are part of macOS.

set -euo pipefail

KEEP_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$KEEP_DIR"

STATE=(
  config.yaml vault.bin
  floxy.db floxy.db-wal floxy.db-shm floxy.db-journal
  memory first_run.json voice_id cache
  whatsapp.session whatsapp.session-journal whatsapp.session-wal whatsapp.session-shm
  telegram_feed.session telegram_feed.session-journal
)
# Rebuilt on the new Mac, and the bulk of cache/: the wipe backups and Chrome's disk caches.
EXCLUDE=(
  "cache/wipe-backups/*" "cache/voice/*"
  "cache/chrome_profile/*/Cache/*" "cache/chrome_profile/*/Code Cache/*"
  "cache/chrome_profile/*/GPUCache/*" "cache/chrome_profile/*/Service Worker/*"
  "cache/chrome_profile/GrShaderCache/*" "cache/chrome_profile/GraphiteDawnCache/*"
  "cache/chrome_profile/ShaderCache/*" "cache/chrome_profile/*_crx_cache/*"
  "cache/chrome_profile/optimization_guide_model_store/*"
)

die() { printf '\n  ✖ %s\n\n' "$1" >&2; exit 1; }

require_stopped() {
  # Chrome writes into its profile and SQLite into the database while the server runs:
  # a copy taken then is torn. Same rule either way: nothing runs during a pack or unpack.
  # This user's processes only: the Keep of another macOS user on this Mac writes into its own tree.
  if pgrep -u "$(id -u)" -f floxy_ai_server >/dev/null 2>&1; then
    die "Keep is running. Stop it first (./stop.sh), then run this again."
  fi
}

pack() {
  require_stopped
  [ -f config.yaml ] || die "No config.yaml here: this does not look like an installed Keep."
  local out="$HOME/keep-migration-$(date '+%Y%m%d-%H%M').zip"
  local present=()
  for item in "${STATE[@]}"; do
    [ -e "$item" ] && present+=("$item")
  done
  printf '\n  Packing from %s:\n' "$KEEP_DIR"
  printf '    %s\n' "${present[@]}"
  rm -f "$out"
  zip -qry "$out" "${present[@]}" -x "${EXCLUDE[@]}"
  printf '\n  ✔ %s (%s)\n\n' "$out" "$(du -h "$out" | cut -f1)"
  cat <<EOF
  On the new Mac:
    1. install Keep without starting it:
         curl -fsSL https://raw.githubusercontent.com/maciekd/Keep/release/install.sh \\
           | KEEP_NO_START=1 bash
    2. copy the zip over and unpack it into the new checkout:
         ~/Keep/scripts/migrate.sh unpack ~/$(basename "$out")
    3. ./start.sh
  The zip holds your API keys, vault and logins: delete it on both Macs once Keep runs there.

EOF
}

unpack() {
  local archive=${1:-}
  [ -n "$archive" ] || die "Usage: scripts/migrate.sh unpack <keep-migration-....zip>"
  [ -f "$archive" ] || die "No such file: $archive"
  require_stopped
  [ -d .git ] || die "Run this from a Keep checkout (the installer's ~/Keep)."
  if [ -f floxy.db ]; then
    printf '\n  ! This checkout already has data (floxy.db). Unpacking overwrites it.\n'
    printf '    Continue? [y/N] '
    read -r answer
    case "$answer" in y|Y|yes) ;; *) die "Nothing unpacked." ;; esac
  fi
  printf '\n  Unpacking %s into %s\n' "$archive" "$KEEP_DIR"
  # Sidecars of a database that was stopped cleanly here would pair a copied database with a
  # stale journal: drop them before the archive's own (if any) come in.
  rm -f floxy.db-wal floxy.db-shm floxy.db-journal
  unzip -qo "$archive" -d "$KEEP_DIR"
  printf '  ✔ done. Start Keep with ./start.sh\n\n'
}

case "${1:-pack}" in
  pack)   pack ;;
  unpack) shift; unpack "$@" ;;
  *)      die "Usage: scripts/migrate.sh [pack | unpack <zip>]" ;;
esac
