#!/bin/bash
# The half of the clean-install test that runs on the VM host (run.sh copies it there and calls it;
# it can be run there by hand too). It keeps everything under ~/keep-install-tests.
#
#   host.sh run [KEY=VALUE ...]   a fresh VM, the installer in it with those variables, the logs out
#   host.sh fresh                 a fresh VM and nothing run in it: for an install typed by hand in
#                                 its own Terminal, over screen sharing, where the system's consent
#                                 dialogs show
#   host.sh ip <vm>               its address, as the host reaches it
#   host.sh save-model <vm>       keeps the local model a VM downloaded, for MODEL=seeded|partial
#   host.sh again <vm>            the same installer command once more in a VM that was kept: the
#                                 re-run a user does after a failure, or an upgrade
#   host.sh list                  the test VMs that are left
#   host.sh shell <vm>            a shell in one that is running
#   host.sh exec <vm> <command>   one command in it
#   host.sh delete <vm>|all       stops and deletes test VMs (the golden image stays)
#   host.sh golden                builds the golden image if it is missing
#
# Settings of `run`, from the environment: MACOS (sequoia), MEMORY_GB (32), DISK_GB (120), CPUS (8),
# KEEP_VM (failure | always | never: when the VM is left for a look), UNATTENDED (0 | 1), SOURCE —
# `branch:<name>` (install.sh and the code from GitHub) or `bundle:<file>` (a git bundle run.sh sent) —
# and MODEL: full (the installer downloads the local model, 18 GB), seeded (the model saved on the
# host is put into the guest first, so nothing is downloaded) or partial (all of it but the
# smallest weight file: the installer downloads that one, and its bar and its resuming are seen).
#
# The golden image is the vanilla macOS image (github.com/cirruslabs/macos-image-templates: a new
# Mac with the user admin / admin and the Command Line Tools, without Homebrew) plus one ssh key. Every run
# is a clone of it, so every install starts on the same clean Mac.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"

DIR="$HOME/keep-install-tests"
KEY="$DIR/id_ed25519"
MACOS="${MACOS:-sequoia}"
GOLDEN="keep-clean-$MACOS"
IMAGE="ghcr.io/cirruslabs/macos-$MACOS-vanilla:latest"
MODELS="$DIR/models"         # what save-model kept: the folders of a guest's ~/.omlx/models
GUEST_USER=admin
GUEST_PASSWORD=admin
SSH_OPTS=(-o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5)

say() { printf '\n== %s\n' "$*"; }
die() { printf 'clean-install test: %s\n' "$*" >&2; exit 1; }

exists() { tart list --quiet --source local | grep -qx "$1"; }
running() { tart list --source local | awk -v vm="$1" '$2 == vm { print $NF }' | grep -qx running; }

# guest <vm> [ssh options] -- <command>: a command in the VM, as its user.
guest() {
  local vm=$1 ip; shift
  local opts=()
  while [ "${1:-}" != "--" ]; do opts+=("$1"); shift; done
  shift
  ip=$(tart ip "$vm")
  ssh -i "$KEY" "${SSH_OPTS[@]}" ${opts[@]+"${opts[@]}"} "$GUEST_USER@$ip" "$@"
}

boot() {
  local vm=$1 ip
  nohup tart run "$vm" --no-graphics >"$DIR/$vm.run.log" 2>&1 &
  ip=$(tart ip "$vm" --wait 180) || die "$vm did not get an address (see $DIR/$vm.run.log)"
  for _ in $(seq 1 60); do
    if nc -z -G 2 "$ip" 22 2>/dev/null; then return 0; fi
    sleep 2
  done
  die "$vm ($ip) does not answer ssh"
}

halt() {
  local vm=$1
  running "$vm" || return 0
  tart stop "$vm" --timeout 60 >/dev/null 2>&1 || true
}

# shut_down <vm>: from inside, so that what was written is on its disk — a VM that is only stopped
# loses its last writes, and the golden image its key.
shut_down() {
  local vm=$1
  guest "$vm" -- "sync; echo $GUEST_PASSWORD | sudo -S shutdown -h now" >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    running "$vm" || return 0
    sleep 2
  done
  halt "$vm"
}

# The one login by password: it puts the key into the golden image. ssh asks for the password
# through SSH_ASKPASS, so nothing more has to be installed on the host.
install_key() {
  local vm=$1 ip askpass
  ip=$(tart ip "$vm")
  askpass=$(mktemp "$DIR/askpass.XXXXXX")
  printf '#!/bin/sh\necho %s\n' "$GUEST_PASSWORD" >"$askpass"
  chmod 700 "$askpass"
  SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force ssh "${SSH_OPTS[@]}" -o PubkeyAuthentication=no "$GUEST_USER@$ip" \
    "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" <"$KEY.pub"
  rm -f "$askpass"
}

golden() {
  mkdir -p "$DIR"
  command -v tart >/dev/null || die "tart is not installed on this host (brew install openai/tools/tart)"
  [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C keep-install-tests -f "$KEY"
  if exists "$GOLDEN"; then return 0; fi
  say "Building the golden image $GOLDEN from $IMAGE (once; the image is about 25 GB)"
  tart clone "$IMAGE" "$GOLDEN"
  boot "$GOLDEN"
  install_key "$GOLDEN"
  guest "$GOLDEN" -- true || die "the key did not get into $GOLDEN"
  shut_down "$GOLDEN"
  say "$GOLDEN is ready"
}

# The clone's disk is the image's size until the system's container is grown into what `tart set`
# added — which lies behind the recovery partition, so free_disk.py took that out before the boot.
grow_disk() {
  # repairDisk first: it moves the partition table's end to the end of the grown disk.
  guest "$1" -- "echo $GUEST_PASSWORD | sudo -S sh -c 'yes | diskutil repairDisk disk0; diskutil apfs resizeContainer disk0s2 0' >/dev/null 2>&1" || true
  guest "$1" -- "df -g / | awk 'NR == 2 { print \"disk: \" \$4 \" GB free\" }'"
}

# What the installer is started with in the guest: a script, so the same line serves an attended
# run (the person types what a user would) and an unattended one.
write_install_script() {
  local vm=$1 source=$2; shift 2
  local env_line="" pair
  for pair in "$@"; do env_line+=" $(printf '%q' "$pair")"; done
  case "$source" in
    branch:*)
      local branch=${source#branch:}
      guest "$vm" -- "cat > ~/keep-install.sh" <<EOF
#!/bin/bash
curl -fsSL "https://raw.githubusercontent.com/maciekd/Keep/$branch/install.sh" | env KEEP_BRANCH="$branch"$env_line bash
EOF
      ;;
    bundle:*)
      local bundle=${source#bundle:}
      [ -f "$bundle" ] || die "no such bundle: $bundle"
      scp -q -i "$KEY" "${SSH_OPTS[@]}" "$bundle" "$GUEST_USER@$(tart ip "$vm"):keep-src.bundle"
      # The bundle is the repository the installer clones (KEEP_REPO); install.sh itself is the
      # copy sent beside it, as curl would have fetched it.
      scp -q -i "$KEY" "${SSH_OPTS[@]}" "${bundle%.bundle}.install.sh" "$GUEST_USER@$(tart ip "$vm"):keep-src-install.sh"
      guest "$vm" -- "cat > ~/keep-install.sh" <<EOF
#!/bin/bash
env KEEP_REPO="\$HOME/keep-src.bundle" KEEP_BRANCH=keep-test$env_line bash "\$HOME/keep-src-install.sh"
EOF
      ;;
    *) die "SOURCE must be branch:<name> or bundle:<file>, not '$source'" ;;
  esac
  guest "$vm" -- "chmod +x ~/keep-install.sh"
}

# create_vm: a clone of the golden image with the run's memory and disk, booted; its name in VM.
create_vm() {
  local memory=${MEMORY_GB:-32} disk=${DISK_GB:-120} cpus=${CPUS:-8}
  golden
  VM="keep-test-$(date +%m%d-%H%M%S)"
  say "A clean Mac: $VM (macOS $MACOS, $memory GB of memory, $disk GB of disk, $cpus cores)"
  tart clone "$GOLDEN" "$VM"
  tart set "$VM" --memory $((memory * 1024)) --cpu "$cpus" --disk-size "$disk"
  python3 "$DIR/free_disk.py" "$HOME/.tart/vms/$VM/disk.img" >/dev/null
  boot "$VM"
  grow_disk "$VM"
  seed_model "$VM"
}

rsync_guest() { rsync -a -e "ssh -i $KEY ${SSH_OPTS[*]}" "$@"; }

# The model a VM downloaded, kept on the host (with the downloader's own notes under .cache, which
# are what make it take a file for complete).
save_model() {
  local vm=$1
  mkdir -p "$MODELS"
  rsync_guest "$GUEST_USER@$(tart ip "$vm"):.omlx/models/" "$MODELS/"
  du -sh "$MODELS"/* 2>/dev/null
}

# seed_model <vm>: MODEL=seeded puts the saved model into the guest, partial all of it but the
# smallest of its weight files (those over 1 GB), which the installer then downloads.
seed_model() {
  local vm=$1 mode=${MODEL:-full} skip="" excludes=()
  [ "$mode" != full ] || return 0
  [ -n "$(ls -A "$MODELS" 2>/dev/null)" ] || die "MODEL=$mode, but no model is saved on this host (host.sh save-model <vm>)"
  if [ "$mode" = partial ]; then
    skip=$(find "$MODELS" -name '*.safetensors' -size +1G -exec stat -f '%z %N' {} + | sort -n | head -n 1 | cut -d' ' -f2-)
    [ -n "$skip" ] || die "MODEL=partial: no weight file found under $MODELS"
    skip=$(basename "$skip")
    excludes=(--exclude "$skip" --exclude "$skip.metadata")
  fi
  guest "$vm" -- "mkdir -p ~/.omlx/models"
  rsync_guest ${excludes[@]+"${excludes[@]}"} "$MODELS/" "$GUEST_USER@$(tart ip "$vm"):.omlx/models/"
  echo "model: put into the guest${skip:+, without $skip}"
}

# fresh: the clean Mac and nothing more, with what to type to get into it.
fresh() {
  local ip
  create_vm
  ip=$(tart ip "$VM")
  printf '\nREADY vm=%s ip=%s user=%s password=%s\n' "$VM" "$ip" "$GUEST_USER" "$GUEST_PASSWORD"
}

run() {
  local keep=${KEEP_VM:-failure} source=${SOURCE:-branch:release} vm status=0 out
  create_vm
  vm=$VM
  out="$DIR/runs/$vm"
  mkdir -p "$out"
  if [ "${UNATTENDED:-0}" = 1 ]; then
    # Nobody types: sudo asks for no password in this VM and Homebrew's installer is told not to
    # wait for a key (NONINTERACTIVE, its own variable). An attended run leaves both as a new Mac
    # has them.
    guest "$vm" -- "echo $GUEST_PASSWORD | sudo -S sh -c 'echo \"$GUEST_USER ALL=(ALL) NOPASSWD: ALL\" > /etc/sudoers.d/keep-test'" 2>/dev/null
    set -- NONINTERACTIVE=1 "$@"
  fi
  # A test is not a tester: no install events unless the run asks for them (KEEP_NO_TELEMETRY=).
  case " $* " in *" KEEP_NO_TELEMETRY="*) ;; *) set -- KEEP_NO_TELEMETRY=1 "$@" ;; esac
  write_install_script "$vm" "$source" "$@"
  say "The installer ($source${*:+, with $*}); the guest's password is '$GUEST_PASSWORD'"
  install "$vm" "$out" || status=$?
  if [ "$keep" = never ] || { [ "$keep" = failure ] && [ "$status" -eq 0 ]; }; then
    halt "$vm"; tart delete "$vm"
    echo "The VM is deleted."
  else
    echo "The VM is left running:  host.sh shell $vm   (delete it with: host.sh delete $vm)"
  fi
  return "$status"
}

# install <vm> <out>: the installer in the guest, its screen kept, what it left collected.
install() {
  local vm=$1 out=$2 status=0
  # -tt: the installer asks on /dev/tty, and an attended run is typed into from here.
  guest "$vm" -tt -- "~/keep-install.sh" 2>&1 | tee -a "$out/screen.log" || status=${PIPESTATUS[0]}
  say "The installer ended with status $status"
  collect "$vm" "$out"
  printf 'RESULT status=%s vm=%s out=%s\n' "$status" "$vm" "$out"
  return "$status"
}

# What a run leaves to read: the install log, the config without its secrets, oMLX's side.
collect() {
  local vm=$1 out=$2 ip
  ip=$(tart ip "$vm")
  scp -q -i "$KEY" "${SSH_OPTS[@]}" "$GUEST_USER@$ip:Keep/log/install.log" "$out/install.log" 2>/dev/null || true
  guest "$vm" -- 'sed -E "s/^( *(anthropic|key|token|api_key|id):).*/\1 <removed>/" ~/Keep/config.yaml 2>/dev/null' >"$out/config.yaml" || true
  guest "$vm" -- 'export PATH=/opt/homebrew/bin:$PATH; sw_vers -productVersion; command -v brew && { brew list --versions omlx; brew services list; brew list --pinned; }; ls ~/.omlx/models 2>/dev/null; cat ~/.omlx/settings.json 2>/dev/null; tail -n 40 /opt/homebrew/var/log/omlx.log 2>/dev/null' >"$out/machine.txt" 2>&1 || true
}

case "${1:-}" in
  run) shift; run "$@" ;;
  golden) golden ;;
  fresh) fresh ;;
  ip) [ -n "${2:-}" ] || die "usage: host.sh ip <vm>"; tart ip "$2" ;;
  save-model) [ -n "${2:-}" ] || die "usage: host.sh save-model <vm>"; save_model "$2" ;;
  list) tart list --source local | awk 'NR == 1 || $2 ~ /^keep-(test|clean)-/' ;;
  shell) [ -n "${2:-}" ] || die "usage: host.sh shell <vm>"; guest "$2" -tt -- ;;
  again) [ -n "${2:-}" ] || die "usage: host.sh again <vm>"; mkdir -p "$DIR/runs/$2"; install "$2" "$DIR/runs/$2" ;;
  collect) [ -n "${2:-}" ] || die "usage: host.sh collect <vm>"; mkdir -p "$DIR/runs/$2"; collect "$2" "$DIR/runs/$2" ;;
  exec) [ -n "${3:-}" ] || die "usage: host.sh exec <vm> <command>"; vm=$2; shift 2; guest "$vm" -- "$@" ;;
  delete)
    [ -n "${2:-}" ] || die "usage: host.sh delete <vm>|all"
    for vm in $(tart list --quiet --source local | grep '^keep-test-'); do
      if [ "$2" = all ] || [ "$2" = "$vm" ]; then halt "$vm"; tart delete "$vm"; echo "deleted $vm"; fi
    done ;;
  *) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
