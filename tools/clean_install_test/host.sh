#!/bin/bash
# The half of the clean-install test that runs on the VM host (run.sh copies it there and calls it;
# it can be run there by hand too). It keeps everything under ~/keep-install-tests.
#
#   host.sh run [KEY=VALUE ...]   a fresh VM, the installer in it with those variables, the logs out
#   host.sh list                  the test VMs that are left
#   host.sh shell <vm>            a shell in one that is running
#   host.sh delete <vm>|all       stops and deletes test VMs (the golden image stays)
#   host.sh golden                builds the golden image if it is missing
#
# Settings of `run`, from the environment: MACOS (sequoia), MEMORY_GB (32), DISK_GB (120), CPUS (8),
# KEEP_VM (failure | always | never: when the VM is left for a look), UNATTENDED (0 | 1), and SOURCE —
# `branch:<name>` (install.sh and the code from GitHub) or `bundle:<file>` (a git bundle run.sh sent).
#
# The golden image is the vanilla macOS image (github.com/cirruslabs/macos-image-templates: a new
# Mac with the user admin / admin, no Homebrew, no Command Line Tools) plus one ssh key. Every run
# is a clone of it, so every install starts on the same clean Mac.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"

DIR="$HOME/keep-install-tests"
KEY="$DIR/id_ed25519"
MACOS="${MACOS:-sequoia}"
GOLDEN="keep-clean-$MACOS"
IMAGE="ghcr.io/cirruslabs/macos-$MACOS-vanilla:latest"
GUEST_USER=admin
GUEST_PASSWORD=admin
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5)

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
  halt "$GOLDEN"
  say "$GOLDEN is ready"
}

# The clone's disk is the image's until the container is grown into what `tart set` added.
grow_disk() {
  guest "$1" -- "echo $GUEST_PASSWORD | sudo -S sh -c 'yes | diskutil repairDisk disk0 >/dev/null 2>&1; diskutil apfs resizeContainer disk0s2 0' >/dev/null 2>&1" || true
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
      # The bundle is the repository the installer clones (KEEP_REPO). git comes with the Command
      # Line Tools, which a clean Mac lacks: install.sh is taken out of the bundle only once it has
      # put them there, so the first stretch of it runs from the copy sent beside the bundle.
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

run() {
  local memory=${MEMORY_GB:-32} disk=${DISK_GB:-120} cpus=${CPUS:-8} keep=${KEEP_VM:-failure}
  local source=${SOURCE:-branch:release} vm status=0 out
  golden
  vm="keep-test-$(date +%m%d-%H%M%S)"
  out="$DIR/runs/$vm"
  mkdir -p "$out"
  say "A clean Mac: $vm (macOS $MACOS, $memory GB of memory, $disk GB of disk, $cpus cores)"
  tart clone "$GOLDEN" "$vm"
  tart set "$vm" --memory $((memory * 1024)) --cpu "$cpus" --disk-size "$disk"
  boot "$vm"
  grow_disk "$vm"
  if [ "${UNATTENDED:-0}" = 1 ]; then
    # Nobody types a password: sudo asks for none in this VM, so Homebrew's installer and the
    # Command Line Tools go through. An attended run leaves sudo as a new Mac has it.
    guest "$vm" -- "echo $GUEST_PASSWORD | sudo -S sh -c 'echo \"$GUEST_USER ALL=(ALL) NOPASSWD: ALL\" > /etc/sudoers.d/keep-test'" 2>/dev/null
  fi
  write_install_script "$vm" "$source" "$@"
  say "The installer ($source${*:+, with $*}); the guest's password is '$GUEST_PASSWORD'"
  # -tt: the installer asks on /dev/tty, and an attended run is typed into from here.
  guest "$vm" -tt -- "~/keep-install.sh" 2>&1 | tee "$out/screen.log" || status=${PIPESTATUS[0]}
  say "The installer ended with status $status"
  collect "$vm" "$out"
  printf 'RESULT status=%s vm=%s out=%s\n' "$status" "$vm" "$out"
  if [ "$keep" = never ] || { [ "$keep" = failure ] && [ "$status" -eq 0 ]; }; then
    halt "$vm"; tart delete "$vm"
    echo "The VM is deleted."
  else
    echo "The VM is left running:  host.sh shell $vm   (delete it with: host.sh delete $vm)"
  fi
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
  list) tart list --source local | awk 'NR == 1 || $2 ~ /^keep-(test|clean)-/' ;;
  shell) [ -n "${2:-}" ] || die "usage: host.sh shell <vm>"; guest "$2" -tt -- ;;
  delete)
    [ -n "${2:-}" ] || die "usage: host.sh delete <vm>|all"
    for vm in $(tart list --quiet --source local | grep '^keep-test-'); do
      if [ "$2" = all ] || [ "$2" = "$vm" ]; then halt "$vm"; tart delete "$vm"; echo "deleted $vm"; fi
    done ;;
  *) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
