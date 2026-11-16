#!/bin/bash
# Publishes master as the release: fast-forwards origin's `release` branch to master's tip
# and tags the commit with its version (the vYYYY.MM.N of common/Version.py). Installs track
# `release` — install.sh clones it and restart_server pulls it — while the development
# machine stays on master, so a release is nothing more than moving that branch.
#
#   scripts/release.sh            release what origin/master is at
#   scripts/release.sh <commit>   release an earlier master commit instead (must still be ahead
#                           of the current release; a rollback is a deliberate
#                           `git push -f origin <commit>:release`)
#   scripts/release.sh --dry-run [<commit>]
#                           run the checks and print the what's new, but neither tag nor push
#   scripts/release.sh --ignore-commits <hash>[,<hash>…] [<commit>]
#                           leave these released commits out of the what's new (a change
#                           the household should not be told about, or one a later commit
#                           undid); the commits are released all the same
#   scripts/release.sh --urgent "<reason>" [<commit>]
#                           an emergency release: the reason goes into the tag's subject
#                           (`Release v… urgent: <reason>`), and an install that finds it at its
#                           hourly check updates at once instead of at night, telling its owner
#                           the reason (SignedRelease.urgent, CodeQualityMaintainer._urgent_update).
#                           The signature covers it. It restarts every household in the middle
#                           of the day — for what cannot wait for 03:00, nothing else
#   scripts/release.sh [<commit>] <guidance for the note…>
#                           any words that are not a commit go to Claude as guidance for the
#                           what's new, e.g. `scripts/release.sh dont talk about icloud accounts
#                           being blocked`
#
# Before tagging, Claude Code (`claude -p`) reads the released commits and writes a "what's
# new" for the household, which becomes the body of the annotated tag. An install's pull
# brings the tag along with the commit, and the server's first start on it announces the
# note (common/StartupTracker.py reads it through Version.release_notes()). A missing
# `claude` or a failed run leaves the tag without a note and never fails the release.
#
# The tag is signed (`git tag -s`, the SSH key `user.signingkey` names — the maintainer's, held in
# 1Password) and verified before the push against res/release_signers as the current release has
# it: installs move to a release only when its tag verifies against the signers file of the code
# they already run (common/SignedRelease.py), so a push to `release` without that key reaches no one.
#
# The commits listed and summarised reach back to the last release tagged on an earlier day,
# not merely to the previous release: installs update nightly, so a household that gets the
# evening release never saw the midday one, and its note has to cover the whole day.
set -euo pipefail
cd "$(dirname "$0")/.."

ignored=()  # full hashes of the --ignore-commits, resolved once the release range is known
is_ignored() {  # $1: a full commit hash
  local h
  for h in ${ignored[@]+"${ignored[@]}"}; do [ "$h" = "$1" ] && return 0; done
  return 1
}

whats_new() {  # $1..$2: the released commits — prints Claude's user-facing summary of them
  command -v claude >/dev/null || { echo "claude is not on PATH — no what's new." >&2; return 0; }
  local log="" c
  while read -r c; do
    is_ignored "$c" && continue
    log+="$(git log -1 --format='%h %s%n%b' "$c")"$'\n'
  done < <(git rev-list --reverse "$1..$2")
  local left_out=""
  [ ${#ignored[@]} -gt 0 ] && left_out="
Some commits of the release were deliberately left out of the list; write only about the ones listed and do not
go looking for others."
  local guided=""
  [ -n "$guidance" ] && guided="

Guidance from the maintainer for this note, which overrides the rules above where they conflict:
$guidance"
  local prompt
  prompt="You are writing the release note for a home assistant called Keep, sent to the users of the system.
The release is $version. Below are the commits it brings, oldest first, as '<hash> <subject>' followed by the body.
Run \`git show <hash>\` or \`git show --stat <hash>\` when a message does not make the user-facing effect clear.$left_out

Create a user friendly, punchy and nice \"what's new\" for this release. Format it so that it will look nice
in a Telegram chat: the bot sends HTML, so use <b>…</b> and <i>…</i> only, blank lines between items, an emoji
or two where it helps, nothing else (no markdown, no headers, no code spans, no tables). Do not open with a
title or the version: the message that carries the note already says which release it is.

The readers are not developers. Describe what they will notice — what Keep can now do, what got better, what
was fixed and why — never how it was built. Leave out refactors, tests, tooling, docs and anything with no visible
effect (if appropriate add a line about general performance/bug/stability fixes); if nothing visible changed, say so
in one friendly line. Group related commits into one item. Output only the note, nothing before or after it.$guided

Commits:
$log"

  claude -p "$prompt" --allowedTools 'Bash(git show:*) Bash(git log:*) Bash(git diff:*) Read Grep' < /dev/null \
    || echo "claude failed — the tag gets no what's new." >&2
}

dry_run=false
urgent=""
ignore=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --urgent=*) urgent="${1#*=}"; [ -n "$urgent" ] || { echo "--urgent needs the reason, in one line." >&2; exit 1; }; shift ;;
    --urgent)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "--urgent needs the reason, in one line." >&2; exit 1; }
      urgent="$2"; shift 2 ;;
    --ignore-commits=*) list="${1#*=}"; read -ra hashes <<< "${list//,/ }"; ignore+=(${hashes[@]+"${hashes[@]}"}); shift ;;
    --ignore-commits)
      [ $# -ge 2 ] || { echo "--ignore-commits needs a list of commit hashes." >&2; exit 1; }
      read -ra hashes <<< "${2//,/ }"; ignore+=(${hashes[@]+"${hashes[@]}"}); shift 2 ;;
    --*) echo "Unknown option $1." >&2; exit 1 ;;
    *) break ;;
  esac
done
urgent="${urgent//$'\n'/ }"
subject_tail=""
[ -n "$urgent" ] && subject_tail=" urgent: $urgent"

# The signers the installs check this release against: the file of the release they run now,
# or, for the first signed release, the file the release itself brings.
SIGNERS=res/release_signers
signers_now() {  # $1: the commit being released — prints the path of a file with the signers
  local f
  f=$(mktemp)
  git show "origin/release:$SIGNERS" > "$f" 2>/dev/null || git show "$1:$SIGNERS" > "$f" 2>/dev/null \
    || { rm -f "$f"; return 1; }
  echo "$f"
}

if ! $dry_run; then
  [ "$(git config gpg.format)" = ssh ] && [ -n "$(git config user.signingkey)" ] || {
    echo "Tag signing is not set up: run, in this checkout," >&2
    echo "  git config --local gpg.format ssh" >&2
    echo "  git config --local user.signingkey '<the public key of the release signing key>'" >&2
    echo "  git config --local gpg.ssh.program /Applications/1Password.app/Contents/MacOS/op-ssh-sign" >&2
    exit 1
  }
fi

git fetch -q origin
# The first word is the commit to release when it names one; every other word is guidance for the note.
target=origin/master
if [ $# -gt 0 ] && git rev-parse -q --verify "$1^{commit}" >/dev/null; then target="$1"; shift; fi
guidance="$*"
commit=$(git rev-parse --verify "$target^{commit}")
short=$(git rev-parse --short "$commit")
version="v$(git log -1 --format=%cd --date=format:%Y.%m "$commit").$(git rev-list --count "$commit")"

git merge-base --is-ancestor "$commit" origin/master \
  || { echo "$target is not on origin/master — push it first." >&2; exit 1; }

current=$(git rev-parse -q --verify origin/release 2>/dev/null || true)
if [ "$current" = "$commit" ]; then
  echo "release is already at $version ($short)"
  exit 0
fi
if [ -n "$current" ] && ! git merge-base --is-ancestor "$current" "$commit"; then
  echo "origin/release ($(git rev-parse --short "$current")) is not behind $target — refusing to move it backwards." >&2
  exit 1
fi

# The baseline the listing and the note start from: the newest release tag from before today
# that is behind the target. The day's first release has none newer than origin/release itself.
baseline="$current"
today=$(date +%Y-%m-%d)
while read -r day tag; do
  [[ "$day" < "$today" ]] || continue
  tagged=$(git rev-parse -q --verify "refs/tags/$tag^{commit}") || continue
  if git merge-base --is-ancestor "$tagged" "$commit"; then baseline="$tagged"; break; fi
done < <(git for-each-ref --sort=-taggerdate 'refs/tags/v*' --format='%(taggerdate:format-local:%Y-%m-%d) %(refname:short)')

# An ignored commit has to be one the note would otherwise cover: anything else is a typo.
for h in ${ignore[@]+"${ignore[@]}"}; do
  full=$(git rev-parse -q --verify "$h^{commit}") || { echo "--ignore-commits: $h is not a commit." >&2; exit 1; }
  if ! git merge-base --is-ancestor "$full" "$commit" || { [ -n "$baseline" ] && git merge-base --is-ancestor "$full" "$baseline"; }; then
    echo "--ignore-commits: $h is not among the commits this release brings." >&2; exit 1
  fi
  ignored+=("$full")
done

echo "Releasing $version ($short)"
[ -n "$urgent" ] && echo "  URGENT — installs update within the hour: $urgent"
if [ -n "$baseline" ]; then
  [ "$baseline" != "$current" ] && echo "  (since $(git describe --tags --exact-match "$baseline" 2>/dev/null || git rev-parse --short "$baseline"), the last release before today — the ones released earlier today are included)"
  while read -r c; do
    line=$(git log -1 --oneline "$c")
    is_ignored "$c" && line="$line   (left out of the what's new)"
    echo "  $line"
  done < <(git rev-list "$baseline..$commit")
fi

existing=$(git rev-parse -q --verify "refs/tags/$version^{commit}" || true)
if [ -n "$existing" ] && [ "$existing" != "$commit" ]; then
  echo "Tag $version already names $(git rev-parse --short "$existing")." >&2; exit 1
fi

note=""
if [ -n "$existing" ]; then
  if [ -n "$urgent" ] && [ "$(git tag -l --format='%(contents:subject)' "$version")" != "Release $version$subject_tail" ]; then
    echo "Tag $version exists already and does not say this — delete it (git tag -d $version) and release again." >&2
    exit 1
  fi
  note=$(git tag -l --format='%(contents:body)' "$version")
  echo "Tag $version exists already; keeping its what's new."
elif [ -n "$baseline" ]; then
  note=$(whats_new "$baseline" "$commit")
fi
echo; echo "What's new in $version:"; echo; echo "${note:-(none)}"; echo

if $dry_run; then
  echo "Dry run: not tagging or pushing."
  exit 0
fi
if [ -t 0 ]; then
  read -r -p "Tag $version with this note and push the ${urgent:+URGENT }release? [y/N] " answer
  [[ "$answer" =~ ^[Yy] ]] || { echo "Not released."; exit 1; }
fi
if [ -z "$existing" ]; then
  printf 'Release %s%s\n\n%s\n' "$version" "$subject_tail" "$note" | git tag -s "$version" -F - "$commit"
fi
signers=$(signers_now "$commit") || { echo "Neither the current release nor $short carries $SIGNERS." >&2; exit 1; }
if ! git -c gpg.ssh.allowedSignersFile="$signers" verify-tag "$version"; then
  rm -f "$signers"
  echo "$version does not verify against $SIGNERS — installs would refuse it, so it is not pushed." >&2
  echo "Delete the local tag (git tag -d $version) and release again with the right key." >&2
  exit 1
fi
rm -f "$signers"
git push origin "$commit:refs/heads/release" "refs/tags/$version"
if [ -n "$urgent" ]; then
  echo "Done: release is $version, urgent — installs pick it up at their next hourly check."
else
  echo "Done: release is $version — installs pick it up on their next restart_server."
fi
