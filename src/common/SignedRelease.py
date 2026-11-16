"""Moves an install to the newest release only when that release's tag is signed by a key in `res/release_signers`.

GitHub is only the delivery route: anyone holding a token that can push `release` (a stolen
login, a collaborator's machine, an agent running as the maintainer) can put a commit there,
but only the maintainer's signing key can sign the tag release.sh puts on it. The tip of
`origin/release` must carry a `v*` tag whose signature `git verify-tag` accepts against the
signers file of the code that is running now — read from the working tree before anything
is checked out, so a release cannot bring the key it is checked with. A tip without such a
tag is refused and the checkout stays where it is.

Only the `release` branch is verified: the development machine follows master, and a PR
trial runs a branch whose author was checked when it started — the release a trial takes in
overnight is verified here like any other (`fetch`).

A release can be marked urgent: `scripts/release.sh --urgent <reason>` writes the reason into
the subject of the tag (`Release v… urgent: <reason>`). The signature covers it, so only the
holder of the release key can ask every install to update within the hour instead of at
night (`urgent`)."""
import re
import subprocess

from common.Paths import REPO_DIR, RES_DIR
from common.logging import info, error, trace

SIGNERS_FILE = RES_DIR / "release_signers"
RELEASE_BRANCH = "release"
REMOTE_REF = f"origin/{RELEASE_BRANCH}"
GIT_TIMEOUT = 120
URGENT_SUBJECT = re.compile(r"^Release \S+ urgent: (.+)$")   # a contract with scripts/release.sh
URGENT_REASON_LIMIT = 200


def _git(*args: str, timeout: int = GIT_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, timeout=timeout)


class SignedRelease:

    @staticmethod
    def fetch() -> tuple[str, str, str]:
        """Fetch `release` and verify the tag on its tip, moving nothing. Returns (failure,
        commit, tag): *failure* is "" when *commit* is the newest release and *tag* its
        verified tag."""
        if not SIGNERS_FILE.exists():
            error(f"SignedRelease: {SIGNERS_FILE} is missing, no release can be verified")
            return f"No release can be verified: {SIGNERS_FILE.name} is missing.", "", ""
        signers = str(SIGNERS_FILE.resolve())

        fetch = _git("fetch", "--force", "--tags", "origin", RELEASE_BRANCH)
        if fetch.returncode != 0:
            error(f"SignedRelease: git fetch failed: {fetch.stderr.strip()}")
            return f"Git fetch failed: {fetch.stderr.strip()}", "", ""

        commit = _git("rev-parse", f"{REMOTE_REF}^{{commit}}").stdout.strip()
        if not commit:
            error(f"SignedRelease: {REMOTE_REF} cannot be resolved after the fetch")
            return f"{REMOTE_REF} cannot be resolved.", "", ""
        tag = SignedRelease._verified_tag(commit, signers)
        if not tag:
            return (f"The newest release ({commit[:8]}) carries no tag signed by a key in "
                    f"{SIGNERS_FILE.name} — not updating, the checkout stays where it is."), "", ""
        return "", commit, tag

    @staticmethod
    def urgent() -> tuple[str, str]:
        """Fetch `release` and look for a release marked urgent among the ones the checkout does
        not hold yet — every one of them, since a later ordinary release must not hide the
        urgent one before it. Returns (tag, reason) of the newest, ("", "") when there is none
        or the tip does not verify: a release an install would refuse to move to is no reason
        to hurry. A marker counts only on a tag whose own signature verifies."""
        failure, commit, tip_tag = SignedRelease.fetch()
        if failure:
            info(f"SignedRelease: no urgent release looked for — {failure}")
            return "", ""
        signers = str(SIGNERS_FILE.resolve())
        missed = _git("tag", "--list", "v*", "--sort=-creatordate", "--merged", commit, "--no-merged", "HEAD")
        if missed.returncode != 0:
            error(f"SignedRelease: the releases since HEAD could not be listed: {missed.stderr.strip()}")
            return "", ""
        tags = missed.stdout.split()
        if not tags:
            trace(f"SignedRelease: the checkout holds {tip_tag} already, nothing urgent to look for")
            return "", ""
        for tag in tags:
            subject = _git("tag", "--list", "--format=%(contents:subject)", tag).stdout.strip()
            marked = URGENT_SUBJECT.match(subject)
            if not marked:
                continue
            if tag != tip_tag and _git("-c", f"gpg.ssh.allowedSignersFile={signers}", "verify-tag", tag).returncode != 0:
                error(f"SignedRelease: {tag} says it is urgent but its signature does not verify, ignored")
                continue
            reason = marked.group(1).strip()[:URGENT_REASON_LIMIT]
            info(f"SignedRelease: {tag} is an urgent release the checkout does not hold — {reason}")
            return tag, reason
        info(f"SignedRelease: {len(tags)} release(s) since HEAD up to {tip_tag}, none urgent")
        return "", ""

    @staticmethod
    def update(current_branch: str) -> tuple[str, str]:
        """Fetch `release`, verify the tag on its tip, check the branch out when the checkout is
        on another one and fast-forward it to the verified commit. Returns (failure, output) as
        a pull does: *failure* is "" when the checkout is at the verified release, *output* says
        "Already up to date." when nothing moved."""
        failure, commit, tag = SignedRelease.fetch()
        if failure:
            return failure, ""

        if current_branch != RELEASE_BRANCH:
            info(f"SignedRelease: on {current_branch}, switching to {RELEASE_BRANCH}")
            checkout = _git("checkout", RELEASE_BRANCH)
            if checkout.returncode != 0:
                error(f"SignedRelease: git checkout {RELEASE_BRANCH} failed: {checkout.stderr.strip()}")
                return (f"Could not switch to {RELEASE_BRANCH}: {checkout.stderr.strip()} — "
                        f"still on {current_branch}."), ""

        head = _git("rev-parse", "HEAD").stdout.strip()
        if head == commit:
            info(f"SignedRelease: already at {tag} ({commit[:8]})")
            return "", "Already up to date."
        merge = _git("merge", "--ff-only", commit)
        if merge.returncode != 0:
            error(f"SignedRelease: fast-forward to {tag} failed: {merge.stderr.strip()}")
            return f"Could not move to {tag}: {merge.stderr.strip()}", ""
        info(f"SignedRelease: moved {head[:8]} → {commit[:8]}, {tag} with a verified signature")
        return "", f"Updated {head[:8]}..{commit[:8]} to {tag} (signature verified)."

    @staticmethod
    def _verified_tag(commit: str, signers: str) -> str:
        """The `v*` tag on *commit* whose signature verifies against *signers*; "" when none does."""
        tags = _git("tag", "--points-at", commit, "--list", "v*").stdout.split()
        if not tags:
            error(f"SignedRelease: the tip of {REMOTE_REF} ({commit[:8]}) carries no v* tag")
            return ""
        for tag in tags:
            result = _git("-c", f"gpg.ssh.allowedSignersFile={signers}", "verify-tag", tag)
            if result.returncode == 0:
                trace(f"SignedRelease: {tag} verified — {result.stderr.strip()}")
                return tag
            error(f"SignedRelease: {tag} on {commit[:8]} failed verification: {result.stderr.strip()}")
        return ""
