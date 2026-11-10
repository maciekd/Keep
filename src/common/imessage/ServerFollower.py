import re
import threading
import time
from typing import Callable, Optional

import common.DevelopmentConfig as DevelopmentConfig
from common import Version
from common.imessage import Wire
from common.imessage.MessagesHost import SatelliteStatus
from common.SignedRelease import RELEASE_BRANCH
from common.Updater import Updater
from common.logging import error, info, note

FOLLOWED_BRANCHES = (RELEASE_BRANCH, "master")
RETRY_SECONDS = 3600        # how long a target that could not be reached is left alone
# Between two tries whatever they aim at: a caller that names another commit with every request
# would otherwise get a `git pull` for each.
MIN_INTERVAL_SECONDS = 120
START_SCRIPT = "start.sh"
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")


def _on_a_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="satellite-update", daemon=True).start()


class ServerFollower:
    """Keeps an iMessage satellite on the code its server runs.

    The satellite has no nightly clock: every request names the branch and commit of the server
    that sent it (`notice`), and when they are not the ones this process started on, it updates
    through `Updater.update` and relaunches itself through start.sh. The rules: only `release`
    and `master` are followed (a server trying a pull request moves nothing); one update at a
    time, on a thread of its own — pip can take minutes and the request that brought the news
    is answered first; a target that could not be reached is left alone for `RETRY_SECONDS`
    instead of being pulled at every poll, and no two tries are closer than
    `MIN_INTERVAL_SECONDS` whatever they aim at; a satellite on the server's branch whose commit
    already contains the server's pulls nothing — the server is the one to move — while one on
    the other branch moves to the server's whichever is ahead: `master` is the unsigned line,
    and a server gone to `release` takes its satellite there; a failed update leaves this
    process serving and is told in `status`, which travels in the satellite's health answer."""

    def __init__(self, branch: str, commit: str, updater: Optional[Updater] = None,
                 relaunch: Callable[..., None] = Updater.relaunch,
                 spawn: Callable[[Callable[[], None]], None] = _on_a_thread,
                 clock: Callable[[], float] = time.monotonic):
        self.branch = branch                # what the checkout follows: the start's, or the server's after a switch
        self.commit = commit                # what this process started on
        self._updater = updater or Updater()
        self._relaunch = relaunch
        self._spawn = spawn
        self._clock = clock
        self._lock = threading.Lock()
        self._updating = False
        self._failure = ""
        self._tried: Optional[tuple[str, str]] = None      # the target of the last try, until the server names another
        self._tried_at: Optional[float] = None              # when the last try began, whatever its target
        self._seen: Optional[tuple[str, str]] = None        # the last caller, so its news is logged once
        if not commit:
            self._failure = ("this install cannot tell which commit it runs (git answered nothing at its start), "
                             "so it does not follow the server — start it again from its Keep folder")
            error(f"Satellite: {self._failure}")
        else:
            info(f"Satellite: on {branch or 'a detached checkout'} at {commit[:8]}; follows a server on "
                 f"{' or '.join(FOLLOWED_BRANCHES)}")

    def status(self) -> SatelliteStatus:
        with self._lock:
            return SatelliteStatus(Wire.WIRE_VERSION, Version.detail(), self.branch, self.commit,
                                   self._updating, self._failure)

    def notice(self, branch: str, commit: str) -> None:
        """A request came from a server on this branch and commit. Returns at once: an update,
        when one is due, runs on its own thread."""
        if not _COMMIT_RE.match(commit or "") or not branch:
            return      # a caller that does not say what it runs, or says it in no shape git has
        if not self.commit:
            return
        target = (branch, commit)
        news = target != self._seen
        self._seen = target
        with self._lock:
            if self._tried not in (None, target) and not self._updating:
                self._failure, self._tried = "", None   # what the last try came to was about another target
        if target == (self.branch, self.commit):
            if news:
                info(f"Satellite: the server is on {branch} at {commit[:8]}, as this process is — nothing to follow")
            return
        if branch not in FOLLOWED_BRANCHES:
            if news:
                info(f"Satellite: the server runs a branch that is not followed ({branch[:60]!r} at {commit[:8]}) — "
                     f"staying on {self.branch} at {self.commit[:8]}")
            return
        if DevelopmentConfig.LOCAL_DEVELOPMENT_MODE:
            if news:
                info(f"Satellite: LOCAL_DEVELOPMENT_MODE is on, not following the server to {branch} at {commit[:8]}")
            return
        with self._lock:
            if self._updating:
                return
            since = None if self._tried_at is None else self._clock() - self._tried_at
            if since is not None and self._tried == target and since < RETRY_SECONDS:
                return
            if since is not None and since < MIN_INTERVAL_SECONDS:
                if news:
                    info(f"Satellite: the server now runs {branch} at {commit[:8]}, but the last try at following "
                         f"it began {since:.0f}s ago — waiting until {MIN_INTERVAL_SECONDS}s have passed")
                return
            self._updating, self._tried, self._tried_at = True, target, self._clock()
        note(f"Satellite: the server runs {branch} at {commit[:8]}, this process {self.branch or 'no branch'} at "
             f"{self.commit[:8]} — following it")
        self._spawn(lambda: self._update(branch, commit))

    def _ahead_of(self, commit: str) -> bool:
        """Whether this process's commit already contains the server's and more. False when git
        cannot tell (a commit this checkout has never fetched counts nothing): the pull then
        goes ahead, as it does for a server that is the newer one."""
        return (self._updater.commits_between(commit, self.commit) > 0
                and self._updater.commits_between(self.commit, commit) == 0)

    def _follows(self, branch: str) -> None:
        """The pull went through, so the checkout is on the server's branch now: say so in the status."""
        if branch == self.branch:
            return
        info(f"Satellite: the checkout moved from {self.branch or 'no branch'} to {branch}")
        with self._lock:
            self.branch = branch
        Version.branch.cache_clear()
        Version.detail.cache_clear()

    def _update(self, branch: str, commit: str) -> None:
        failure, restart = "", False
        try:
            if branch == self.branch and self._ahead_of(commit):
                # pulling would only take this Mac further from a server that has yet to update
                info(f"Satellite: this process runs {self.commit[:8]}, which holds the server's {commit[:8]} and "
                     f"more — nothing is pulled until the server updates")
            else:
                failure, restart = self._pull(branch, commit)
        except Exception as e:
            failure = f"{type(e).__name__}: {e}"
        with self._lock:
            self._updating, self._failure = False, failure
        if failure:
            error(f"Satellite: could not follow the server to {branch} at {commit[:8]}, still serving "
                  f"{self.commit[:8]} — {failure} (next try in {RETRY_SECONDS // 60} min or when the server moves)")
        elif restart:
            note(f"Satellite: updated to {branch} at {commit[:8]}, {START_SCRIPT} is replacing this process")

    def _pull(self, branch: str, commit: str) -> tuple[str, bool]:
        """One `Updater.update` towards the server: (why it left this process where it was, whether
        the start script is replacing it). Neither when there was nothing to do."""
        result = self._updater.update(branch, self.commit)
        if result.head and not result.pull_failure:
            self._follows(branch)
        if result.failure:
            return result.failure, False
        if result.restart:
            self._relaunch(START_SCRIPT, f"following the server to {branch} at {commit[:8]}: {result.change}")
            return "", True
        if not result.head:
            return "git could not say where the checkout is after the pull", False
        if result.head == commit:
            info(f"Satellite: on {branch} at {commit[:8]} now, the commit this process already runs — "
                 f"the server is followed without a restart")
            return "", False
        if self._updater.commits_between(commit, result.head) > 0:
            # the newer of the two has nowhere to go: the server's own update settles it
            info(f"Satellite: {branch} is at {result.head[:8]} here, ahead of the server's {commit[:8]} — "
                 f"nothing to follow until the server updates")
            return "", False
        return (f"nothing to pull: {branch} is at {result.head[:8]} here, the server runs {commit[:8]} — "
                f"{'a release this Mac cannot verify yet' if branch == RELEASE_BRANCH else 'a commit that is not pushed'}, "
                f"or this checkout has changes of its own"), False
