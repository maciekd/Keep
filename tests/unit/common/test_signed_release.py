"""SignedRelease on real throwaway repositories: a bare origin whose `release` tip carries a
tag signed (or not) by a throwaway ed25519 key, and an install's clone one release behind. The
install trusts the keys in its own res/release_signers — the maintainer's — and nothing a
fetched release brings, so a push to `release` signed by anyone else never moves it."""
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from common import SignedRelease as module
from common.SignedRelease import SignedRelease

from ..git_sandbox import History, empty_repo, git, signing_key, spawn

SIGNERS = 'res/release_signers'


class Install:
    """Commit 1 is the release the install runs, commit 2 the one pushed to origin's `release`
    since — or, when `current`, the one the install runs already; `tag` puts the v-tag on
    commit 2 in origin."""

    def __init__(self, tmp_path: Path, next_signers: str | None = None, on: str = 'release',
                 current: bool = False):
        maintainer = signing_key('maintainer')[1]
        history = History()
        history.commit(datetime(2026, 10, 1, 12, tzinfo=timezone.utc), 'feat: one',
                       {'readme.md': 'Keep 1\n', SIGNERS: maintainer})
        history.branch('release', 1)
        self.origin = empty_repo(tmp_path / 'origin.git', bare=True)
        self.clone = empty_repo(tmp_path / 'install', origin=self.origin)
        if not current:
            self.running = history.write(self.clone)[0]
        history.commit(datetime(2026, 10, 2, 12, tzinfo=timezone.utc), 'feat: two',
                       {'readme.md': 'Keep 2\n', SIGNERS: next_signers or maintainer})
        history.branch('release', 2)
        self.pushed = history.write(self.origin)[1]
        if current:
            self.running = history.write(self.clone)[1]
        (self.clone / '.git' / 'HEAD').write_text(f'ref: refs/heads/{on}\n')
        git(self.clone, 'reset', '-q', '--hard')

    def tag(self, key: str | None, name: str = 'v2026.10.2'):
        """The release tag on origin's tip: signed with the named throwaway key, or plain annotated."""
        if key is None:
            git(self.origin, 'tag', '-a', name, '-m', f'Release {name}', self.pushed)
        else:
            git(self.origin, '-c', 'gpg.format=ssh', '-c', f'user.signingkey={signing_key(key)[0]}',
                'tag', '-s', name, '-m', f'Release {name}', self.pushed)

    def head(self) -> str:
        return git(self.clone, 'rev-parse', 'HEAD')

    def branch(self) -> str:
        return git(self.clone, 'rev-parse', '--abbrev-ref', 'HEAD')


def _spawned(command: list, cwd: Path, timeout: int, **_) -> subprocess.CompletedProcess:
    """The module's `subprocess.run`, through the sandbox's `spawn`: the same git in the same
    directory (`-C` for the `cwd`), but posix_spawned — a `cwd` makes Python fork the worker
    instead, and forking one with the tree imported costs more than the git it starts."""
    return spawn([command[0], '-C', str(cwd), *command[1:]], timeout=timeout)


@pytest.fixture
def install(tmp_path, monkeypatch):
    """The install's clone is the repository SignedRelease works on, its git the sandbox's (no
    user or system config, the cast's identity)."""
    def make(**kwargs) -> Install:
        made = Install(tmp_path, **kwargs)
        monkeypatch.setattr(module, 'REPO_DIR', made.clone)
        monkeypatch.setattr(module, 'SIGNERS_FILE', made.clone / SIGNERS)
        return made
    monkeypatch.setattr(module, 'subprocess', SimpleNamespace(run=_spawned))
    return make


class TestVerifiedRelease:
    def test_a_tip_signed_by_a_listed_key_fast_forwards_the_install(self, install):
        repo = install()
        repo.tag('maintainer')
        failure, output = SignedRelease.update('release')
        assert failure == ''
        assert repo.head() == repo.pushed
        assert 'v2026.10.2' in output

    def test_an_install_already_at_the_signed_tip_reports_it_is_up_to_date(self, install):
        repo = install(current=True)
        repo.tag('maintainer')
        assert SignedRelease.update('release') == ('', 'Already up to date.'), \
            "pull_and_restart reads this phrase to tell 'no new changes'"

    def test_an_install_on_master_is_switched_to_release_and_brought_to_the_signed_tip(self, install):
        repo = install(on='master')
        repo.tag('maintainer')
        failure, _ = SignedRelease.update('master')
        assert failure == ''
        assert repo.branch() == 'release'
        assert repo.head() == repo.pushed


class TestRefusedRelease:
    def test_an_unsigned_tip_is_refused_and_the_install_stays_where_it_is(self, install):
        repo = install()
        repo.tag(None)
        failure, _ = SignedRelease.update('release')
        assert failure
        assert repo.head() == repo.running

    def test_a_tip_with_no_version_tag_at_all_is_refused(self, install):
        repo = install()
        failure, _ = SignedRelease.update('release')
        assert failure
        assert repo.head() == repo.running

    def test_a_tip_signed_by_a_key_the_install_does_not_list_is_refused(self, install):
        repo = install()
        repo.tag('stranger')
        failure, _ = SignedRelease.update('release')
        assert failure
        assert repo.head() == repo.running

    def test_a_release_that_lists_its_own_signing_key_is_still_refused(self, install):
        """The signers file that counts is the running code's: a release cannot vouch for itself."""
        repo = install(next_signers=signing_key('stranger')[1])
        repo.tag('stranger')
        failure, _ = SignedRelease.update('release')
        assert failure
        assert repo.head() == repo.running
        assert (repo.clone / SIGNERS).read_text() == signing_key('maintainer')[1]

    def test_a_refused_release_never_switches_an_install_off_its_branch(self, install):
        repo = install(on='master')
        repo.tag('stranger')
        failure, _ = SignedRelease.update('master')
        assert failure
        assert repo.branch() == 'master'

    def test_without_a_signers_file_nothing_is_fetched_or_moved(self, install, monkeypatch):
        repo = install()
        repo.tag('maintainer')
        monkeypatch.setattr(module, 'SIGNERS_FILE', repo.clone / 'res' / 'missing')
        failure, _ = SignedRelease.update('release')
        assert failure
        assert repo.head() == repo.running
        assert git(repo.clone, 'rev-parse', '--verify', '-q', 'origin/release', check=False) == ''
