"""Tests for release.sh, the whole release mechanism: origin's `release` branch is
fast-forwarded to a master commit and that commit is tagged with its vYYYY.MM.N version
(common/Version.py's scheme, so an install on `release` reports the version master had).

Real git against a throwaway origin in tmp_path: the script is copied into a clone of
it, since it works on the checkout it lives in.
"""
import os
import subprocess
from pathlib import Path

import pytest

RELEASE_SH = Path(__file__).resolve().parents[2] / 'release.sh'
# A fixed commit clock makes the version derivable: commits dated March 2026 in a linear
# history give v2026.03.<commit count>.
GIT_ENV = {
    'GIT_AUTHOR_NAME': 'Marcus', 'GIT_AUTHOR_EMAIL': 'm@mlab.test',
    'GIT_COMMITTER_NAME': 'Marcus', 'GIT_COMMITTER_EMAIL': 'm@mlab.test',
    'GIT_AUTHOR_DATE': '2026-03-05T10:00:00+01:00', 'GIT_COMMITTER_DATE': '2026-03-05T10:00:00+01:00',
}


def git(path: Path, *args: str) -> str:
    env = {**os.environ, **GIT_ENV, 'GIT_CONFIG_GLOBAL': os.devnull}
    return subprocess.run(['git', *args], cwd=path, env=env,
                          capture_output=True, text=True, check=True).stdout.strip()


def commit(path: Path, message: str) -> str:
    (path / 'readme.md').write_text(f'{message}\n')
    git(path, 'add', '-A')
    git(path, 'commit', '-qm', message)
    return git(path, 'rev-parse', 'HEAD')


class Repos:
    """`origin` with a linear master, and `clone` — the checkout release.sh runs from."""

    def __init__(self, tmp_path: Path, commits: int):
        self.origin = tmp_path / 'origin'
        self.origin.mkdir()
        git(self.origin, 'init', '-q', '-b', 'master')
        self.commits = [commit(self.origin, f'feat: {n}') for n in range(1, commits + 1)]
        self.clone = tmp_path / 'clone'
        git(tmp_path, 'clone', '-q', str(self.origin), str(self.clone))
        self.script = self.clone / 'release.sh'
        self.script.write_text(RELEASE_SH.read_text())

    def release(self, *args: str) -> subprocess.CompletedProcess:
        env = {**os.environ, **GIT_ENV, 'GIT_CONFIG_GLOBAL': os.devnull}
        return subprocess.run(['bash', str(self.script), *args], cwd=self.clone, env=env,
                              capture_output=True, text=True, timeout=30)

    def origin_ref(self, ref: str) -> str | None:
        result = subprocess.run(['git', 'rev-parse', '-q', '--verify', f'{ref}^{{commit}}'],
                                cwd=self.origin, capture_output=True, text=True)
        return result.stdout.strip() or None

    def origin_tags(self) -> list[str]:
        return git(self.origin, 'tag', '--list').split()


@pytest.mark.unit
class TestReleaseSh:
    def test_first_release_moves_the_branch_to_master_and_tags_its_version(self, tmp_path):
        repos = Repos(tmp_path, commits=3)
        result = repos.release()
        assert result.returncode == 0, result.stdout + result.stderr
        assert repos.origin_ref('release') == repos.commits[-1]
        assert repos.origin_tags() == ['v2026.03.3']
        assert repos.origin_ref('v2026.03.3') == repos.commits[-1]

    def test_a_current_release_is_left_alone(self, tmp_path):
        repos = Repos(tmp_path, commits=2)
        assert repos.release().returncode == 0
        again = repos.release()
        assert again.returncode == 0, again.stderr
        assert 'already' in again.stdout
        assert repos.origin_tags() == ['v2026.03.2'], "a second run must not add a tag"

    def test_a_later_master_advances_the_release_and_adds_a_tag(self, tmp_path):
        repos = Repos(tmp_path, commits=1)
        assert repos.release().returncode == 0
        newer = commit(repos.origin, 'feat: 2')
        result = repos.release()
        assert result.returncode == 0, result.stderr
        assert repos.origin_ref('release') == newer
        assert sorted(repos.origin_tags()) == ['v2026.03.1', 'v2026.03.2']

    def test_an_explicit_earlier_master_commit_can_be_released(self, tmp_path):
        repos = Repos(tmp_path, commits=3)
        result = repos.release(repos.commits[1])
        assert result.returncode == 0, result.stderr
        assert repos.origin_ref('release') == repos.commits[1]
        assert repos.origin_tags() == ['v2026.03.2']

    def test_an_unpushed_commit_is_refused(self, tmp_path):
        repos = Repos(tmp_path, commits=1)
        local_only = commit(repos.clone, 'feat: local')
        result = repos.release(local_only)
        assert result.returncode == 1
        assert repos.origin_ref('release') is None
        assert repos.origin_tags() == []

    def test_moving_the_release_backwards_is_refused(self, tmp_path):
        repos = Repos(tmp_path, commits=3)
        assert repos.release().returncode == 0
        result = repos.release(repos.commits[0])
        assert result.returncode == 1
        assert 'backwards' in result.stderr
        assert repos.origin_ref('release') == repos.commits[-1]
        assert repos.origin_tags() == ['v2026.03.3']

    def test_a_version_tag_on_another_commit_stops_the_release(self, tmp_path):
        """The count-based version names one master commit; a tag that says otherwise
        means the history is not what the scheme assumes, so nothing is published."""
        repos = Repos(tmp_path, commits=2)
        git(repos.clone, 'tag', 'v2026.03.2', repos.commits[0])
        result = repos.release()
        assert result.returncode == 1
        assert repos.origin_ref('release') is None
