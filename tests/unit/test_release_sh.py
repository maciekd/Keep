"""Tests for release.sh, the whole release mechanism: it fast-forwards origin's `release`
branch to a master commit and tags that commit with its vYYYY.MM.N version. Installs pull
`release`, so the script must refuse anything that would hand them a commit master never
had (unpushed work) or move them backwards, and must be a no-op when already current.

Each test builds a bare origin and a development clone in tmp_path and runs a copy of the
script from inside the clone (it cds to its own directory), so nothing touches this
repository's remote. Commits carry fixed dates, so the expected version is known up front:
the month of the released commit and how many commits lead up to it.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

RELEASE_SH = Path(__file__).resolve().parents[2] / 'release.sh'
IDENTITY = ['-c', 'user.name=Marcus', '-c', 'user.email=m@mlab.test']


def git(cwd: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(['git', *IDENTITY, *args], cwd=cwd, capture_output=True, text=True,
                          check=check).stdout.strip()


class Repos:
    """A bare origin with a `master` branch and a clone that pushes to it."""

    def __init__(self, tmp_path: Path):
        self.origin = tmp_path / 'origin.git'
        self.dev = tmp_path / 'dev'
        subprocess.run(['git', 'init', '-q', '--bare', '-b', 'master', self.origin], check=True)
        subprocess.run(['git', 'clone', '-q', self.origin, self.dev], check=True,
                       capture_output=True)
        shutil.copy(RELEASE_SH, self.dev / 'release.sh')
        self.commits: list[str] = []

    def commit(self, month: str, push: bool = True) -> str:
        """One more commit on master, dated in `month` (YYYY.MM); returns its full hash."""
        n = len(self.commits) + 1
        (self.dev / 'readme.md').write_text(f'Keep, edit {n}\n')
        date = f'{month.replace(".", "-")}-{n:02d}T12:00:00'
        env = {'GIT_AUTHOR_DATE': date, 'GIT_COMMITTER_DATE': date}
        subprocess.run(['git', *IDENTITY, 'add', '-A'], cwd=self.dev, check=True)
        subprocess.run(['git', *IDENTITY, 'commit', '-qm', f'feat: edit {n}'], cwd=self.dev,
                       check=True, env={**os.environ, **env})
        if push:
            git(self.dev, 'push', '-q', 'origin', 'master')
        self.commits.append(git(self.dev, 'rev-parse', 'HEAD'))
        return self.commits[-1]

    def release(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(['bash', './release.sh', *args], cwd=self.dev, capture_output=True,
                              text=True, timeout=30)

    def origin_release(self) -> str | None:
        return git(self.origin, 'rev-parse', '-q', '--verify', 'refs/heads/release', check=False) or None

    def origin_tags(self) -> dict[str, str]:
        out = git(self.origin, 'for-each-ref', '--format=%(refname:short) %(*objectname)', 'refs/tags')
        return dict(line.split() for line in out.splitlines())


@pytest.fixture
def repos(tmp_path):
    return Repos(tmp_path)


class TestFirstRelease:
    def test_moves_release_to_master_and_tags_it_with_the_commit_month_and_count(self, repos):
        repos.commit('2026.08')
        repos.commit('2026.09')
        tip = repos.commit('2026.09')
        result = repos.release()
        assert result.returncode == 0, result.stderr
        assert repos.origin_release() == tip
        assert repos.origin_tags() == {'v2026.09.3': tip}

    def test_an_earlier_master_commit_can_be_released_instead_of_the_tip(self, repos):
        repos.commit('2026.09')
        second = repos.commit('2026.09')
        repos.commit('2026.09')
        result = repos.release(second)
        assert result.returncode == 0, result.stderr
        assert repos.origin_release() == second
        assert repos.origin_tags() == {'v2026.09.2': second}


class TestRefusals:
    def test_work_not_yet_on_origin_master_is_refused(self, repos):
        repos.commit('2026.09')
        local = repos.commit('2026.09', push=False)
        result = repos.release(local)
        assert result.returncode != 0
        assert 'push' in result.stderr
        assert repos.origin_release() is None
        assert repos.origin_tags() == {}

    def test_release_is_never_moved_backwards(self, repos):
        first = repos.commit('2026.09')
        second = repos.commit('2026.09')
        assert repos.release().returncode == 0
        result = repos.release(first)
        assert result.returncode != 0
        assert 'backwards' in result.stderr
        assert repos.origin_release() == second

    def test_a_version_tag_already_naming_another_commit_stops_the_release(self, repos):
        """Two commits with the same month and ancestor count share a version string; the
        tag must not silently be left pointing at the other one while release moves on."""
        repos.commit('2026.09')
        tip = repos.commit('2026.09')
        stray = git(repos.dev, 'commit-tree', 'HEAD^{tree}', '-m', 'stray')
        git(repos.dev, 'tag', '-a', 'v2026.09.2', '-m', 'wrong', stray)
        result = repos.release()
        assert result.returncode != 0
        assert 'v2026.09.2' in result.stderr
        assert repos.origin_release() is None
        assert tip not in repos.origin_tags().values()


class TestRepeatedRelease:
    def test_releasing_the_current_release_again_changes_nothing(self, repos):
        tip = repos.commit('2026.09')
        assert repos.release().returncode == 0
        result = repos.release()
        assert result.returncode == 0, result.stderr
        assert 'already' in result.stdout
        assert repos.origin_release() == tip
        assert repos.origin_tags() == {'v2026.09.1': tip}

    def test_a_later_release_keeps_the_earlier_tag_and_adds_the_new_one(self, repos):
        first = repos.commit('2026.09')
        assert repos.release().returncode == 0
        second = repos.commit('2026.10')
        result = repos.release()
        assert result.returncode == 0, result.stderr
        assert repos.origin_release() == second
        assert repos.origin_tags() == {'v2026.09.1': first, 'v2026.10.2': second}
