"""Tests for release.sh, the whole release mechanism: it fast-forwards origin's `release`
branch to a master commit and tags that commit with its vYYYY.MM.N version. Installs pull
`release`, so the script must refuse anything that would hand them a commit master never
had (unpushed work) or move them backwards, and must be a no-op when already current.

Each test builds a bare origin and a development clone in tmp_path and runs a copy of the
script from inside the clone (it cds to its own directory), so nothing touches this
repository's remote. Commits carry fixed dates, so the expected version is known up front:
the month of the released commit and how many commits lead up to it. The repositories come
from git_sandbox, written without a git process — the script's own dozen are the test's
whole budget — and an earlier release is origin's `release` branch and tag put there the
same way, not a first run of the script. For the same reason no test here lets the script
get as far as its `git push`: the run that pushes costs more than a unit test may.
"""
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from .git_sandbox import History, empty_repo, git, git_env, spawn

RELEASE_SH = Path(__file__).resolve().parents[2] / 'release.sh'


class Repos:
    """A bare origin with a `master` branch and the clone that releases to it. `commits`
    only extends the history; `push` gives origin all of it so far, `keep_local` the clone."""

    def __init__(self, tmp_path: Path):
        self.origin = empty_repo(tmp_path / 'origin.git', bare=True)
        self.dev = empty_repo(tmp_path / 'dev', origin=self.origin)
        shutil.copy(RELEASE_SH, self.dev / 'release.sh')
        self.history = History()

    def commits(self, *months: str):
        """One more commit on master per month (YYYY.MM) given, each dated in its month."""
        for month in months:
            year, number = month.split('.')
            n = self.history.count + 1
            self.history.commit(datetime(int(year), int(number), n, 12, tzinfo=timezone.utc),
                                f'feat: edit {n}', {'readme.md': f'Keep, edit {n}\n'})

    def released(self, version: str):
        """The newest commit is what an earlier release left on origin: `release` and its tag."""
        self.history.branch('release', self.history.count)
        self.history.tag(version, self.history.count, f'Release {version}')

    def push(self) -> list[str]:
        """The full hashes of the commits, oldest first."""
        return self.history.write(self.origin)

    def keep_local(self) -> list[str]:
        return self.history.write(self.dev)

    def release(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return spawn(['bash', str(self.dev / 'release.sh'), *args], timeout=30, env=env)

    def stub_claude(self, tmp_path: Path) -> dict:
        """An environment whose `claude` prints the prompt it was given, so the dry run's
        "what's new" shows what the script asked for."""
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        stub = bin_dir / 'claude'
        stub.write_text('#!/bin/sh\necho "$2"\n')
        stub.chmod(0o755)
        env = git_env()
        return {**env, 'PATH': str(bin_dir) + os.pathsep + env['PATH']}

    def origin_state(self) -> tuple[str | None, dict[str, str]]:
        """Where origin's `release` branch is (None without one) and the commit each tag names."""
        out = git(self.origin, 'for-each-ref', '--format=%(refname) %(objectname) %(*objectname)',
                  'refs/heads/release', 'refs/tags')
        refs = {ref: hashes for ref, *hashes in (line.split() for line in out.splitlines())}
        tags = {ref.removeprefix('refs/tags/'): hashes[-1] for ref, hashes in refs.items() if ref.startswith('refs/tags/')}
        return refs.get('refs/heads/release', [None])[0], tags


@pytest.fixture
def repos(tmp_path):
    return Repos(tmp_path)


class TestRefusals:
    def test_work_not_yet_on_origin_master_is_refused(self, repos):
        repos.commits('2026.09')
        repos.push()
        repos.commits('2026.09')
        local = repos.keep_local()[-1]
        result = repos.release(local)
        assert result.returncode != 0
        assert 'push' in result.stderr
        assert repos.origin_state() == (None, {})

    def test_release_is_never_moved_backwards(self, repos):
        repos.commits('2026.09', '2026.09')
        repos.released('v2026.09.2')
        first, second = repos.push()
        result = repos.release(first)
        assert result.returncode != 0
        assert 'backwards' in result.stderr
        assert repos.origin_state()[0] == second

    def test_a_version_tag_already_naming_another_commit_stops_the_release(self, repos):
        """Two commits with the same month and ancestor count share a version string; the
        tag must not silently be left pointing at the other one while release moves on."""
        repos.commits('2026.09', '2026.09')
        tip = repos.push()[-1]
        repos.history.tag('v2026.09.2', 1, 'wrong')     # in the clone only, and on the first commit
        repos.keep_local()
        result = repos.release()
        assert result.returncode != 0
        assert 'v2026.09.2' in result.stderr
        release, tags = repos.origin_state()
        assert release is None
        assert tip not in tags.values()


class TestRepeatedRelease:
    def test_releasing_the_current_release_again_changes_nothing(self, repos):
        repos.commits('2026.09')
        repos.released('v2026.09.1')
        tip, = repos.push()
        result = repos.release()
        assert result.returncode == 0, result.stderr
        assert 'already' in result.stdout
        assert repos.origin_state() == (tip, {'v2026.09.1': tip})

    def test_the_version_is_the_month_of_the_named_commit_and_the_commits_up_to_it(self, repos):
        """The vYYYY.MM.N of common/Version.py, read off the one run that prints it without
        pushing: the release of an earlier master commit that is the release already."""
        repos.commits('2026.08', '2026.09')
        repos.released('v2026.09.2')
        repos.commits('2026.10')
        second = repos.push()[1]
        result = repos.release(second)
        assert result.returncode == 0, result.stderr
        assert 'already at v2026.09.2 ' in result.stdout


class TestIgnoredCommits:
    """--ignore-commits keeps the named commits out of the note Claude writes, and out of
    nothing else: they are released, and the listing says they were left out."""

    def test_ignored_commits_are_listed_but_kept_out_of_the_prompt(self, repos, tmp_path):
        repos.commits('2026.08')
        repos.released('v2026.08.1')
        repos.commits('2026.09', '2026.09', '2026.09')
        _, second, third, fourth = repos.push()
        result = repos.release('--dry-run', '--ignore-commits', f'{second[:7]},{third[:7]}',
                               env=repos.stub_claude(tmp_path))
        assert result.returncode == 0, result.stderr
        listing, prompt = result.stdout.split("What's new in", 1)
        assert all(edit in listing for edit in ('edit 2', 'edit 3', 'edit 4'))
        assert listing.count('left out') == 2
        assert 'edit 4' in prompt
        assert 'edit 2' not in prompt and 'edit 3' not in prompt
        assert repos.origin_state()[0] != fourth

    def test_a_commit_the_release_does_not_bring_is_refused(self, repos):
        repos.commits('2026.08')
        repos.released('v2026.08.1')
        repos.commits('2026.09')
        first, _ = repos.push()
        result = repos.release('--dry-run', '--ignore-commits', first[:7])
        assert result.returncode != 0
        assert first[:7] in result.stderr
        result = repos.release('--dry-run', '--ignore-commits', 'zzzz')
        assert result.returncode != 0
        assert 'zzzz' in result.stderr


class TestReachBack:
    """Installs update nightly, so a household that gets the evening release never saw the
    midday one: the listing and the note start from the last release tagged on an earlier day,
    not from origin/release, and --ignore-commits reaches as far."""

    def _two_releases_today(self, repos) -> list[str]:
        """A release tagged in August, two more commits released earlier today, one commit since."""
        repos.commits('2026.08')
        repos.history.tag('v2026.08.1', 1, 'Release v2026.08.1', when=datetime(2026, 8, 1, 12, tzinfo=timezone.utc))
        repos.commits('2026.09', '2026.09')
        repos.history.branch('release', 3)
        repos.history.tag('v2026.09.3', 3, 'Release v2026.09.3', when=datetime.now(timezone.utc))
        repos.commits('2026.09')
        return repos.push()

    def test_the_days_second_release_covers_the_commits_of_its_first(self, repos, tmp_path):
        self._two_releases_today(repos)
        result = repos.release('--dry-run', env=repos.stub_claude(tmp_path))
        assert result.returncode == 0, result.stderr
        listing, prompt = result.stdout.split("What's new in", 1)
        assert 'v2026.08.1' in listing
        for text in (listing, prompt):
            assert all(edit in text for edit in ('edit 2', 'edit 3', 'edit 4'))
            assert 'edit 1' not in text

    def test_a_commit_of_the_days_first_release_may_be_left_out_of_the_note(self, repos, tmp_path):
        _, second, _, _ = self._two_releases_today(repos)
        result = repos.release('--dry-run', '--ignore-commits', second[:7], env=repos.stub_claude(tmp_path))
        assert result.returncode == 0, result.stderr
        listing, prompt = result.stdout.split("What's new in", 1)
        assert 'edit 2' in listing and 'left out' in listing
        assert 'edit 2' not in prompt and 'edit 3' in prompt
