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

from .git_sandbox import History, _program, empty_repo, git, git_env, signing_key, spawn

RELEASE_SH = Path(__file__).resolve().parents[2] / 'release.sh'


class Repos:
    """A bare origin with a `master` branch and the clone that releases to it. `commits`
    only extends the history; `push` gives origin all of it so far (and the clone, which
    made it), `keep_local` the clone alone."""

    def __init__(self, tmp_path: Path):
        self.origin = empty_repo(tmp_path / 'origin.git', bare=True)
        self.dev = empty_repo(tmp_path / 'dev', origin=self.origin)
        self.sign_with('maintainer')
        shutil.copy(RELEASE_SH, self.dev / 'release.sh')
        self.history = History()
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        (bin_dir / 'claude').symlink_to('/bin/echo')
        env = git_env()
        self.env = {**env, 'PATH': str(bin_dir) + os.pathsep + env['PATH']}
        git_dir = str(Path(_program('git')).parent)
        self.no_claude_path = os.pathsep.join((git_dir, '/usr/bin', '/bin'))

    def sign_with(self, key: str | None):
        """The clone's tag signing as the maintainer sets it up (`git config --local`), with the
        throwaway key of that name; None leaves signing unconfigured."""
        config = self.dev / '.git' / 'config'
        text = config.read_text().split('[gpg]')[0]
        if key is not None:
            text += f'[gpg]\n\tformat = ssh\n[user]\n\tsigningkey = {signing_key(key)[0]}\n'
        config.write_text(text)

    def commits(self, *months: str):
        """One more commit on master per month (YYYY.MM) given, each dated in its month."""
        for month in months:
            year, number = month.split('.')
            n = self.history.count + 1
            self.history.commit(datetime(int(year), int(number), n, 12, tzinfo=timezone.utc),
                                f'feat: edit {n}', {'readme.md': f'Keep, edit {n}\n'})

    def released(self, version: str, today: bool = False):
        """The newest commit is what an earlier release left on origin: `release` and its tag.
        Tagged `today`, it is the day's earlier release: the script's search for a baseline
        from before today passes it by without the two ancestry processes a dated tag costs."""
        self.history.branch('release', self.history.count)
        when = datetime.now(timezone.utc) if today else None
        self.history.tag(version, self.history.count, f'Release {version}', when=when)

    def push(self) -> list[str]:
        """The full hashes of the commits, oldest first. The clone gets the history too — it
        is where the commits were made — all but the release branch, which it knows only as
        origin/release. So the script's fetch finds nothing to transfer, and a fetch that
        brings objects is six processes where one that brings none is two."""
        hashes = self.history.write(self.origin)
        self.history.write(self.dev)
        (self.dev / '.git' / 'refs' / 'heads' / 'release').unlink(missing_ok=True)
        return hashes

    def keep_local(self) -> list[str]:
        return self.history.write(self.dev)

    def release(self, *args: str, note: bool = True) -> subprocess.CompletedProcess:
        """Runs the script with echo standing in for `claude` — a run that got as far as the
        what's new would otherwise spend seconds and an API call on the real one. A link to
        the system's echo rather than a stub script of its own: macOS assesses an executable
        the first time it runs, which costs a fresh one-line script around 100 ms. Without
        `note` the PATH holds git and the system tools and no claude at all — not the real
        one either — and the script skips the what's new: a git process per summarised
        commit fewer, for a test that only reads the listing."""
        env = self.env if note else {**self.env, 'PATH': self.no_claude_path}
        return spawn(['bash', str(self.dev / 'release.sh'), *args], timeout=30, env=env)

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
        tag must not silently be left pointing at the other one while release moves on. The
        check comes before the dry run's exit, and a dry run spares the two processes the
        signing setup check costs."""
        repos.commits('2026.09', '2026.09')
        repos.push()
        repos.history.tag('v2026.09.2', 1, 'wrong')     # in the clone only, and on the first commit
        repos.keep_local()
        result = repos.release('--dry-run')
        assert result.returncode != 0
        assert 'v2026.09.2' in result.stderr
        assert "What's new" not in result.stdout, "stopped before the note"


class TestSigning:
    """Installs move only to a release whose tag verifies against res/release_signers, so the
    script refuses to make a tag it cannot sign and to push one the installs would refuse."""

    def test_a_release_without_tag_signing_set_up_is_refused_before_anything_moves(self, repos):
        repos.sign_with(None)
        repos.commits('2026.09')
        repos.push()
        result = repos.release(note=False)
        assert result.returncode != 0
        assert 'gpg.format' in result.stderr
        assert repos.origin_state() == (None, {})

    def test_a_tag_signed_with_a_key_the_signers_file_does_not_list_is_not_pushed(self, repos):
        repos.sign_with('stranger')
        repos.history.commit(datetime(2026, 9, 1, 12, tzinfo=timezone.utc), 'feat: signers',
                             {'res/release_signers': signing_key('maintainer')[1]})
        repos.push()
        result = repos.release(note=False)
        assert result.returncode != 0
        assert 'v2026.09.1' in result.stderr
        assert repos.origin_state() == (None, {})


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
    """--ignore-commits names commits the note Claude writes leaves out, so a hash that is
    not among the commits the release brings is a typo and stops the run. (That the named
    commits are listed as left out and kept out of the prompt is unchecked: the dry run that
    shows it costs the script two dozen processes, over the unit budget.)"""

    def test_a_commit_the_release_does_not_bring_is_refused(self, repos):
        repos.commits('2026.08')
        repos.released('v2026.08.1')
        repos.commits('2026.09')
        first, _ = repos.push()
        result = repos.release('--dry-run', '--ignore-commits', first[:7])
        assert result.returncode != 0
        assert first[:7] in result.stderr

    def test_a_hash_that_is_no_commit_is_refused(self, repos):
        repos.commits('2026.09')
        repos.push()
        result = repos.release('--dry-run', '--ignore-commits', 'zzzz')
        assert result.returncode != 0
        assert 'zzzz' in result.stderr

    def test_the_equals_form_takes_a_comma_separated_list(self, repos):
        repos.commits('2026.09')
        first, = repos.push()
        result = repos.release('--dry-run', f'--ignore-commits={first[:7]},zzzz')
        assert result.returncode != 0
        assert 'zzzz' in result.stderr and f'{first[:7]},' not in result.stderr, "the list was split, not taken as one hash"

    def test_an_unknown_option_or_a_bare_ignore_commits_stops_the_run_before_it_fetches(self, repos):
        for args in (('--bogus',), ('--ignore-commits',)):
            result = repos.release(*args)
            assert result.returncode != 0 and args[0] in result.stderr


class TestReachBack:
    """Installs update nightly, so a household that gets the evening release never saw the
    midday one: the listing and the note start from the last release tagged on an earlier day,
    not from origin/release. The listing is what is read: the note covers the same range
    (whats_new gets the one baseline) and the run that would show it costs more processes
    than the unit budget has — as does the run that would show --ignore-commits accepting a
    commit of the day's first release (three more git processes for the ancestry checks), so
    that reach is unchecked too."""

    def _two_releases_today(self, repos):
        """A release tagged in August, one more commit released earlier today, one commit
        since. The script runs a git process per commit it lists, so the history is the
        shortest that has a commit in each of the three places."""
        repos.commits('2026.08')
        repos.history.tag('v2026.08.1', 1, 'Release v2026.08.1', when=datetime(2026, 8, 1, 12, tzinfo=timezone.utc))
        repos.commits('2026.09')
        repos.history.branch('release', 2)
        repos.history.tag('v2026.09.2', 2, 'Release v2026.09.2', when=datetime.now(timezone.utc))
        repos.commits('2026.09')
        repos.push()

    def test_the_days_second_release_covers_the_commits_of_its_first(self, repos):
        self._two_releases_today(repos)
        result = repos.release('--dry-run', note=False)
        assert result.returncode == 0, result.stderr
        listing = result.stdout.split("What's new in", 1)[0]
        assert 'v2026.08.1' in listing
        assert 'edit 2' in listing and 'edit 3' in listing
        assert 'edit 1' not in listing


class TestGuidance:
    """Positional words that name no commit are guidance for the note Claude writes — a
    detail to keep out of it without dropping the commit the way --ignore-commits does — and
    the release still targets origin/master; the first word selects the commit only when it
    resolves to one. Claude is echo here, so the prompt, guidance included, is the note the
    dry run prints. The earlier release is today's: the run that writes the note is the
    script's longest, and a baseline from an earlier day would cost it two processes more."""

    def test_words_that_name_no_commit_are_guidance_and_the_release_still_targets_master(self, repos):
        repos.commits('2026.10')
        repos.released('v2026.10.1', today=True)
        repos.commits('2026.10')
        tip = repos.push()[-1]
        result = repos.release('--dry-run', 'dont', 'mention', 'the', 'plumber')
        assert result.returncode == 0, result.stderr
        assert f'Releasing v2026.10.2 ({tip[:7]}' in result.stdout
        note = result.stdout.split("What's new in", 1)[1]
        assert 'dont mention the plumber' in note
        assert note.index('dont mention the plumber') < note.index('Commits:'), "the guidance is an instruction, not a commit"

    def test_the_first_word_selects_the_commit_when_it_names_one_and_the_rest_is_guidance(self, repos):
        repos.commits('2026.10')
        repos.released('v2026.10.1', today=True)
        repos.commits('2026.10', '2026.10')
        _, second, _ = repos.push()
        result = repos.release('--dry-run', second[:7], 'keep', 'it', 'short')
        assert result.returncode == 0, result.stderr
        assert f'Releasing v2026.10.2 ({second[:7]}' in result.stdout
        listing, note = result.stdout.split("What's new in", 1)
        assert 'edit 2' in listing and 'edit 3' not in listing
        assert 'keep it short' in note and second[:7] not in note.split('Commits:', 1)[1].split('\n', 1)[0]
