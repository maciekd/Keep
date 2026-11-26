"""tools/clean_install_test/run.sh: what of a checkout goes to the VM host as "the working tree".

The run itself needs another machine; `bundle_worktree` is lifted out of the script and run on a
throwaway checkout, with `scp` a function that keeps what would be sent."""
import re
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from common.Paths import AISERVER_DIR
from tests.unit.git_sandbox import History, empty_repo, git, git_env, spawn

RUN_SH = AISERVER_DIR / 'tools' / 'clean_install_test' / 'run.sh'
BASH = shutil.which('bash')
SECRET = 'anthropic_api_key: sk-ant-kept-47L\n'
COPIES = ['config.yaml.bak-2026-11-25', 'config.yaml copy', 'config.yml.old']


class Bundle:
    def __init__(self, tmp: Path):
        self.repo = empty_repo(tmp / 'Keep')
        history = History()
        history.commit(datetime(2026, 11, 25, 9, 0), 'the tree', {
            'install.sh': '#!/bin/bash\n', 'config.example.yaml': 'owner_name: Marcus\n',
            '.gitignore': 'config.yaml\n', 'src/app.py': 'VERSION = 1\n'})
        history.write(self.repo)
        git(self.repo, 'reset', '-q', '--hard')
        for name in ['config.yaml', *COPIES]:
            (self.repo / name).write_text(SECRET)
        (self.repo / 'notes.txt').write_text('an untracked file of the work in progress\n')
        (self.repo / 'src' / 'app.py').write_text('VERSION = 2\n')
        self.status_before = git(self.repo, 'status', '--porcelain')

        function = re.search(r'^bundle_worktree\(\) \{\n.*?^\}\n', RUN_SH.read_text(), re.S | re.M).group(0)
        self.sent = tmp / 'sent'
        self.sent.mkdir()
        script = tmp / 'bundle.sh'
        script.write_text('set -euo pipefail\n'
                          f'cd "{self.repo}"\n'
                          'HOST=mac-mini.example; REMOTE_DIR=keep-install-tests\n'
                          f'scp() {{ cp "$2" "$3" "{self.sent}/"; }}\n'      # scp -q <bundle> <install.sh> <host:dir>
                          f'{function}bundle_worktree\n')
        self.result = spawn([BASH, str(script)], env=git_env())
        assert self.result.returncode == 0, self.result.stderr
        head = spawn(['git', 'bundle', 'list-heads', str(self.sent / 'keep-src.bundle')], check=True).stdout.split()[0]
        # commit-tree stored the commit in the checkout's own objects, so it is read there.
        self.files = git(self.repo, 'ls-tree', '-r', '--name-only', head).splitlines()
        self.app = git(self.repo, 'show', f'{head}:src/app.py')


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    return Bundle(tmp_path_factory.mktemp('clean_install'))


@pytest.mark.unit
class TestTheBundle:
    def test_no_copy_of_the_config_goes_to_the_host_whatever_it_is_called(self, bundle):
        """The ignore rules know `config.yaml`; a copy somebody made beside it holds the same secrets."""
        assert [name for name in bundle.files if name.startswith(('config.yaml', 'config.yml'))] == []

    def test_the_example_config_the_installer_reads_is_still_in_it(self, bundle):
        assert 'config.example.yaml' in bundle.files and 'install.sh' in bundle.files

    def test_it_is_the_checkout_as_it_stands_uncommitted_and_untracked_files_included(self, bundle):
        assert 'notes.txt' in bundle.files
        assert bundle.app == 'VERSION = 2'

    def test_the_checkouts_own_index_is_not_touched(self, bundle):
        assert git(bundle.repo, 'status', '--porcelain') == bundle.status_before
        assert all((bundle.repo / name).read_text() == SECRET for name in COPIES)
