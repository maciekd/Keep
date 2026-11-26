"""tools/clean_install_test/run.sh: what of a checkout goes to the VM host as "the working tree".

The run itself needs another machine; `bundle_worktree` is lifted out of the script and run on a
throwaway checkout, with `scp` a function that keeps what would be sent."""
import re
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from common.Paths import AISERVER_DIR
from tests.unit.git_sandbox import History, empty_repo, files_at, git, git_env, spawn

RUN_SH = AISERVER_DIR / 'tools' / 'clean_install_test' / 'run.sh'
BASH = shutil.which('bash')
SECRET = 'anthropic_api_key: sk-ant-kept-47L\n'
COPIES = ['config.yaml.bak-2026-11-25', 'config.yaml copy', 'config.yml.old']


class Bundle:
    def __init__(self, tmp: Path):
        self.repo = empty_repo(tmp / 'Keep')
        committed = {'install.sh': '#!/bin/bash\n', 'config.example.yaml': 'owner_name: Marcus\n',
                     '.gitignore': 'config.yaml\n', 'src/app.py': 'VERSION = 1\n'}
        history = History()
        history.commit(datetime(2026, 11, 25, 9, 0), 'the tree', committed)
        history.write(self.repo)
        git(self.repo, 'read-tree', 'HEAD')     # the index of a checkout; the files are written here
        (self.repo / 'src').mkdir()
        work_tree = {**committed, 'src/app.py': 'VERSION = 2\n',
                     'notes.txt': 'an untracked file of the work in progress\n',
                     **{name: SECRET for name in ['config.yaml', *COPIES]}}
        for name, text in work_tree.items():
            (self.repo / name).write_text(text)
        self.index = self.repo / '.git' / 'index'
        self.index_before = self.index.read_bytes()

        function = re.search(r'^bundle_worktree\(\) \{\n.*?^\}\n', RUN_SH.read_text(), re.S | re.M).group(0)
        work, sent = tmp / 'work', tmp / 'sent'
        work.mkdir()
        script = tmp / 'bundle.sh'
        script.write_text('set -euo pipefail\n'
                          f'cd "{self.repo}"\n'
                          'HOST=mac-mini.example; REMOTE_DIR=keep-install-tests\n'
                          # Functions for what is not git, each a process less: scp -q <bundle>
                          # <install.sh> <host:dir> says what it was given, and the files stay.
                          f'scp() {{ printf "%s\\n" "$2" "$3" > "{sent}"; }}\n'
                          f'mktemp() {{ echo "{work}"; }}\n'
                          'rm() { :; }\n'
                          f'{function}bundle_worktree\n')
        self.result = spawn([BASH, str(script)], env=git_env())
        assert self.result.returncode == 0, self.result.stderr
        # A bundle opens with its signature line and then a line per head: "<commit> <ref>".
        head = Path(sent.read_text().splitlines()[0]).read_bytes().split(b'\n', 2)[1].split()[0].decode()
        # commit-tree stored the commit in the checkout's own objects, so it is read there —
        # off the files: every git process is ~5 ms of each worker's first test of this module.
        self.files = files_at(self.repo, head)


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
        assert bundle.files['src/app.py'] == 'VERSION = 2\n'

    def test_the_checkouts_own_index_is_not_touched(self, bundle):
        assert bundle.index.read_bytes() == bundle.index_before
        assert all((bundle.repo / name).read_text() == SECRET for name in COPIES)
