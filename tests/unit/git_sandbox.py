"""Throwaway git repositories for the tests of the shell scripts (release.sh, install.sh).

The scripts run real git, and a git process costs milliseconds while a unit test has a
hundred of them to spend — under the suite's eight workers a process costs half as much
again. So the scripts get the fastest git there is (`git_env`), started the cheapest way
(`spawn`), and the repositories they work on are written here without any process:
`empty_repo` leaves what `git init` would, `History` stores the commits as the loose
objects `git commit` would.
"""
import functools
import hashlib
import os
import shutil
import subprocess
import zlib
from datetime import datetime
from pathlib import Path

NAME, EMAIL = 'Marcus', 'm@mlab.test'


@functools.cache
def git_env() -> dict:
    """The environment for a git that is fast and knows nothing of this machine: git's own
    directory leads PATH (macOS's /usr/bin/git is an xcrun shim that adds ~6 ms to every
    call, and release.sh alone makes a dozen), the user's and the system's config are out of
    the picture (a signing key, hooks, a credential helper), the identity is the cast's, and
    no fetch or merge ends by starting a maintenance process for a repository that lives a
    tenth of a second."""
    exec_path = subprocess.run(['git', '--exec-path'], capture_output=True, text=True, check=True).stdout.strip()
    path = os.environ.get('PATH', '')
    if (Path(exec_path) / 'git').exists():
        path = exec_path + os.pathsep + path
    return {**os.environ, 'PATH': path, 'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_AUTHOR_NAME': NAME, 'GIT_AUTHOR_EMAIL': EMAIL,
            'GIT_COMMITTER_NAME': NAME, 'GIT_COMMITTER_EMAIL': EMAIL,
            'GIT_CONFIG_COUNT': '2', 'GIT_CONFIG_KEY_0': 'gc.auto', 'GIT_CONFIG_VALUE_0': '0',
            'GIT_CONFIG_KEY_1': 'maintenance.auto', 'GIT_CONFIG_VALUE_1': 'false'}


git_env()  # at collection, so the `git --exec-path` process is not the first script test's to pay


def spawn(command: list, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run on posix_spawn, in git's environment. Python forks instead as soon as
    it is given a `cwd`, a bare program name or descriptors to close, and forking a pytest
    worker with the whole tree imported takes ~6 ms where the spawn takes under one — so
    the program is resolved here and a command that needs a directory is told it in its
    own arguments (`git -C`, a script's full path)."""
    return subprocess.run([_program(command[0]), *command[1:]], env=kwargs.pop('env', None) or git_env(),
                          capture_output=True, text=True, close_fds=False, **kwargs)


@functools.cache
def _program(name: str) -> str:
    return shutil.which(name, path=git_env()['PATH']) or name


def git(repo: Path, *args: str, check: bool = True) -> str:
    return spawn(['git', '-C', str(repo), *args], check=check).stdout.strip()


def empty_repo(path: Path, bare: bool = False, origin: Path | None = None) -> Path:
    """What `git init -b master` leaves that git needs — HEAD, objects/, refs/ and a config —
    written without the process; `origin` adds the remote a clone would have."""
    git_dir = path if bare else path / '.git'
    (git_dir / 'objects').mkdir(parents=True)
    (git_dir / 'refs' / 'heads').mkdir(parents=True)
    (git_dir / 'HEAD').write_text('ref: refs/heads/master\n')
    config = f'[core]\n\trepositoryformatversion = 0\n\tbare = {str(bare).lower()}\n'
    if origin is not None:
        config += f'[remote "origin"]\n\turl = {origin}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
    (git_dir / 'config').write_text(config)
    return path


def git_dir(repo: Path) -> Path:
    return repo / '.git' if (repo / '.git').is_dir() else repo


def _store(repo: Path, kind: str, body: bytes) -> str:
    """One loose object, git's way: zlib("<kind> <size>\\0<body>") under the SHA-1 of the same."""
    data = f'{kind} {len(body)}\0'.encode() + body
    name = hashlib.sha1(data).hexdigest()
    path = git_dir(repo) / 'objects' / name[:2] / name[2:]
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(zlib.compress(data))
    return name


def _tree(repo: Path, files: dict[str, str]) -> str:
    """The tree of `files` (paths with `/` become subtrees), entries in git's order: a
    directory sorts as its name with a trailing slash."""
    blobs = {name: text for name, text in files.items() if '/' not in name}
    dirs: dict[str, dict[str, str]] = {}
    for name, text in files.items():
        if '/' in name:
            head, rest = name.split('/', 1)
            dirs.setdefault(head, {})[rest] = text
    entries = [(name, f'100644 {name}', _store(repo, 'blob', text.encode())) for name, text in blobs.items()]
    entries += [(name + '/', f'40000 {name}', _tree(repo, sub)) for name, sub in dirs.items()]
    return _store(repo, 'tree', b''.join(f'{mode}\0'.encode() + bytes.fromhex(sha)
                                         for _, mode, sha in sorted(entries)))


def _load(repo: Path, name: str) -> bytes:
    data = zlib.decompress((git_dir(repo) / 'objects' / name[:2] / name[2:]).read_bytes())
    return data[data.index(b'\0') + 1:]


def files_at(repo: Path, commit: str, tree: str = '', prefix: str = '') -> dict[str, str]:
    """Every file of a commit with its text, read off the loose objects — what `git ls-tree -r`
    and a `git show` per file would answer, without the processes. Only for objects no gc has
    packed, which is every object of a repository made here."""
    body = _load(repo, tree or _load(repo, commit).split(b'\n', 1)[0].split()[1].decode())
    files = {}
    while body:
        entry, rest = body.split(b'\0', 1)
        mode, name = entry.decode().split(' ', 1)
        sha, body = rest[:20].hex(), rest[20:]
        if mode == '40000':
            files.update(files_at(repo, commit, sha, f'{prefix}{name}/'))
        else:
            files[prefix + name] = _load(repo, sha).decode()
    return files


@functools.cache
def signing_key(name: str) -> tuple[Path, str]:
    """A throwaway ed25519 key that signs tags the way the maintainer's does (`gpg.format ssh`,
    `user.signingkey` the private key file), made once per worker: its file and the line
    `res/release_signers` lists it with."""
    import tempfile
    path = Path(tempfile.mkdtemp(prefix='keep-signing-')) / name
    spawn(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', name, '-f', str(path)], check=True)
    return path, f"{EMAIL} {Path(f'{path}.pub').read_text().strip()}\n"


class History:
    """A line of commits on `master`, each holding just the files it is given, plus branches
    and annotated tags on them. `write` stores all of it in a repository and returns the
    commit hashes, oldest first. The hashes follow from the content, so the same history
    written into two repositories is the same commits: a clone and its origin, without a
    fetch or a push. Only objects and refs are written — a work tree wants a
    `git reset --hard` afterwards."""

    def __init__(self):
        self._commits: list[tuple[datetime, str, dict[str, str]]] = []
        self._branches: dict[str, int] = {}
        self._tags: dict[str, tuple[int, str, datetime | None]] = {}

    @property
    def count(self) -> int:
        return len(self._commits)

    def commit(self, when: datetime, message: str, files: dict[str, str]) -> int:
        """One more commit on master at `when` (its author and committer date); returns its
        number, 1 for the first."""
        self._commits.append((when, message, files))
        return self.count

    def branch(self, name: str, at: int):
        self._branches[name] = at

    def tag(self, name: str, at: int, message: str = 'tagged', when: datetime | None = None):
        """An annotated tag, as `git tag -a` makes; tagged at `when`, or when the newest
        commit was made."""
        self._tags[name] = (at, message, when)

    def write(self, repo: Path) -> list[str]:
        hashes: list[str] = []
        who = ''
        for when, message, files in self._commits:
            tree = _tree(repo, files)
            parent = f'parent {hashes[-1]}\n' if hashes else ''
            who = f'{NAME} <{EMAIL}> {int(when.timestamp())} +0000'
            hashes.append(_store(repo, 'commit',
                                 f'tree {tree}\n{parent}author {who}\ncommitter {who}\n\n{message}\n'.encode()))
        refs = {'refs/heads/master': hashes[-1]}
        refs.update({f'refs/heads/{name}': hashes[at - 1] for name, at in self._branches.items()})
        for name, (at, message, when) in self._tags.items():
            tagger = f'{NAME} <{EMAIL}> {int(when.timestamp())} +0000' if when else who
            refs[f'refs/tags/{name}'] = _store(
                repo, 'tag', f'object {hashes[at - 1]}\ntype commit\ntag {name}\ntagger {tagger}\n\n{message}\n'.encode())
        for ref, name in refs.items():
            path = git_dir(repo) / ref
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + '\n')
        return hashes
