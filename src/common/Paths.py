"""Filesystem anchors for the server tree.

The source packages live under src/, so per-module `Path(__file__).parents[N]`
arithmetic is fragile. Import these instead. Cwd-relative names ("log/…", "res/…")
elsewhere still assume the process runs from AISERVER_DIR, as start.sh guarantees.
"""
from pathlib import Path

AISERVER_DIR = Path(__file__).resolve().parents[2]
# The server tree is the repository root; both names stay so callers say which one they mean.
REPO_DIR = AISERVER_DIR
SRC_DIR = AISERVER_DIR / "src"
# One folder per outside service; the core finds what is in them by walking this directory.
INTEGRATIONS_DIR = SRC_DIR / "integrations"
RES_DIR = AISERVER_DIR / "res"
CACHE_DIR = AISERVER_DIR / "cache"
LOG_DIR = AISERVER_DIR / "log"
VOICE_ID_DIR = AISERVER_DIR / "voice_id"
# The assistant's long-term memory, all of it: the bank, the procedures, the watch list
# (assistant/MemoryFile.py, Procedures.py, WatchList.py). Only what is live; what they
# replace, delete or close is copied to MEMORY_LOG_DIR first, for debugging.
MEMORY_DIR = AISERVER_DIR / "memory"
MEMORY_LOG_DIR = LOG_DIR / "memory"
# The ids of the Claude Code sessions the server itself starts (CodeQualityMaintainer's `claude -p`
# runs), one per line, so the Claude Code feed can leave them out of the owner's journal.
CLAUDE_OWN_SESSIONS_PATH = LOG_DIR / "claude_code_own_sessions.txt"
CONFIG_PATH = AISERVER_DIR / "config.yaml"
# A developer's stand-in config: //wipe puts it in place of the wiped config.yaml when it
# exists, so a dev install comes back with keys instead of the skeleton.
DUMMY_CONFIG_PATH = AISERVER_DIR / "config.dummy.yaml"
VAULT_PATH = AISERVER_DIR / "vault.bin"
# pyicloud's default session store is the per-user temp dir, which macOS wipes on
# reboot — every reboot then forced a from-scratch password signin + 2FA (2026-08-30).
ICLOUD_SESSION_DIR = CACHE_DIR / "icloud_session"
