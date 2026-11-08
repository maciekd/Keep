"""Ds4DiskCacheSweeper — the age sweep of ds4-server's disk KV cache directory."""
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from assistant.llm import Ds4DiskCacheSweeper as module
from assistant.llm.Ds4DiskCacheSweeper import Ds4DiskCacheSweeper

NOW = 1_800_000_000.0
HOUR = 3600


def _entry(directory: Path, name: str, age_hours: float, size: int = 8) -> Path:
    path = directory / name
    path.write_bytes(b"k" * size)
    os.utime(path, (NOW - age_hours * HOUR, NOW - age_hours * HOUR))
    return path


SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40


class TestSweep:
    def test_removes_only_the_entries_last_used_before_the_limit(self, tmp_path):
        old = _entry(tmp_path, f"{SHA_A}.kv", age_hours=30)
        fresh = _entry(tmp_path, f"{SHA_B}.kv", age_hours=2)
        just_inside = _entry(tmp_path, f"{SHA_C}.kv", age_hours=23.9)

        removed = Ds4DiskCacheSweeper(tmp_path, max_age_hours=24).sweep(now=NOW)

        assert removed == 1
        assert not old.exists()
        assert fresh.exists() and just_inside.exists()

    def test_a_hit_keeps_an_old_entry(self, tmp_path):
        # The server rewrites the header on every hit, which is what makes mtime the last use:
        # an entry created long ago but touched today stays.
        path = _entry(tmp_path, f"{SHA_A}.kv", age_hours=100)
        os.utime(path, (NOW - HOUR, NOW - HOUR))

        assert Ds4DiskCacheSweeper(tmp_path, max_age_hours=24).sweep(now=NOW) == 0
        assert path.exists()

    def test_leaves_the_servers_store_in_progress_and_foreign_files_alone(self, tmp_path):
        in_progress = _entry(tmp_path, f"{SHA_A}.kv.tmp.4242", age_hours=50)
        foreign = _entry(tmp_path, "notes.txt", age_hours=50)
        not_a_sha = _entry(tmp_path, "cache.kv", age_hours=50)

        assert Ds4DiskCacheSweeper(tmp_path, max_age_hours=24).sweep(now=NOW) == 0
        assert in_progress.exists() and foreign.exists() and not_a_sha.exists()

    def test_a_missing_directory_is_nothing_to_do(self, tmp_path):
        assert Ds4DiskCacheSweeper(tmp_path / "gone", max_age_hours=24).sweep(now=NOW) == 0

    def test_an_entry_the_server_evicts_mid_sweep_is_not_an_error(self, tmp_path):
        _entry(tmp_path, f"{SHA_A}.kv", age_hours=30)
        real_unlink = Path.unlink

        def vanish(self, *args, **kwargs):
            real_unlink(self)
            raise FileNotFoundError(self)

        with patch.object(Path, 'unlink', vanish):
            assert Ds4DiskCacheSweeper(tmp_path, max_age_hours=24).sweep(now=NOW) == 0

    @pytest.mark.xfail(strict=True, reason="sweep() catches FileNotFoundError alone: any other OSError from unlink "
                                           "aborts the whole sweep, so the entries after the stuck one are never removed")
    def test_an_entry_that_cannot_be_removed_does_not_shield_the_others(self, tmp_path):
        # One entry the process may not unlink (a stray root-owned file, a directory of that
        # name) comes back every hour; the sweep must still free the rest of the budget.
        stuck = _entry(tmp_path, f"{SHA_A}.kv", age_hours=30)
        other = _entry(tmp_path, f"{SHA_B}.kv", age_hours=30)
        real_unlink = Path.unlink

        def unlink(self, *args, **kwargs):
            if self == stuck:
                raise PermissionError(self)
            real_unlink(self, *args, **kwargs)

        with patch.object(Path, 'unlink', unlink):
            removed = Ds4DiskCacheSweeper(tmp_path, max_age_hours=24).sweep(now=NOW)

        assert removed == 1 and not other.exists() and stuck.exists()

    def test_uses_the_clock_when_no_time_is_given(self, tmp_path):
        path = tmp_path / f"{SHA_A}.kv"
        path.write_bytes(b"k")
        os.utime(path, (time.time() - 48 * HOUR, time.time() - 48 * HOUR))
        assert Ds4DiskCacheSweeper(tmp_path, max_age_hours=24).sweep() == 1


class TestLoop:
    def test_a_failed_sweep_does_not_end_the_hourly_loop(self, tmp_path):
        sweeper = Ds4DiskCacheSweeper(tmp_path, max_age_hours=24)
        outcomes = iter([OSError("directory unreadable"), None])

        def sweep(now=None):
            if (failure := next(outcomes)) is not None:
                raise failure
            sweeper.stop()

        with patch.object(module, 'SWEEP_INTERVAL_S', 0.0), patch.object(sweeper, 'sweep', side_effect=sweep) as sweeps:
            sweeper._loop()

        assert sweeps.call_count == 2


class TestConfig:
    @pytest.fixture(autouse=True)
    def _no_started_sweeper(self):
        saved = Ds4DiskCacheSweeper._started
        Ds4DiskCacheSweeper._started = None
        yield
        Ds4DiskCacheSweeper._started = None
        Ds4DiskCacheSweeper._started = saved

    def _config(self, values: dict):
        return patch.object(module.config, 'get', side_effect=lambda key, default=None: values.get(key, default))

    def test_the_directory_and_age_keys_are_read_from_the_ds4_section(self, tmp_path):
        # Contract with config.yaml: `ds4.kv_disk_dir` and `ds4.kv_disk_max_age_hours`.
        with self._config({'ds4.kv_disk_dir': str(tmp_path), 'ds4.kv_disk_max_age_hours': 6}):
            sweeper = Ds4DiskCacheSweeper.from_config()
        assert sweeper.directory == tmp_path
        assert sweeper.max_age_hours == 6

    def test_the_default_directory_is_the_launchers_expanded(self):
        with self._config({}):
            sweeper = Ds4DiskCacheSweeper.from_config()
        assert sweeper.directory.is_absolute()
        assert "~" not in str(sweeper.directory)

    def test_zero_hours_turns_the_sweep_off(self):
        with self._config({'ds4.kv_disk_max_age_hours': 0}):
            assert Ds4DiskCacheSweeper.from_config() is None

    def test_starts_once_and_only_for_a_local_server(self, tmp_path):
        with self._config({'ds4.kv_disk_dir': str(tmp_path)}), patch.object(Ds4DiskCacheSweeper, 'start') as start:
            Ds4DiskCacheSweeper.start_once(server_is_local=False)
            assert start.call_count == 0 and Ds4DiskCacheSweeper._started is None
            Ds4DiskCacheSweeper.start_once(server_is_local=True)
            Ds4DiskCacheSweeper.start_once(server_is_local=True)
            assert start.call_count == 1
            assert Ds4DiskCacheSweeper._started.directory == tmp_path
