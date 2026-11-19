import os
import pytest
from unittest.mock import patch

from common.audio_log import cleanup_audio_folder


@pytest.mark.unit
class TestCleanupAudioFolder:
    def test_under_limit_no_files_deleted(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"x" * 100)
        (tmp_path / "b.wav").write_bytes(b"x" * 100)

        cleanup_audio_folder(str(tmp_path), max_size_gb=1)

        assert len(list(tmp_path.iterdir())) == 2

    def test_oserror_on_delete_continues(self, tmp_path):
        file1 = tmp_path / "locked.wav"
        file1.write_bytes(b"x" * 500)
        os.utime(file1, (1000, 1000))

        file2 = tmp_path / "deletable.wav"
        file2.write_bytes(b"x" * 500)
        os.utime(file2, (2000, 2000))

        original_remove = os.remove

        def failing_remove(path):
            if "locked" in path:
                raise OSError("Permission denied")
            original_remove(path)

        with patch('os.remove', side_effect=failing_remove):
            cleanup_audio_folder(str(tmp_path), max_size_gb=100 / (1024 ** 3))

        # locked.wav should still exist (remove failed), deletable.wav should be deleted
        remaining = {f.name for f in tmp_path.iterdir()}
        assert "locked.wav" in remaining
        assert "deletable.wav" not in remaining


from common.StderrLogger import StderrLogger


@pytest.mark.unit
class TestStderrLogger:
    def test_rapid_writes_buffered_into_single_message(self):
        logger = StderrLogger()
        with patch('common.logging.info') as mock_info:
            logger.write("T")
            logger.write("r")
            logger.write("a")
            logger.write("c")
            logger.write("e")
            logger.flush()
            mock_info.assert_called_once_with("Trace")

    def test_whitespace_only_writes_do_not_log(self):
        logger = StderrLogger()
        with patch('common.logging.info') as mock_info:
            logger.write("   ")
            logger.write("\n")
            logger.flush()
            mock_info.assert_not_called()

    def test_timer_flushes_after_idle(self):
        logger = StderrLogger()
        with patch('common.logging.info') as mock_info, patch('common.StderrLogger.threading.Timer') as timer:
            logger.write("delayed message")
            idle, flush = timer.call_args.args
            assert idle == 0.05 and timer.return_value.start.called and not mock_info.called
            flush()
            mock_info.assert_called_once_with("delayed message")
