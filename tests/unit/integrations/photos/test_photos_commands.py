from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from assistant.ChannelProfile import profile_for
from assistant.commands.Commands import Commands
from integrations.photos.ICloudPhotoDownloader import NoVideoRenditionError, PhotoNotInCloudError
from integrations.photos.PhotosCommands import MediaPayload, PhotosCommands


@contextmanager
def _screen_config(**values):
    with patch("integrations.photos.PhotosCommands.config") as cfg:
        cfg.get.side_effect = lambda key, default=None: values.get(key, default)
        yield


def photo_commands(commands) -> PhotosCommands:
    """The photo command set a Commands hosts."""
    return commands._owner("show_photo")


class TestCmdSearchPhotos:
    def test_formats_results(self):
        mock_results = [
            {
                "id": 42,
                "short_description": "Family dinner at Italian restaurant",
                "location_text": "Frankfurt/Germany",
                "creation_date": 1710460800.0,  # 2024-03-15
                "pretty_score": 78,
                "family_importance_score": 92,
            },
            {
                "id": 87,
                "short_description": "Kids playing in the park",
                "location_text": "Home",
                "creation_date": 1719014400.0,  # 2024-06-22
                "pretty_score": 65,
                "family_importance_score": 85,
            },
        ]
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.get_instance.return_value.search_photos.return_value = mock_results
            result = PhotosCommands(Commands(None))._run_search_photos("family dinner")

        assert "42: Family dinner at Italian restaurant" in result
        assert "Frankfurt/Germany" in result
        assert "[pretty: 78, family: 92]" in result
        assert "87: Kids playing in the park" in result
        assert "[pretty: 65, family: 85]" in result

    def test_empty_results(self):
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.get_instance.return_value.search_photos.return_value = []
            result = PhotosCommands(Commands(None))._run_search_photos("nonexistent")

        assert result == 'No photos found matching "nonexistent".'

    def test_passes_sort_by_to_database(self):
        mock_results = [
            {
                "id": 1,
                "short_description": "Pretty sunset",
                "location_text": "Home",
                "creation_date": 1710460800.0,
                "pretty_score": 95,
                "family_importance_score": 30,
            },
        ]
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.SEARCH_SORT_OPTIONS = {"relevance": "rank", "pretty": "p.pretty_score DESC",
                                               "family": "p.family_importance_score DESC",
                                               "best": "(p.pretty_score + p.family_importance_score) DESC"}
            mock_db_cls.get_instance.return_value.search_photos.return_value = mock_results
            PhotosCommands(Commands(None))._run_search_photos("sunset", "pretty")

        mock_db_cls.get_instance.return_value.search_photos.assert_called_once_with("sunset", limit=25, sort_by="pretty")

    def test_invalid_sort_by_defaults_to_relevance(self):
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.SEARCH_SORT_OPTIONS = {"relevance": "rank", "pretty": "p.pretty_score DESC",
                                               "family": "p.family_importance_score DESC",
                                               "best": "(p.pretty_score + p.family_importance_score) DESC"}
            mock_db_cls.get_instance.return_value.search_photos.return_value = []
            PhotosCommands(Commands(None))._run_search_photos("sunset", "invalid_sort")

        mock_db_cls.get_instance.return_value.search_photos.assert_called_once_with("sunset", limit=25, sort_by="relevance")

    def test_no_sort_by_defaults_to_relevance(self):
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.SEARCH_SORT_OPTIONS = {"relevance": "rank", "pretty": "p.pretty_score DESC",
                                               "family": "p.family_importance_score DESC",
                                               "best": "(p.pretty_score + p.family_importance_score) DESC"}
            mock_db_cls.get_instance.return_value.search_photos.return_value = []
            PhotosCommands(Commands(None))._run_search_photos("sunset beach")

        mock_db_cls.get_instance.return_value.search_photos.assert_called_once_with("sunset beach", limit=25, sort_by="relevance")

    def test_llm_str_prefix_workaround(self):
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.SEARCH_SORT_OPTIONS = {"relevance": "rank", "pretty": "p.pretty_score DESC",
                                               "family": "p.family_importance_score DESC",
                                               "best": "(p.pretty_score + p.family_importance_score) DESC"}
            mock_db_cls.get_instance.return_value.search_photos.return_value = []
            PhotosCommands(Commands(None))._run_search_photos("str = ingeborg", "str = best")

        mock_db_cls.get_instance.return_value.search_photos.assert_called_once_with("ingeborg", limit=25, sort_by="best")

    def test_wildcard_keyword_passes_to_database(self):
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.SEARCH_SORT_OPTIONS = {"relevance": "rank", "pretty": "p.pretty_score DESC",
                                               "family": "p.family_importance_score DESC",
                                               "best": "(p.pretty_score + p.family_importance_score) DESC"}
            mock_db_cls.get_instance.return_value.search_photos.return_value = []
            result = PhotosCommands(Commands(None))._run_search_photos("*", "pretty")

        mock_db_cls.get_instance.return_value.search_photos.assert_called_once_with("*", limit=25, sort_by="pretty")
        assert result == "No photos found."

    def test_screenshot_tag_shown(self):
        mock_results = [
            {
                "id": 101,
                "short_description": "App settings screen",
                "location_text": "Home",
                "creation_date": 1710460800.0,
                "pretty_score": 10,
                "family_importance_score": 5,
                "photo_type": "photo, screenshot",
            },
        ]
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.get_instance.return_value.search_photos.return_value = mock_results
            result = PhotosCommands(Commands(None))._run_search_photos("settings")

        assert "(screenshot)" in result

    def test_persons_tag_shown(self):
        mock_results = [
            {
                "id": 42,
                "short_description": "Family dinner",
                "location_text": "Frankfurt/Germany",
                "creation_date": 1710460800.0,
                "pretty_score": 78,
                "family_importance_score": 92,
                "persons": "Alice, Bob",
            },
        ]
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.get_instance.return_value.search_photos.return_value = mock_results
            result = PhotosCommands(Commands(None))._run_search_photos("family dinner")

        assert "[Alice, Bob]" in result

    def test_no_persons_tag_when_empty(self):
        mock_results = [
            {
                "id": 42,
                "short_description": "Landscape photo",
                "location_text": "Norway",
                "creation_date": 1710460800.0,
                "pretty_score": 90,
                "family_importance_score": 10,
                "persons": "",
            },
        ]
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.get_instance.return_value.search_photos.return_value = mock_results
            result = PhotosCommands(Commands(None))._run_search_photos("landscape")

        assert result.endswith("]")
        assert "[pretty: 90, family: 10]" in result
        # No persons tag after the scores
        assert result.strip().endswith("[pretty: 90, family: 10]")

    def test_missing_location_shows_unknown(self):
        mock_results = [
            {
                "id": 1,
                "short_description": "A sunset",
                "location_text": None,
                "creation_date": 1710460800.0,
                "pretty_score": 50,
                "family_importance_score": 30,
            },
        ]
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.get_instance.return_value.search_photos.return_value = mock_results
            result = PhotosCommands(Commands(None))._run_search_photos("sunset")

        assert "Unknown location" in result

    def test_search_results_carry_video_tag(self):
        results = [{"id": 7, "short_description": "Frances sings", "photo_type": "video", "duration_seconds": 12.0,
                    "creation_date": 1700000000.0, "location_text": "Home", "pretty_score": 60,
                    "family_importance_score": 90, "persons": "Frances"}]
        with patch('integrations.photos.PhotosCommands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.SEARCH_SORT_OPTIONS = ("relevance",)
            mock_db_cls.get_instance.return_value.search_photos.return_value = results
            out = PhotosCommands(Commands(None))._run_search_photos("frances")
        assert "7: Frances sings (video, 12s) (Home, " in out


class TestCmdShowPhoto:
    def test_calls_background_command_with_the_requests_route(self, commands_instance):
        """The dispatched function carries the chat of the request, captured at dispatch."""
        commands_instance._current_context = Mock(source="TextChat", conversation=222)
        with patch.object(commands_instance, 'background_command') as mock_bg, \
             patch.object(photo_commands(commands_instance), '_run_show_photo', return_value="") as run:
            commands_instance.run("show_photo", ["42"])
            (function, params), _ = mock_bg.call_args
            assert params == ["42"]
            function(*params)
        run.assert_called_once_with("42", route={"source": "TextChat", "conversation": 222})

    def test_show_photo_not_found(self, commands_instance):
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = None
            result = photo_commands(commands_instance)._run_show_photo("999")

        assert "no photo found" in result

    def test_show_photo_invalid_id(self, commands_instance):
        result = photo_commands(commands_instance)._run_show_photo("not_a_number")
        assert "not a valid photo ID" in result

    def test_show_photo_image_missing(self, commands_instance):
        mock_photo = {
            "id": 42,
            "asset_guid": "ABC-DEF",
            "short_description": "A photo",
        }
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("LOCAL-UUID", "IMG_1234.HEIC", "DCIM/100APPLE", None, None, 0)
        mock_conn.cursor.return_value = mock_cursor

        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls, \
             patch("sqlite3.connect", return_value=mock_conn), \
             patch("integrations.photos.PhotoProcessor.PhotoProcessor._get_image_bytes", return_value=None):
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = mock_photo
            result = photo_commands(commands_instance)._run_show_photo("42")

        assert "could not extract image" in result

    def test_show_photo_single_id_still_uses_send_photo(self, commands_instance):
        mock_photo = {
            "id": 42,
            "asset_guid": "ABC-DEF",
            "short_description": "Beach sunset",
        }
        mock_telegram = Mock()
        jpeg_bytes = b"\xff\xd8\xff\xe0fake_jpeg"
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("LOCAL-UUID", "IMG_1234.HEIC", "DCIM/100APPLE", None, None, 0)
        mock_conn.cursor.return_value = mock_cursor

        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls, \
             patch("sqlite3.connect", return_value=mock_conn), \
             patch("integrations.photos.PhotoProcessor.PhotoProcessor._get_image_bytes", return_value=jpeg_bytes), \
             patch("assistant.Notifier._notifier", mock_telegram):
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = mock_photo
            result = photo_commands(commands_instance)._run_show_photo("42")

        assert result == ""
        mock_telegram.send_photo.assert_called_once_with(jpeg_bytes, caption="Beach sunset")
        mock_telegram.send_photo_album.assert_not_called()

    def test_show_photo_without_notifier_reports_missing_channel(self, commands_instance):
        mock_photo = {"id": 42, "asset_guid": "ABC-DEF", "short_description": "Sunset"}
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("LOCAL-UUID", "IMG_1234.HEIC", "DCIM/100APPLE", None, None, 0)
        mock_conn.cursor.return_value = mock_cursor
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls, \
             patch("sqlite3.connect", return_value=mock_conn), \
             patch("integrations.photos.PhotoProcessor.PhotoProcessor._get_image_bytes", return_value=b"jpeg"):
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = mock_photo
            result = photo_commands(commands_instance)._run_show_photo("42")
        assert "no notification channel configured" in result

    def test_show_photo_multiple_ids_sends_album(self, commands_instance):
        photos = {
            10: {"id": 10, "asset_guid": "GUID-10", "short_description": "Photo A"},
            20: {"id": 20, "asset_guid": "GUID-20", "short_description": "Photo B"},
        }
        jpeg_a = b"\xff\xd8jpeg_a"
        jpeg_b = b"\xff\xd8jpeg_b"
        mock_telegram = Mock()

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            ("LOCAL-UUID-A", "IMG_A.HEIC", "DCIM/A", None, None, 0),
            ("LOCAL-UUID-B", "IMG_B.HEIC", "DCIM/B", None, None, 0),
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls, \
             patch("sqlite3.connect", return_value=mock_conn), \
             patch("integrations.photos.PhotoProcessor.PhotoProcessor._get_image_bytes",
                   side_effect=[jpeg_a, jpeg_b]), \
             patch("assistant.Notifier._notifier", mock_telegram):
            mock_db_cls.get_instance.return_value.get_photo_by_id.side_effect = lambda pid: photos.get(pid)
            result = photo_commands(commands_instance)._run_show_photo("10", "20")

        assert result == ""
        mock_telegram.send_photo_album.assert_called_once()
        album_arg = mock_telegram.send_photo_album.call_args[0][0]
        assert len(album_arg) == 2
        assert album_arg[0] == (jpeg_a, "Photo A")
        assert album_arg[1] == (jpeg_b, "Photo B")

    def test_show_photo_multiple_ids_skips_missing(self, commands_instance):
        photo_10 = {"id": 10, "asset_guid": "GUID-10", "short_description": "Photo A"}
        jpeg_a = b"\xff\xd8jpeg_a"
        mock_telegram = Mock()

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("LOCAL-UUID-A", "IMG_A.HEIC", "DCIM/A", None, None, 0)
        mock_conn.cursor.return_value = mock_cursor

        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls, \
             patch("sqlite3.connect", return_value=mock_conn), \
             patch("integrations.photos.PhotoProcessor.PhotoProcessor._get_image_bytes",
                   return_value=jpeg_a), \
             patch("assistant.Notifier._notifier", mock_telegram):
            mock_db_cls.get_instance.return_value.get_photo_by_id.side_effect = lambda pid: photo_10 if pid == 10 else None
            result = photo_commands(commands_instance)._run_show_photo("10", "999")

        assert "skipped" in result
        mock_telegram.send_photo.assert_called_once_with(jpeg_a, caption="Photo A")

    def test_show_photo_multiple_ids_all_missing(self, commands_instance):
        # the Photos library is faked: unpatched, the album path opens the real one of the Mac it runs on
        with patch("integrations.photos.PhotosCommands.FloxyDatabase") as mock_db_cls, \
             patch("sqlite3.connect", return_value=Mock()):
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = None
            result = photo_commands(commands_instance)._run_show_photo("999", "888")

        assert "no valid photos found" in result

    def test_show_photo_multiple_ids_voice_to_voice_shows_first(self, commands_instance):
        commands_instance._current_profile = profile_for("VoiceToVoice")
        with _screen_config(**{"photos.show_on_screen": True}), \
             patch.object(photo_commands(commands_instance), '_run_show_single_photo', return_value="") as mock_single:
            result = photo_commands(commands_instance)._run_show_photo("10", "20", "30")

        mock_single.assert_called_once_with(10, route=None)
        assert result == ""


class TestShowPhotoOnScreen:
    @pytest.fixture(autouse=True)
    def screen(self):
        with _screen_config(**{"photos.show_on_screen": True, "photos.screen_prepare_command": "switch-tv"}):
            yield

    def test_prepare_command_nonzero_exit(self):
        mock_result = Mock(returncode=1, stdout="", stderr="device not found")
        with patch("subprocess.run", return_value=mock_result) as run:
            result = PhotosCommands._show_photo_on_screen("ABC-123", "Sunset")
        assert run.call_args.args[0] == "switch-tv" and run.call_args.kwargs["shell"] is True
        assert "exited with code 1" in result
        assert "device not found" in result

    def test_prepare_command_exception(self):
        with patch("subprocess.run", side_effect=TimeoutError("timed out")):
            result = PhotosCommands._show_photo_on_screen("ABC-123", "Sunset")
        assert "could not prepare the screen" in result
        assert "timed out" in result

    def test_osascript_nonzero_exit(self):
        ok_result = Mock(returncode=0, stdout="", stderr="")
        fail_result = Mock(returncode=1, stdout="", stderr="script error")
        with patch("subprocess.run", side_effect=[ok_result, fail_result]):
            result = PhotosCommands._show_photo_on_screen("ABC-123", "Sunset")
        assert "AppleScript exited with code 1" in result
        assert "script error" in result

    def test_osascript_exception(self):
        ok_result = Mock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", side_effect=[ok_result, TimeoutError("script hung")]):
            result = PhotosCommands._show_photo_on_screen("ABC-123", "Sunset")
        assert "could not open photo in Photos app" in result
        assert "script hung" in result

    def test_without_a_prepare_command_only_the_applescript_runs(self):
        with _screen_config(**{"photos.show_on_screen": True}):
            with patch("subprocess.run", return_value=Mock(returncode=0, stdout="", stderr="")) as run:
                result = PhotosCommands._show_photo_on_screen("ABC-123", "Sunset")
        assert result == ""
        assert run.call_count == 1 and run.call_args.args[0][0] == "osascript"


class TestSpokenPhotoWithoutAScreen:
    """A voice request on an install without `photos.show_on_screen` gets its photo in the chat."""

    def test_falls_back_to_the_chat(self, commands_instance):
        from assistant.ChannelProfile import VOICE_TO_VOICE
        commands_instance._current_profile = VOICE_TO_VOICE
        owner = photo_commands(commands_instance)
        with _screen_config(), \
             patch.object(owner, "_prepare_photo_send_payload", return_value=(MediaPayload(b"jpg", "cap", "uuid", "photo"), None)), \
             patch.object(owner, "_show_photo_on_screen") as screen, \
             patch("assistant.Notifier._notifier", Mock()) as bot:
            result = owner._run_show_single_photo(1)
        assert result == "" and not screen.called and bot.send_photo.called


@pytest.mark.unit
class TestPreparePhotoSendPayload:
    def test_photo_not_found_returns_not_found(self, commands_instance):
        with patch('integrations.photos.PhotosCommands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = None
            payload, err = photo_commands(commands_instance)._prepare_photo_send_payload(999)
        assert payload is None
        assert err == "not_found"

    def test_missing_asset_guid_returns_error(self, commands_instance):
        with patch('integrations.photos.PhotosCommands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = {
                "asset_guid": "",
                "short_description": "test",
            }
            payload, err = photo_commands(commands_instance)._prepare_photo_send_payload(1)
        assert payload is None
        assert err == "missing_asset_guid"

    def test_guid_not_found_in_photos_db(self, commands_instance):
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None

        with patch('integrations.photos.PhotosCommands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = {
                "asset_guid": "GUID-ABC",
                "short_description": "test",
            }
            with patch.object(PhotosCommands, '_fetch_photo_asset_row', return_value=None):
                payload, err = photo_commands(commands_instance)._prepare_photo_send_payload(1, cursor=mock_cursor)
        assert payload is None
        assert err == "guid_not_found:GUID-ABC"

    def test_image_extract_failed(self, commands_instance):
        mock_cursor = Mock()
        row = ("LOCAL-UUID", "IMG_001.jpg", "DCIM/100APPLE", "cloud-guid", "asset-guid", 0)

        with patch('integrations.photos.PhotosCommands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = {
                "asset_guid": "GUID-ABC",
                "short_description": "A nice photo",
            }
            with patch.object(PhotosCommands, '_fetch_photo_asset_row', return_value=row):
                with patch('integrations.photos.PhotoProcessor.PhotoProcessor') as mock_pp:
                    mock_pp._get_image_bytes.return_value = None
                    payload, err = photo_commands(commands_instance)._prepare_photo_send_payload(1, cursor=mock_cursor)
        assert payload is None
        assert err == "image_extract_failed"

    def test_happy_path_returns_payload(self, commands_instance):
        mock_cursor = Mock()
        row = ("LOCAL-UUID", "IMG_001.jpg", "DCIM/100APPLE", "cloud-guid", "asset-guid", 0)
        fake_bytes = b"\xff\xd8\xff\xe0"

        with patch('integrations.photos.PhotosCommands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = {
                "asset_guid": "GUID-ABC",
                "short_description": "A nice photo",
            }
            with patch.object(PhotosCommands, '_fetch_photo_asset_row', return_value=row):
                with patch('integrations.photos.PhotoProcessor.PhotoProcessor') as mock_pp:
                    mock_pp._get_image_bytes.return_value = fake_bytes
                    payload, err = photo_commands(commands_instance)._prepare_photo_send_payload(1, cursor=mock_cursor)
        assert err is None
        assert payload.data == fake_bytes
        assert payload.caption == "A nice photo"
        assert payload.uuid == "LOCAL-UUID"
        assert payload.kind == "photo"

    def _video_payload(self, commands_instance, downloader_result=None, downloader_error=None,
                       poster=b"jpg", poster_error=None):
        mock_cursor = Mock()
        row = ("LOCAL-UUID", "clip.mov", "0/00", "master-guid", "asset-guid", 15)
        with patch('integrations.photos.PhotosCommands.FloxyDatabase') as mock_db_cls, \
             patch.object(PhotosCommands, '_fetch_photo_asset_row', return_value=row), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader.get_instance') as dl:
            mock_db_cls.get_instance.return_value.get_photo_by_id.return_value = {
                "asset_guid": "GUID-VID", "short_description": "Frances sings",
                "photo_type": "video, slo-mo", "duration_seconds": 12.5,
            }
            dl.return_value.download_video_bytes = Mock(return_value=downloader_result, side_effect=downloader_error)
            dl.return_value.download_video_poster = Mock(return_value=poster, side_effect=poster_error)
            payload, err = photo_commands(commands_instance)._prepare_photo_send_payload(1, cursor=mock_cursor)
            return payload, err, dl.return_value

    def test_video_row_downloads_rendition_and_poster(self, commands_instance):
        payload, err, downloader = self._video_payload(commands_instance, downloader_result=(b"mp4", "video/mp4"))
        assert err is None
        assert payload == MediaPayload(b"mp4", "Frances sings", "LOCAL-UUID", "video", thumbnail=b"jpg")
        downloader.download_video_bytes.assert_called_once_with("master-guid", 15, max_bytes=PhotosCommands.TELEGRAM_VIDEO_MAX_BYTES,
                                                                cloud_asset_guid="asset-guid")
        downloader.download_video_poster.assert_called_once_with("master-guid", 15)

    def test_video_sent_without_thumbnail_when_poster_fails(self, commands_instance):
        payload, err, _ = self._video_payload(commands_instance, downloader_result=(b"mp4", "video/mp4"),
                                              poster=None, poster_error=RuntimeError("cdn down"))
        assert err is None
        assert payload == MediaPayload(b"mp4", "Frances sings", "LOCAL-UUID", "video")
        assert payload.thumbnail is None

    def test_video_too_large_for_telegram(self, commands_instance):
        from integrations.photos.ICloudPhotoDownloader import VideoTooLargeError
        payload, err, _ = self._video_payload(commands_instance, downloader_error=VideoTooLargeError(60_000_000, 50_000_000))
        assert payload is None
        assert err == "video_too_large"
        assert "too large" in PhotosCommands._format_single_photo_error(1, err)
        assert "too large" in PhotosCommands._format_album_photo_error(1, err)

    def test_video_download_failure(self, commands_instance):
        payload, err, _ = self._video_payload(commands_instance, downloader_result=None)
        assert payload is None
        assert err == "video_extract_failed"
        assert "could not download video 1" in PhotosCommands._format_single_photo_error(1, err)

    @pytest.mark.parametrize("exc", [NoVideoRenditionError(["rec-1"]), PhotoNotInCloudError("gone")])
    def test_video_with_no_rendition_or_cloud_record_is_reported_not_raised(self, commands_instance, exc):
        """The downloader raises for a clip iCloud cannot serve; show_photo must turn that
        into the per-photo error text instead of letting the whole command blow up."""
        payload, err, _ = self._video_payload(commands_instance, downloader_error=exc)
        assert payload is None
        assert err == "video_extract_failed"
        assert "video 1 could not be downloaded" == PhotosCommands._format_album_photo_error(1, err)


class TestShowVideo:
    def _video(self, commands_instance, notifier_mock, payloads):
        """payloads: photo_id -> (MediaPayload | None, err | None)."""
        with patch.object(photo_commands(commands_instance), '_prepare_photo_send_payload',
                          side_effect=lambda pid, cursor=None: payloads[pid]), \
             patch.object(photo_commands(commands_instance), '_open_photos_db_connection', return_value=Mock()), \
             patch("assistant.Notifier._notifier", notifier_mock):
            return photo_commands(commands_instance)._run_show_photo(*[str(pid) for pid in payloads])

    def test_single_video_uses_send_video(self, commands_instance):
        notifier_mock = Mock()
        result = self._video(commands_instance, notifier_mock,
                             {5: (MediaPayload(b"mp4", "Frances sings", "U", "video"), None)})
        assert result == ""
        notifier_mock.send_video.assert_called_once_with(b"mp4", caption="Frances sings", thumbnail=None)
        notifier_mock.send_photo.assert_not_called()

    def test_single_video_without_notifier_reports_missing_channel(self, commands_instance):
        notifier_mock = Mock()
        notifier_mock.send_video.return_value = False
        result = self._video(commands_instance, notifier_mock,
                             {5: (MediaPayload(b"mp4", "cap", "U", "video"), None)})
        assert "no notification channel configured" in result

    def test_mixed_album_sends_photos_as_album_then_videos(self, commands_instance):
        notifier_mock = Mock()
        result = self._video(commands_instance, notifier_mock, {
            1: (MediaPayload(b"a", "Photo A", "UA", "photo"), None),
            2: (MediaPayload(b"v", "Clip", "UV", "video"), None),
            3: (MediaPayload(b"b", "Photo B", "UB", "photo"), None),
        })
        assert result == ""
        notifier_mock.send_photo_album.assert_called_once_with([(b"a", "Photo A"), (b"b", "Photo B")])
        notifier_mock.send_video.assert_called_once_with(b"v", caption="Clip", thumbnail=None)

    def test_album_of_only_videos_sends_each(self, commands_instance):
        notifier_mock = Mock()
        result = self._video(commands_instance, notifier_mock, {
            1: (MediaPayload(b"v1", "Clip 1", "U1", "video"), None),
            2: (None, "video_too_large"),
            3: (MediaPayload(b"v3", "Clip 3", "U3", "video"), None),
        })
        assert result == "Sent 2 photos (skipped: video 2 is too large to send)"
        assert notifier_mock.send_video.call_count == 2
        notifier_mock.send_photo_album.assert_not_called()
        notifier_mock.send_photo.assert_not_called()
