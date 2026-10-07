import io
import json
from unittest.mock import Mock, patch, MagicMock

import pytest

from integrations.photos.ICloudPhotoDownloader import (
    ICloudPhotoDownloader,
    FALLBACK_PRIMARY_ZONE,
    PhotoNotInCloudError,
    VideoTooLargeError,
    NoVideoRenditionError,
    _RECORD_NOT_FOUND,
)
@pytest.fixture(autouse=True)
def reset_singleton():
    ICloudPhotoDownloader.reset_instance()
    yield
    ICloudPhotoDownloader.reset_instance()


@pytest.fixture
def downloader():
    dl = ICloudPhotoDownloader()
    dl._service_endpoint = "https://ckdatabase.icloud.com/database/1/com.apple.photos.cloud/production/private"
    dl._session = Mock()
    dl._params = {"dsid": "12345", "remapEnums": True}
    dl._primary_zone = {"zoneName": "PrimarySync", "zoneType": "REGULAR_CUSTOM_ZONE", "ownerRecordName": "_owner1"}
    dl._shared_zone = {"zoneName": "SharedSync-ABCDEF", "zoneType": "REGULAR_CUSTOM_ZONE", "ownerRecordName": "_owner2"}
    dl._ready = True
    return dl


# ---------------------------------------------------------------------------
# _zone_order
# ---------------------------------------------------------------------------

class TestZoneOrder:

    @pytest.mark.unit
    def test_shared_first_then_primary(self, downloader):
        order = downloader._zone_order()
        assert len(order) == 2
        assert order[0]["zoneName"].startswith("SharedSync")
        assert order[1]["zoneName"] == "PrimarySync"

    @pytest.mark.unit
    def test_no_shared_zone_returns_primary_only(self, downloader):
        downloader._shared_zone = None
        order = downloader._zone_order()
        assert len(order) == 1
        assert order[0]["zoneName"] == "PrimarySync"

# ---------------------------------------------------------------------------
# _discover_zones
# ---------------------------------------------------------------------------

class TestDiscoverZones:

    @pytest.mark.unit
    def test_discovers_primary_and_shared(self, downloader):
        downloader._primary_zone = None
        downloader._shared_zone = None
        response_data = {
            "zones": [
                {"zoneID": {"zoneName": "PrimarySync", "zoneType": "REGULAR_CUSTOM_ZONE", "ownerRecordName": "_o1"}},
                {"zoneID": {"zoneName": "SharedSync-ABC", "zoneType": "REGULAR_CUSTOM_ZONE", "ownerRecordName": "_o2"}},
            ]
        }
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response_data
        downloader._session.post.return_value = mock_resp

        downloader._discover_zones()

        assert downloader._primary_zone["zoneName"] == "PrimarySync"
        assert "SharedSync" in downloader._shared_zone["zoneName"]

    @pytest.mark.unit
    def test_skips_deleted_zones(self, downloader):
        downloader._primary_zone = None
        downloader._shared_zone = None
        response_data = {
            "zones": [
                {"zoneID": {"zoneName": "SharedSync-DEL"}, "deleted": True},
                {"zoneID": {"zoneName": "PrimarySync", "zoneType": "REGULAR_CUSTOM_ZONE"}},
            ]
        }
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response_data
        downloader._session.post.return_value = mock_resp

        downloader._discover_zones()

        assert downloader._shared_zone is None
        assert downloader._primary_zone["zoneName"] == "PrimarySync"

    @pytest.mark.unit
    def test_uses_fallback_when_primary_missing(self, downloader):
        downloader._primary_zone = None
        downloader._shared_zone = None
        response_data = {"zones": []}
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response_data
        downloader._session.post.return_value = mock_resp

        downloader._discover_zones()

        assert downloader._primary_zone == FALLBACK_PRIMARY_ZONE

        # Should not crash, primary fallback set if it was None
        # (it was already set in fixture, so this tests the HTTP error path)

    @pytest.mark.unit
    def test_exception_uses_fallback(self, downloader):
        downloader._primary_zone = None
        downloader._session.post.side_effect = Exception("network error")

        downloader._discover_zones()

        assert downloader._primary_zone == FALLBACK_PRIMARY_ZONE


# ---------------------------------------------------------------------------
# _lookup_download_url
# ---------------------------------------------------------------------------

class TestLookupDownloadUrl:

    def _make_response(self, fields, status_code=200, server_error=None):
        record = {"fields": fields}
        if server_error:
            record["serverErrorCode"] = server_error
        mock_resp = Mock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {"records": [record]}
        return mock_resp

    @pytest.mark.unit
    def test_prefers_med_res(self, downloader):
        fields = {
            "resJPEGMedRes": {"value": {"downloadURL": "https://cdn/med.jpg"}},
            "resJPEGFullRes": {"value": {"downloadURL": "https://cdn/full.jpg"}},
        }
        downloader._session.post.return_value = self._make_response(fields)
        zone = downloader._primary_zone

        url = downloader._lookup_download_url("GUID-123", zone)
        assert url == "https://cdn/med.jpg"

    @pytest.mark.unit
    def test_falls_back_to_full_res(self, downloader):
        fields = {
            "resJPEGFullRes": {"value": {"downloadURL": "https://cdn/full.jpg"}},
        }
        downloader._session.post.return_value = self._make_response(fields)

        url = downloader._lookup_download_url("GUID-123", downloader._primary_zone)
        assert url == "https://cdn/full.jpg"

    @pytest.mark.unit
    def test_falls_back_to_original_res(self, downloader):
        fields = {
            "resOriginalRes": {"value": {"downloadURL": "https://cdn/orig.jpg"}},
        }
        downloader._session.post.return_value = self._make_response(fields)

        url = downloader._lookup_download_url("GUID-123", downloader._primary_zone)
        assert url == "https://cdn/orig.jpg"

    @pytest.mark.unit
    def test_returns_none_when_no_download_url(self, downloader):
        fields = {"resJPEGMedRes": {"value": {}}}
        downloader._session.post.return_value = self._make_response(fields)

        url = downloader._lookup_download_url("GUID-123", downloader._primary_zone)
        assert url is None

    @pytest.mark.unit
    def test_returns_none_on_http_error(self, downloader):
        downloader._session.post.return_value = self._make_response({}, status_code=404)

        url = downloader._lookup_download_url("GUID-123", downloader._primary_zone)
        assert url is None

    @pytest.mark.unit
    def test_returns_sentinel_on_not_found(self, downloader):
        downloader._session.post.return_value = self._make_response(
            {}, server_error="NOT_FOUND"
        )

        result = downloader._lookup_download_url("GUID-123", downloader._primary_zone)
        assert result is _RECORD_NOT_FOUND

    @pytest.mark.unit
    def test_returns_none_on_other_server_error(self, downloader):
        downloader._session.post.return_value = self._make_response(
            {}, server_error="QUOTA_EXCEEDED"
        )

        url = downloader._lookup_download_url("GUID-123", downloader._primary_zone)
        assert url is None

    @pytest.mark.unit
    def test_returns_none_on_empty_records(self, downloader):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": []}
        downloader._session.post.return_value = mock_resp

        url = downloader._lookup_download_url("GUID-123", downloader._primary_zone)
        assert url is None


# ---------------------------------------------------------------------------
# _download_raw
# ---------------------------------------------------------------------------

class TestDownloadRaw:

    @pytest.mark.unit
    def test_success(self, downloader):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 500
        downloader._session.get.return_value = mock_resp

        result = downloader._download_raw("https://cdn/photo.jpg")
        assert result == b"x" * 500

    @pytest.mark.unit
    def test_rejects_tiny_response(self, downloader):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 50
        downloader._session.get.return_value = mock_resp

        assert downloader._download_raw("https://cdn/photo.jpg") is None

    @pytest.mark.unit
    def test_http_error(self, downloader):
        mock_resp = Mock()
        mock_resp.status_code = 403
        downloader._session.get.return_value = mock_resp

        assert downloader._download_raw("https://cdn/photo.jpg") is None



# ---------------------------------------------------------------------------
# download_photo_bytes — integration of the above
# ---------------------------------------------------------------------------

class TestDownloadPhotoBytes:

    @pytest.mark.unit
    def test_tries_shared_then_primary(self, downloader):
        from PIL import Image
        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        # SharedSync returns None, PrimarySync returns URL
        lookup_calls = []
        def side_effect(record_name, zone):
            lookup_calls.append(zone["zoneName"])
            if "SharedSync" in zone["zoneName"]:
                return None
            return "https://cdn/photo.jpg"

        downloader._lookup_download_url = Mock(side_effect=side_effect)

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = jpeg_bytes
        downloader._session.get.return_value = mock_resp

        result = downloader.download_photo_bytes("GUID-ABC")

        assert result is not None
        assert "SharedSync" in lookup_calls[0]
        assert lookup_calls[1] == "PrimarySync"

    @pytest.mark.unit
    def test_raises_not_in_cloud_when_all_zones_not_found(self, downloader):
        downloader._lookup_download_url = Mock(return_value=_RECORD_NOT_FOUND)

        with pytest.raises(PhotoNotInCloudError):
            downloader.download_photo_bytes("GUID-GONE")

    @pytest.mark.unit
    def test_returns_none_when_mixed_failure_and_not_found(self, downloader):
        call_count = [0]
        def side_effect(record_name, zone):
            call_count[0] += 1
            if call_count[0] == 1:
                return _RECORD_NOT_FOUND
            return None  # non-NOT_FOUND failure

        downloader._lookup_download_url = Mock(side_effect=side_effect)

        result = downloader.download_photo_bytes("GUID-MIX")
        assert result is None

    @pytest.mark.unit
    def test_returns_none_when_service_not_ready(self):
        dl = ICloudPhotoDownloader()
        with patch.object(dl, '_ensure_service', return_value=False):
            assert dl.download_photo_bytes("GUID-123") is None

    @pytest.mark.unit
    def test_resets_state_on_exception(self, downloader):
        downloader._lookup_download_url = Mock(side_effect=RuntimeError("boom"))
        result = downloader.download_photo_bytes("GUID-ERR")

        assert result is None
        assert downloader._ready is False
        assert downloader._session is None


# ---------------------------------------------------------------------------
# _ensure_service — PCS authorization
# ---------------------------------------------------------------------------

class TestEnsureService:

    def _make_mock_api(self):
        api = MagicMock()
        api.get_webservice_url.return_value = "https://ckdatabase.icloud.com"
        api.session = Mock()
        api.params = {"key": "val"}
        api.data = {"dsInfo": {"dsid": "12345"}}
        mock_zone_resp = Mock()
        mock_zone_resp.status_code = 200
        mock_zone_resp.json.return_value = {
            "zones": [{"zoneID": {"zoneName": "PrimarySync", "zoneType": "REGULAR_CUSTOM_ZONE"}}]
        }
        api.session.post.return_value = mock_zone_resp
        return api

    @pytest.mark.unit
    def test_pcs_called_during_init(self):
        api = self._make_mock_api()
        dl = ICloudPhotoDownloader()
        with patch("integrations.icloud.ICloudIntegration.ICloudIntegration") as mock_ic:
            mock_ic.get_instance.return_value.api = api
            result = dl._ensure_service()

        assert result is True
        api._request_pcs_for_service.assert_called_once_with("photos")

    @pytest.mark.unit
    def test_pcs_failure_continues_and_initializes(self):
        """PCS failure is non-fatal — init should proceed (non-ADP accounts get 422)."""
        api = self._make_mock_api()
        api._request_pcs_for_service.side_effect = Exception("Unprocessable Entity (422)")
        dl = ICloudPhotoDownloader()
        with patch("integrations.icloud.ICloudIntegration.ICloudIntegration") as mock_ic:
            mock_ic.get_instance.return_value.api = api
            result = dl._ensure_service()

        assert result is True
        assert dl._ready is True

    @pytest.mark.unit
    def test_missing_config_disables_api_and_stops_retrying(self):
        """Missing 'icloud' config raises on every get_instance() call (failed __init__
        is never cached) — the downloader must remember it and not re-attempt per photo."""
        from integrations.icloud.ICloudIntegration import iCloudConfigurationError
        dl = ICloudPhotoDownloader()
        with patch("integrations.icloud.ICloudIntegration.ICloudIntegration") as mock_ic:
            mock_ic.get_instance.side_effect = iCloudConfigurationError(
                "Missing 'icloud' section in configuration.")
            assert dl._ensure_service() is False
            assert dl._not_configured is True
            assert dl._ensure_service() is False
            mock_ic.get_instance.assert_called_once()

        # _reset() (session-level recovery) must not clear the config verdict
        dl._reset()
        assert dl._not_configured is True

    @pytest.mark.unit
    def test_auth_error_triggers_session_refresh_and_retry(self, downloader):
        """401 on CloudKit should refresh the iCloud session and retry once."""
        from PIL import Image
        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        call_count = [0]
        def do_download_side_effect(guid):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Unauthorized (401)")
            # Second call succeeds
            return jpeg_bytes

        downloader._do_download = Mock(side_effect=do_download_side_effect)

        with patch("integrations.icloud.ICloudIntegration.ICloudIntegration") as mock_ic:
            mock_instance = mock_ic.get_instance.return_value
            mock_instance.refresh_connection.return_value = None
            mock_instance._api = Mock()
            mock_instance._api.requires_2fa = False
            with patch.object(downloader, '_ensure_service', return_value=True):
                result = downloader.download_photo_bytes("GUID-AUTH")

        assert result == jpeg_bytes
        assert call_count[0] == 2

    @pytest.mark.unit
    def test_auth_error_with_2fa_pending_kicks_auto_2fa_flow(self, downloader):
        """When 2FA is pending, a refresh would race the 2FA flow — but silently
        giving up left photos broken for a day (2026-08-19/20). The downloader
        must kick the automatic SMS 2FA flow instead."""
        downloader._do_download = Mock(side_effect=Exception("Unauthorized (401)"))

        with patch("integrations.icloud.ICloudIntegration.ICloudIntegration") as mock_ic:
            mock_instance = mock_ic.get_instance.return_value
            mock_instance._api = Mock()
            mock_instance._api.requires_2fa = True
            with patch.object(downloader, '_ensure_service', return_value=True):
                result = downloader.download_photo_bytes("GUID-2FA-PENDING")

        assert result is None
        mock_instance.on_needs_2fa.assert_called_once()
        mock_instance.refresh_connection.assert_not_called()

    @pytest.mark.unit
    def test_session_refresh_failure_is_a_system_note(self):
        """b4308ef7: the failure is a history row from "icloud_photos" (so the assistant can
        explain the broken downloads on its next turn), pushed to the chats as such."""
        from assistant import SystemNote
        history = Mock()
        mock_bot = Mock()
        mock_bot.send_message.return_value = True
        SystemNote.set_history(history)
        try:
            with patch("assistant.Notifier._notifier", mock_bot):
                ICloudPhotoDownloader._notify_session_refresh_failed(Exception("2FA required"))
        finally:
            SystemNote.set_history(None)

        role, msg, _ts = history.add_context_message.call_args[0]
        assert role == "system"
        assert history.add_context_message.call_args[1] == {"source": "icloud_photos"}
        assert "session refresh failed" in msg and "2FA required" in msg
        mock_bot.send_message.assert_called_once_with(msg, disable_notification=False, in_history=True)


# ---------------------------------------------------------------------------
# Video renditions
# ---------------------------------------------------------------------------

def _rendition(prefix: str, size: int, file_type: str = "public.mpeg-4") -> dict:
    return {
        f"{prefix}Res": {"value": {"downloadURL": f"https://cdn/{prefix}.bin", "size": size}},
        f"{prefix}FileType": {"value": file_type},
    }


class TestDownloadVideoPoster:

    def _lookup_response(self, fields):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"records": [{"fields": fields}]}
        return resp

    def _poster(self, name):
        return {name: {"value": {"downloadURL": f"https://cdn/{name}.jpg", "size": 44110}}}

    @pytest.mark.unit
    def test_prefers_thumb_and_shrinks_to_telegram_size(self, downloader):
        downloader._session.post.return_value = self._lookup_response(
            {**self._poster("resJPEGMedRes"), **self._poster("resJPEGThumbRes")})
        get_resp = Mock()
        get_resp.status_code = 200
        get_resp.content = b"\xff\xd8" + b"x" * 200
        downloader._session.get.return_value = get_resp

        with patch('integrations.photos.ICloudPhotoDownloader.resize_to_jpeg',
                   return_value=b"small-jpeg") as resize:
            assert downloader.download_video_poster("GUID-VID") == b"small-jpeg"
        downloader._session.get.assert_called_once_with("https://cdn/resJPEGThumbRes.jpg", timeout=60, stream=False)
        resize.assert_called_once_with(get_resp.content, max_dim=320)
        body = json.loads(downloader._session.post.call_args.kwargs["data"])
        assert body["desiredKeys"] == ["resJPEGThumbRes", "resJPEGMedRes"]

    @pytest.mark.unit
    def test_falls_back_to_medium_poster(self, downloader):
        downloader._session.post.return_value = self._lookup_response(self._poster("resJPEGMedRes"))
        get_resp = Mock()
        get_resp.status_code = 200
        get_resp.content = b"\xff\xd8" + b"x" * 200
        downloader._session.get.return_value = get_resp
        with patch('integrations.photos.ICloudPhotoDownloader.resize_to_jpeg',
                   return_value=b"small-jpeg"):
            assert downloader.download_video_poster("GUID-VID") == b"small-jpeg"
        downloader._session.get.assert_called_once_with("https://cdn/resJPEGMedRes.jpg", timeout=60, stream=False)

    @pytest.mark.unit
    def test_no_poster_returns_none(self, downloader):
        downloader._session.post.return_value = self._lookup_response(_rendition("resVidMed", 10))
        assert downloader.download_video_poster("GUID-VID") is None
        downloader._session.get.assert_not_called()

    @pytest.mark.unit
    def test_not_found_raises(self, downloader):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"records": [{"serverErrorCode": "NOT_FOUND"}]}
        downloader._session.post.return_value = resp
        with pytest.raises(PhotoNotInCloudError):
            downloader.download_video_poster("GUID-GONE")


class TestPickVideoRendition:

    @pytest.mark.unit
    def test_prefers_medium_when_it_fits(self):
        fields = {**_rendition("resVidMed", 4_000_000), **_rendition("resVidSmall", 1_000_000)}
        prefix, url, size, mime = ICloudPhotoDownloader._pick_video_rendition(fields, 20_000_000)
        assert (prefix, url, size, mime) == ("resVidMed", "https://cdn/resVidMed.bin", 4_000_000, "video/mp4")

    @pytest.mark.unit
    def test_falls_back_to_small_when_medium_too_big(self):
        fields = {**_rendition("resVidMed", 26_000_000), **_rendition("resVidSmall", 8_000_000)}
        prefix, _, size, _ = ICloudPhotoDownloader._pick_video_rendition(fields, 20_000_000)
        assert (prefix, size) == ("resVidSmall", 8_000_000)

    @pytest.mark.unit
    def test_raises_too_large_with_smallest_size(self):
        fields = {**_rendition("resVidMed", 140_000_000), **_rendition("resVidSmall", 41_000_000)}
        with pytest.raises(VideoTooLargeError) as exc:
            ICloudPhotoDownloader._pick_video_rendition(fields, 20_000_000)
        assert exc.value.smallest_size == 41_000_000
        assert exc.value.max_bytes == 20_000_000

    @pytest.mark.unit
    def test_large_rendition_is_the_last_resort(self):
        """A 2 ms clip's master carries only resVidLarge (a 230 kB MOV) — take it when it fits."""
        fields = _rendition("resVidLarge", 230_000, file_type="com.apple.quicktime-movie")
        prefix, url, size, mime = ICloudPhotoDownloader._pick_video_rendition(fields, 20_000_000)
        assert (prefix, size, mime) == ("resVidLarge", 230_000, "video/quicktime")

    @pytest.mark.unit
    def test_large_rendition_not_preferred_over_medium(self):
        fields = {**_rendition("resVidLarge", 2_000_000), **_rendition("resVidMed", 4_000_000)}
        assert ICloudPhotoDownloader._pick_video_rendition(fields, 20_000_000)[0] == "resVidMed"

    @pytest.mark.unit
    def test_none_when_record_has_no_video_renditions(self):
        fields = {"resJPEGMedRes": {"value": {"downloadURL": "https://cdn/poster.jpg", "size": 100}}}
        assert ICloudPhotoDownloader._pick_video_rendition(fields, 20_000_000) is None

    @pytest.mark.unit
    def test_quicktime_file_type_maps_to_mov_mime(self):
        fields = _rendition("resVidMed", 1_000, file_type="com.apple.quicktime-movie")
        assert ICloudPhotoDownloader._pick_video_rendition(fields, 20_000_000)[3] == "video/quicktime"

    @pytest.mark.unit
    def test_unknown_file_type_defaults_to_mp4(self):
        fields = {"resVidMedRes": {"value": {"downloadURL": "https://cdn/x", "size": 10}}}
        assert ICloudPhotoDownloader._pick_video_rendition(fields, 20_000_000)[3] == "video/mp4"


class TestDownloadVideoBytes:

    def _lookup_response(self, fields):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"records": [{"fields": fields}]}
        return resp

    @pytest.mark.unit
    def test_downloads_chosen_rendition_untouched(self, downloader):
        fields = {**_rendition("resVidMed", 4_000_000), **_rendition("resVidSmall", 1_000_000)}
        downloader._session.post.return_value = self._lookup_response(fields)
        raw = b"\x00\x00\x00\x1cftypmp42" + b"x" * 200
        get_resp = Mock()
        get_resp.status_code = 200
        get_resp.iter_content.return_value = [raw[:100], raw[100:]]
        downloader._session.get.return_value = get_resp

        result = downloader.download_video_bytes("GUID-VID")

        assert result == (raw, "video/mp4")
        downloader._session.get.assert_called_once_with("https://cdn/resVidMed.bin", timeout=120, stream=True)
        body = json.loads(downloader._session.post.call_args.kwargs["data"])
        assert "resVidMedRes" in body["desiredKeys"]
        assert "resOriginalRes" not in body["desiredKeys"]

    @pytest.mark.unit
    def test_too_large_propagates(self, downloader):
        fields = {**_rendition("resVidMed", 140_000_000), **_rendition("resVidSmall", 41_000_000)}
        downloader._session.post.return_value = self._lookup_response(fields)

        with pytest.raises(VideoTooLargeError):
            downloader.download_video_bytes("GUID-LONG", max_bytes=20_000_000)
        downloader._session.get.assert_not_called()
        assert downloader._ready is True

    @pytest.mark.unit
    def test_not_found_in_every_zone_raises(self, downloader):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"records": [{"serverErrorCode": "NOT_FOUND"}]}
        downloader._session.post.return_value = resp

        with pytest.raises(PhotoNotInCloudError):
            downloader.download_video_bytes("GUID-GONE")

    @pytest.mark.unit
    def test_shared_zone_not_found_then_primary_hit(self, downloader):
        not_found = Mock()
        not_found.status_code = 200
        not_found.json.return_value = {"records": [{"serverErrorCode": "NOT_FOUND"}]}
        downloader._session.post.side_effect = [not_found, self._lookup_response(_rendition("resVidMed", 10))]
        get_resp = Mock()
        get_resp.status_code = 200
        get_resp.iter_content.return_value = [b"v" * 200]
        downloader._session.get.return_value = get_resp

        assert downloader.download_video_bytes("GUID-PRIMARY") == (b"v" * 200, "video/mp4")
        zones = [json.loads(c.kwargs["data"])["zoneID"]["zoneName"] for c in downloader._session.post.call_args_list]
        assert zones[0].startswith("SharedSync") and zones[1] == "PrimarySync"

    @pytest.mark.unit
    def test_no_rendition_on_master_or_asset_raises(self, downloader):
        downloader._session.post.return_value = self._lookup_response({})

        with pytest.raises(NoVideoRenditionError) as exc:
            downloader.download_video_bytes("GUID-NOVID", cloud_asset_guid="ASSET-NOVID")

        assert exc.value.record_names == ["GUID-NOVID", "ASSET-NOVID"]
        looked_up = [json.loads(c.kwargs["data"])["records"][0]["recordName"]
                     for c in downloader._session.post.call_args_list]
        # Each record is tried in both zones (an empty hit does not end the search) before the next.
        assert looked_up == ["GUID-NOVID", "GUID-NOVID", "ASSET-NOVID", "ASSET-NOVID"]
        downloader._session.get.assert_not_called()
        assert downloader._ready is True

    @pytest.mark.unit
    def test_no_rendition_without_asset_guid_raises_after_master_only(self, downloader):
        downloader._session.post.return_value = self._lookup_response({})

        with pytest.raises(NoVideoRenditionError):
            downloader.download_video_bytes("GUID-NOVID")

        looked_up = {json.loads(c.kwargs["data"])["records"][0]["recordName"]
                     for c in downloader._session.post.call_args_list}
        assert looked_up == {"GUID-NOVID"}

    @pytest.mark.unit
    def test_slomo_renditions_come_from_the_asset_record(self, downloader):
        """A slo-mo master carries only the original and the posters; the renditions are on the asset."""
        master_only_original = self._lookup_response({"resOriginalRes": {"value": {"downloadURL": "https://cdn/orig", "size": 800_000_000}}})
        downloader._session.post.side_effect = [
            master_only_original,
            self._lookup_response({**_rendition("resVidMed", 4_000_000), **_rendition("resVidSmall", 1_000_000)}),
        ]
        get_resp = Mock()
        get_resp.status_code = 200
        get_resp.iter_content.return_value = [b"s" * 200]
        downloader._session.get.return_value = get_resp

        assert downloader.download_video_bytes("MASTER-SLOMO", cloud_asset_guid="ASSET-SLOMO") == (b"s" * 200, "video/mp4")

        looked_up = [json.loads(c.kwargs["data"])["records"][0]["recordName"]
                     for c in downloader._session.post.call_args_list]
        assert looked_up == ["MASTER-SLOMO", "ASSET-SLOMO"]
        downloader._session.get.assert_called_once_with("https://cdn/resVidMed.bin", timeout=120, stream=True)

    @pytest.mark.unit
    def test_asset_guid_equal_to_master_is_not_looked_up_twice(self, downloader):
        downloader._session.post.return_value = self._lookup_response({})

        with pytest.raises(NoVideoRenditionError) as exc:
            downloader.download_video_bytes("SAME-GUID", cloud_asset_guid="SAME-GUID")

        assert exc.value.record_names == ["SAME-GUID"]
        assert downloader._session.post.call_count == 2  # one lookup per zone, no second record

    @pytest.mark.unit
    def test_transient_lookup_failure_on_asset_record_returns_none(self, downloader):
        failed = Mock()
        failed.status_code = 503
        downloader._session.post.side_effect = [self._lookup_response({}), failed]

        assert downloader.download_video_bytes("MASTER-X", cloud_asset_guid="ASSET-X") is None

    @pytest.mark.unit
    def test_empty_record_in_shared_zone_falls_through_to_primary(self, downloader):
        """PrimarySync keeps a stub of a shared-library record with none of the desired keys —
        and vice versa — so an empty hit must not stop the search."""
        downloader._session.post.side_effect = [
            self._lookup_response({}),
            self._lookup_response(_rendition("resVidSmall", 10)),
        ]
        get_resp = Mock()
        get_resp.status_code = 200
        get_resp.iter_content.return_value = [b"p" * 200]
        downloader._session.get.return_value = get_resp

        assert downloader.download_video_bytes("GUID-STUB") == (b"p" * 200, "video/mp4")
        zones = [json.loads(c.kwargs["data"])["zoneID"]["zoneName"] for c in downloader._session.post.call_args_list]
        assert zones[0].startswith("SharedSync") and zones[1] == "PrimarySync"
