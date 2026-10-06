from dataclasses import dataclass
import os
import queue
import re
import threading
import time
import urllib.parse
from collections import deque
from typing import Callable, Optional

from assistant.integration.integrations.email.CompositeEmailAccess import CompositeEmailAccess
from assistant.integration.integrations.email.GmailIntegration import GmailAccess
from assistant.integration.integrations.email.EmailProvider import NotConfiguredEmail
from assistant.integration.integrations.email.ICloudEmailIntegration import ICloudEmailAccess
from assistant.integration.integrations.WeatherIntegration import WeatherIntegration
from assistant.integration.integrations.home import HomeAutomation as home
from assistant.tools.WebAccess import WebAccess
from assistant.agents.AutonomousActionManager import AutonomousActionManager
from . import Toolkits, ToolkitPreloader
from assistant import MemoryFile, Procedures, WatchList
from common.logging import trace, info, note, error, fine
from assistant.integration.integrations.MusicIntegration import MusicManager
from assistant.integration.integrations.CalDAVIntegration import CalDAVIntegration
from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
from common.ConfigManager import ConfigManager
from common import ConfigSchema, ConfigStore
from common.ConfigStore import ConfigStoreError
from assistant.ChannelProfile import AUTONOMOUS, PHOTO_TARGET_KITCHEN_TV, SOURCE_CONSOLE
from assistant.Notifier import SecretRejected, notifier
from assistant.SystemNote import system_note
from assistant.Telemetry import Telemetry
from .CommandEcho import describe_completion
from assistant.integration.integrations.RemindersIntegration import add_reminder_item
from assistant.integration.integrations.TeslaIntegration import TeslaIntegration, TeslaAPIError
from assistant.integration.integrations.ICloudIntegration import iCloudAPI
from assistant.integration.integrations.cameras.ArloCameraIntegration import ArloCamera
from assistant.integration.integrations.XIntegration import XIntegration
from assistant.integration.integrations.cameras.RingIntegration import RingIntegration
from common.FloxyDatabase import FloxyDatabase
from common.vault.Vault import Vault, VaultError, DuplicateSecret, KINDS


def normalize_llm_escapes(text: str) -> str:
    """Turn the literal escape sequences the LLM emits (backslash + n/r/t) into
    real whitespace. Single source of truth so the approval preview and the
    delivered message agree — see _gate_sensitive_command."""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


# noinspection PyMethodMayBeStatic
# Returned to the LLM when a photo/report had nowhere to go (NullNotifier installed).
NO_NOTIFICATION_CHANNEL = "failed: no notification channel configured (no chat interface running), nothing was delivered to the user."

# What server_logs() hands the LLM: the tail of the main log (cwd-relative, the server runs from AIServer/).
SERVER_LOG_PATH = "log/floxy.out"
SERVER_LOG_TAIL_LINES = 400


@dataclass(frozen=True)
class MediaPayload:
    """One library asset ready to be sent: JPEG bytes for a still, an MP4 rendition for a video."""
    data: bytes
    caption: str
    uuid: str
    kind: str  # "photo" | "video"
    thumbnail: Optional[bytes] = None  # small JPEG poster frame, videos only


class Commands:
    """Implements every `cmd_*` command the LLM can invoke, from lights to email to photos.

    The command name list handed to the parser is derived from the `cmd_` method names;
    SENSITIVE_COMMANDS go through the notifier's approval gate before running.
    """
    SENSITIVE_COMMANDS = frozenset({"send_imessage", "send_email", "restart_server"})
    # What the approval prompt shows under the title when the command has no parameters
    # to preview — the user should know what saying yes does before the screen goes blank.
    APPROVAL_NOTES = {
        "restart_server": "Pulls the latest code from git (the release branch, or master when "
                          "developer_release is true) and relaunches the server. "
                          "The console goes blank for a moment and reconnects by itself.",
    }
    # Commands whose handler accepts multiple positional arguments. The dispatcher
    # splats the parsed parameter list (`f(*params)`); the handler decides how to
    # interpret arity (e.g. send_imessage merges trailing tokens into the body so
    # under-quoted LLM output like `send_imessage(John, Hi, friend)` still works).
    # Anything not in this set gets `f(params[0] if params else "")` for compat
    # with the historical single-string signature.
    MULTI_PARAM_COMMANDS = frozenset({
        "send_imessage", "send_email", "dim", "play_music",
        "sonos_enqueue_and_play", "show_photo",
        "search_photos", "search_history", "show_history",
        "create_event", "browse_inspect", "browse_click_at", "browse_wait", "move_email", "latest_email", "vault_add",
        "vault_request", "config_set", "load_toolkit", "schedule_wakeup",
        "load_watch", "save_watch", "delete_watch", "load_calendar",
        "load_procedure", "save_procedure", "delete_procedure",
    })
    # Side-effect-free commands that the prompt test suite still runs: a toolkit the LLM
    # never receives would fail every expectation that depends on its reference.
    PURE_COMMANDS = frozenset({"load_toolkit", "load_procedure", "load_watch", "load_calendar"})
    AI_MESSAGE_SUFFIX = "[Sent with the help of AI 🤖]"

    # Integrations are looked up per call: get_instance() is the cache, so a restarted
    # integration (restart_integration) is what the next command uses.
    @staticmethod
    def _get_web_access() -> WebAccess:
        return WebAccess.get_instance()

    # Built by the first music command when Sonos & Spotify is disabled — never at startup,
    # where it would start discovering speakers on a network nobody asked it to look at.
    @staticmethod
    def _get_music_manager() -> MusicManager:
        return MusicManager.get_instance()

    @property
    def music_manager(self) -> MusicManager:
        return self._get_music_manager()

    @staticmethod
    def _get_x_integration() -> XIntegration:
        return XIntegration.get_instance()

    @property
    def web_access(self) -> WebAccess:
        return self._get_web_access()

    @property
    def _email_provider(self):
        return self._create_email_provider()

    false_positive_timestamps = deque(maxlen=10)
    # Installed by the launcher when the voice interface is running
    # (interfaces.voice.microphones.FalsePositiveFlood.report_false_positive_flood);
    # None in a text-only process.
    false_positive_reporter: Callable[[], None] | None = None

    @staticmethod
    def _home_name(name: str) -> str:
        # The prompt shows every "(Room)" in a device or action name as "[Room]" and the LLM
        # copies that form back; the backend gets the raw name (HomeAutomation.from_display).
        return home.HomeAutomation.from_display(name.strip())

    def __init__(self, cp):
        self.command_processor = cp
        # Set by CommandProcessor before each run: the policy of the channel that asked
        # and the RequestContext of the request (None for autonomous runs).
        self._current_profile = AUTONOMOUS
        self._current_context = None
        # The notifier's handle for the running command's chat echo; background_command
        # takes it along so the background worker can complete the echo when the work ends.
        self._current_echo = None
        # Background batch state — see the "background commands" section below.
        # bg_lock serializes submit / finish / publish so commands running on several
        # lane threads can't race the batch bookkeeping.
        self.bg_lock = threading.Lock()
        self._bg_outstanding = 0
        self._bg_seq = 0
        self._bg_results: dict[int, object] = {}
        self.bg_results_contain_error = False
        self._bg_batch_started = 0.0
        self._bg_batch_count = 0
        self._bg_busy_seconds = 0.0
        self._bg_idle = threading.Event()
        self._bg_idle.set()
        # One queue + at most one drain thread per serial lane (browser, config).
        self._serial_queues: dict[str, queue.Queue] = {}
        self._serial_workers: dict[str, threading.Thread] = {}

    @staticmethod
    def _create_email_provider():
        """The monitored singletons (their short-id caches must survive between a listing
        and a read), wrapped per call so a restarted or newly configured account is used."""
        config = ConfigManager.get_instance()
        providers = []
        if config.get("email.address", ""):
            providers.append(GmailAccess.get_instance())
        if config.get("icloud_email.address", ""):
            providers.append(ICloudEmailAccess.get_instance())
        if len(providers) > 1:
            return CompositeEmailAccess(providers)
        if providers:
            return providers[0]
        trace("No email account configured — mail commands answer with the setup hint")
        return NotConfiguredEmail()

    def cmd_weather(self, parameter_string: str):
        self.background_command(WeatherIntegration.fetch_weather_forecast, False)

    def cmd_home_automation_logs(self, parameter_string: str):
        self.background_command(self._fetch_home_automation_logs, None)

    @staticmethod
    def _fetch_home_automation_logs(_params) -> str:
        logs = home.active().error_logs()
        info(f"Home automation logs fetched: {len(logs)} chars")
        return f"# Home Automation Error Logs\n{logs}"

    def cmd_server_logs(self, parameter_string: str):
        self.background_command(self._fetch_server_logs, None)

    @staticmethod
    def _fetch_server_logs(_params) -> str:
        try:
            # The log is redirected process output; a stray non-UTF-8 byte must not take the
            # assistant's own diagnostics down with it.
            with open(SERVER_LOG_PATH, "r", errors="replace") as f:
                lines = f.readlines()
        except FileNotFoundError:
            error(f"Server log {SERVER_LOG_PATH} not found")
            return "# Keep AI Server Logs\nLog file not found."
        tail = "".join(lines[-SERVER_LOG_TAIL_LINES:])
        tail = re.sub(r'\x1b\[[0-9;]*m', '', tail)
        info(f"Server logs fetched: last {min(len(lines), SERVER_LOG_TAIL_LINES)} of {len(lines)} lines, {len(tail)} chars")
        return f"# Keep AI Server Logs\n{tail}"

    def cmd_open_url(self, parameter_string: str):
        self.background_command(self.web_access.fetch_webpage, parameter_string, lane=self.BROWSER_LANE)

    def cmd_google(self, parameter_string: str):
        self.background_command(self.web_access.search, parameter_string)

    def cmd_search_x(self, parameter_string: str):
        self.background_command(self._get_x_integration().search, parameter_string)

    def cmd_browse_open(self, parameter_string: str):
        self.background_command(self.web_access.browse_open, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_act(self, parameter_string: str):
        self.background_command(self.web_access.browse_act, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_key(self, parameter_string: str):
        self.background_command(self.web_access.browse_key, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_eval(self, parameter_string: str):
        self.background_command(self.web_access.browse_eval, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_inspect(self, *args: str):
        # Accepts either a single quoted spec ("3,href") or two positional args
        # (3, href). Re-join with comma so the underlying parser sees a single
        # spec string in the format it expects.
        spec = ",".join(a.strip() for a in args)
        self.background_command(self.web_access.browse_inspect, spec, lane=self.BROWSER_LANE)

    def cmd_browse_snapshot(self, parameter_string: str):
        self.background_command(self.web_access.browse_snapshot, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_click_at(self, *args: str):
        # "412, 583" arrives as two positional args — re-join so WebAccess parses one spec.
        spec = ",".join(a.strip() for a in args)
        self.background_command(self.web_access.browse_click_at, spec, lane=self.BROWSER_LANE)

    def cmd_browse_wait(self, *args: str):
        # `browse_wait("Continue|callback", 45)` arrives as two positional args — re-join so
        # WebAccess parses one "<pattern>, <seconds>" spec (a comma inside the pattern survives).
        spec = ", ".join(a.strip() for a in args)
        self.background_command(self.web_access.browse_wait, spec, lane=self.BROWSER_LANE)

    def cmd_browse_network(self, parameter_string: str):
        self.background_command(self.web_access.browse_network, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_extract(self, parameter_string: str):
        self.background_command(self.web_access.browse_extract, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_screenshot(self, parameter_string: str):
        self.background_command(self.web_access.browse_screenshot, parameter_string, lane=self.BROWSER_LANE)

    def cmd_send_screenshot(self, parameter_string: str):
        route = self._request_route()
        self.background_command(lambda caption: self._run_send_screenshot(caption, route=route),
                                parameter_string, lane=self.BROWSER_LANE)

    def _request_route(self) -> Optional[dict]:
        """The chat a request-driven send answers to, as the notifier's source/conversation
        kwargs; None when no request is being processed (the autonomous check), which the
        notifier takes as a proactive push. Captured on the command worker: by the time a
        background command runs, another request may have replaced _current_context."""
        ctx = self._current_context
        source = getattr(ctx, "source", None)
        if source is None:
            return None
        return {"source": source, "conversation": getattr(ctx, "conversation", None)}

    def _run_send_screenshot(self, caption: str, route: Optional[dict] = None) -> str:
        if not self._current_profile.screenshot_allowed:
            return "send_screenshot failed: not available in voice mode (user has no screen to view it on)."

        png_bytes, path = self.web_access.get_last_screenshot()
        if not png_bytes:
            return "send_screenshot failed: no screenshot available. Call browse_screenshot() first."

        from assistant.tools.ImageAnalyzer import resize_to_jpeg
        jpeg_bytes = resize_to_jpeg(png_bytes, max_dim=2048)
        if jpeg_bytes is None:
            note("send_screenshot: resize failed, sending original PNG")
            jpeg_bytes = png_bytes

        caption = caption.strip()
        if not notifier().send_photo(jpeg_bytes, caption=caption, **(route or {})):
            return NO_NOTIFICATION_CHANNEL
        info(f"Sent screenshot ({path}, {len(jpeg_bytes)} bytes) to notifier with caption: {caption!r}")
        return ""

    def cmd_browse_close(self, parameter_string: str):
        self.fire_and_forget(self.web_access.browse_close, parameter_string)

    _UNREAD_TOKENS = frozenset({"unread", "true", "1", "yes", "unread_only"})

    def cmd_latest_email(self, folder: str = "", unread_only: str = ""):
        unread = str(unread_only).strip().strip('"').strip("'").lower() in self._UNREAD_TOKENS
        self.background_command(self._email_provider.fetch_latest, [folder, unread])

    def cmd_search_email(self, parameter_string: str):
        self.background_command(self._email_provider.search, parameter_string)

    def cmd_read_email(self, parameter_string: str):
        self.background_command(self._email_provider.read, parameter_string)

    def cmd_move_email(self, email_id: str, dest_folder: str):
        self.background_command(self._email_provider.move, [email_id, dest_folder])

    def cmd_list_folders(self, parameter_string: str):
        self.background_command(self._email_provider.list_folders, parameter_string)

    def cmd_schedule_wakeup(self, *args: str):
        """schedule_wakeup(in_minutes, reason): the reason travels with the wake-up so the check
        it triggers knows what it is for even after the history window has moved on. Unquoted
        reasons split on their commas; re-joining them keeps the whole sentence."""
        if not args:
            return "schedule_wakeup failed: expected 'in_minutes, reason' parameters"
        minutes = float(args[0]) + 0.1  # just AFTER the wanted time, otherwise the AI may see it as not yet due
        reason = ", ".join(a for a in args[1:]).strip()
        if not reason:
            info("schedule_wakeup: the LLM gave no reason — the wake-up will have to find its purpose in the history")
        AutonomousActionManager.get_instance().schedule(minutes, reason)

    def cmd_caffeinate(self, parameter_string: str):
        AutonomousActionManager.get_instance().caffeinate()

    @staticmethod
    def _get_tesla_api() -> TeslaIntegration:
        return TeslaIntegration.get_instance()

    def _run_tesla_command(self, tesla_method, parameter_string: str) -> str:
        try:
            tesla_method(parameter_string if parameter_string else None)
            return ""
        except TeslaAPIError as e:
            return f"Tesla command failed: {e}"

    def cmd_tesla_max_defrost(self, parameter_string: str):
        self.background_command(
            lambda p: self._run_tesla_command(self._get_tesla_api().set_max_defrost, p),
            parameter_string
        )

    def cmd_tesla_air_con_on(self, parameter_string: str):
        self.background_command(
            lambda p: self._run_tesla_command(self._get_tesla_api().start_air_conditioning, p),
            parameter_string
        )

    def cmd_tesla_air_con_off(self, parameter_string: str):
        self.background_command(
            lambda p: self._run_tesla_command(self._get_tesla_api().stop_air_conditioning, p),
            parameter_string
        )

    def cmd_tesla_open_trunk(self, parameter_string: str):
        self.background_command(
            lambda p: self._run_tesla_command(self._get_tesla_api().open_trunk, p),
            parameter_string
        )

    def cmd_show_photo(self, *args: str):
        route = self._request_route()
        self.background_command(lambda *ids: self._run_show_photo(*ids, route=route), list(args))

    def _run_show_photo(self, *photo_id_strs: str, route: Optional[dict] = None) -> str:
        try:
            photo_ids = [int(x.strip()) for x in photo_id_strs if x.strip()]
        except ValueError:
            joined = ", ".join(photo_id_strs)
            return f"show_photo failed: '{joined}' is not a valid photo ID"

        if not photo_ids:
            return "show_photo failed: no photo IDs provided"

        if len(photo_ids) == 1:
            return self._run_show_single_photo(photo_ids[0], route=route)

        return self._run_show_photo_album(photo_ids, route=route)

    def _run_show_single_photo(self, photo_id: int, route: Optional[dict] = None) -> str:
        payload, prep_error = self._prepare_photo_send_payload(photo_id)
        if prep_error:
            return f"show_photo failed: {self._format_single_photo_error(photo_id, prep_error)}"

        if self._current_profile.photo_target == PHOTO_TARGET_KITCHEN_TV:
            return self._show_photo_on_kitchen_tv(payload.uuid, payload.caption)

        if payload.kind == "video":
            sent = notifier().send_video(payload.data, caption=payload.caption, thumbnail=payload.thumbnail,
                                         **(route or {}))
        else:
            sent = notifier().send_photo(payload.data, caption=payload.caption, **(route or {}))
        if not sent:
            return NO_NOTIFICATION_CHANNEL
        info(f"Sent {payload.kind} {photo_id} to notifier: {payload.caption}")
        return ""

    def _run_show_photo_album(self, photo_ids: list[int], route: Optional[dict] = None) -> str:
        if self._current_profile.photo_target == PHOTO_TARGET_KITCHEN_TV:
            info(f"{self._current_profile.name}: showing first photo only from album of {len(photo_ids)} (single TV frame)")
            return self._run_show_single_photo(photo_ids[0], route=route)

        album_photos: list[tuple[bytes, str]] = []
        videos: list[MediaPayload] = []
        errors = []

        conn = None
        try:
            conn = self._open_photos_db_connection()
            cursor = conn.cursor()
            for photo_id in photo_ids:
                payload, prep_error = self._prepare_photo_send_payload(photo_id, cursor=cursor)
                if prep_error:
                    formatted = self._format_album_photo_error(photo_id, prep_error)
                    errors.append(formatted)
                    info(f"show_photo album: skipping photo {photo_id} - {formatted}")
                    continue

                if payload.kind == "video":
                    videos.append(payload)
                else:
                    album_photos.append((payload.data, payload.caption))
                info(f"show_photo album: prepared {payload.kind} {photo_id}: {payload.caption}")
        except Exception as e:
            return f"show_photo failed: could not query Photos database: {e}"
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        if not album_photos and not videos:
            return f"show_photo failed: no valid photos found. Errors: {'; '.join(errors)}"

        # Videos go as separate messages after the photo album: send_media_group could carry
        # them too, but a mixed album loses the per-clip caption and streaming preview.
        sent = True
        kwargs = route or {}
        if len(album_photos) == 1:
            sent = notifier().send_photo(album_photos[0][0], caption=album_photos[0][1], **kwargs)
        elif album_photos:
            sent = notifier().send_photo_album(album_photos, **kwargs)
        for video in videos:
            sent = notifier().send_video(video.data, caption=video.caption, thumbnail=video.thumbnail,
                                         **kwargs) and sent
        if not sent:
            return NO_NOTIFICATION_CHANNEL

        total = len(album_photos) + len(videos)
        info(f"Sent {len(album_photos)} photos as album and {len(videos)} video(s) to notifier")
        if errors:
            return f"Sent {total} photos (skipped: {'; '.join(errors)})"
        return ""

    @staticmethod
    def _open_photos_db_connection():
        import sqlite3 as _sqlite3
        from assistant.integration.integrations.photos.PhotoProcessor import PHOTOS_SQLITE_PATH

        return _sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)

    @staticmethod
    def _fetch_photo_asset_row(cursor, asset_guid: str):
        from assistant.integration.integrations.photos.PhotoProcessor import PhotoProcessor
        cursor.execute(PhotoProcessor.ASSET_FILE_SQL, (asset_guid,))
        return cursor.fetchone()

    # Telegram's bot upload limit; the 720p iCloud rendition of a clip under ~2 min fits.
    TELEGRAM_VIDEO_MAX_BYTES = 50 * 1024 * 1024

    def _prepare_photo_send_payload(self, photo_id: int, cursor=None) -> tuple[MediaPayload | None, str | None]:
        from assistant.integration.integrations.photos.PhotoProcessor import PhotoProcessor
        from assistant.integration.integrations.photos.ICloudPhotoDownloader import (
            ICloudPhotoDownloader, NoVideoRenditionError, PhotoNotInCloudError, VideoTooLargeError)

        photo = FloxyDatabase.get_instance().get_photo_by_id(photo_id)
        if not photo:
            return None, "not_found"

        asset_guid = photo.get("asset_guid", "")
        if not asset_guid:
            return None, "missing_asset_guid"

        own_connection = None
        own_cursor = cursor
        try:
            if own_cursor is None:
                own_connection = self._open_photos_db_connection()
                own_cursor = own_connection.cursor()

            row = self._fetch_photo_asset_row(own_cursor, asset_guid)
            if not row:
                return None, f"guid_not_found:{asset_guid}"

            uuid, filename, directory, cloud_master_guid, cloud_asset_guid, library_scope = row
            caption = photo.get("short_description", "")

            if self._is_video(photo):
                record_name = cloud_master_guid or cloud_asset_guid
                try:
                    downloaded = ICloudPhotoDownloader.get_instance().download_video_bytes(
                        record_name, library_scope or 0, max_bytes=self.TELEGRAM_VIDEO_MAX_BYTES,
                        cloud_asset_guid=cloud_asset_guid)
                except VideoTooLargeError as e:
                    info(f"show_photo: video {photo_id} not sent, {e}")
                    return None, "video_too_large"
                except (PhotoNotInCloudError, NoVideoRenditionError) as e:
                    info(f"show_photo: video {photo_id} not sent, {e}")
                    return None, "video_extract_failed"
                if not downloaded:
                    return None, "video_extract_failed"
                return MediaPayload(downloaded[0], caption, uuid, "video",
                                    thumbnail=self._video_poster(record_name, library_scope or 0)), None

            jpeg_bytes = PhotoProcessor._get_image_bytes(
                directory,
                filename,
                cloud_master_guid=cloud_master_guid,
                cloud_asset_guid=cloud_asset_guid,
                library_scope=library_scope or 0,
            )
            if not jpeg_bytes:
                return None, "image_extract_failed"

            return MediaPayload(jpeg_bytes, caption, uuid, "photo"), None
        except Exception as e:
            return None, f"db_error:{e}"
        finally:
            if own_connection is not None:
                own_connection.close()

    @staticmethod
    def _video_poster(record_name: str, library_scope: int) -> Optional[bytes]:
        """The clip's poster frame for the chat thumbnail; a missing one only costs the preview."""
        from assistant.integration.integrations.photos.ICloudPhotoDownloader import ICloudPhotoDownloader
        try:
            poster = ICloudPhotoDownloader.get_instance().download_video_poster(record_name, library_scope)
        except Exception as e:
            trace(f"show_photo: poster frame for {record_name} failed: {e}")
            return None
        if not poster:
            trace(f"show_photo: no poster frame for {record_name}, sending video without thumbnail")
        return poster

    @staticmethod
    def _is_video(photo: dict) -> bool:
        return (photo.get("photo_type") or "").startswith("video")

    @staticmethod
    def _media_tag(photo: dict) -> str:
        """The kind tag shown next to a search hit: screenshots and videos (with their length)."""
        photo_type = photo.get("photo_type") or ""
        if "screenshot" in photo_type:
            return " (screenshot)"
        if photo_type.startswith("video"):
            duration = photo.get("duration_seconds")
            return f" ({photo_type}, {duration:.0f}s)" if duration else f" ({photo_type})"
        return ""

    @staticmethod
    def _format_single_photo_error(photo_id: int, prep_error: str) -> str:
        if prep_error == "not_found":
            return f"no photo found with ID {photo_id}"
        if prep_error == "missing_asset_guid":
            return "photo has no iCloud asset GUID"
        if prep_error.startswith("guid_not_found:"):
            guid = prep_error.split(":", 1)[1]
            return f"asset GUID {guid} not found in Photos database"
        if prep_error == "image_extract_failed":
            return f"could not extract image for photo {photo_id}"
        if prep_error == "video_extract_failed":
            return f"could not download video {photo_id} from iCloud"
        if prep_error == "video_too_large":
            return f"video {photo_id} is too large to send over Telegram (50 MB limit)"
        if prep_error.startswith("db_error:"):
            detail = prep_error.split(":", 1)[1]
            return f"could not query Photos database: {detail}"
        return prep_error

    @staticmethod
    def _format_album_photo_error(photo_id: int, prep_error: str) -> str:
        if prep_error == "not_found":
            return f"photo {photo_id} not found"
        if prep_error == "missing_asset_guid":
            return f"photo {photo_id} has no iCloud asset GUID"
        if prep_error.startswith("guid_not_found:"):
            return f"photo {photo_id} asset GUID not in Photos database"
        if prep_error == "image_extract_failed":
            return f"photo {photo_id} could not extract image"
        if prep_error == "video_extract_failed":
            return f"video {photo_id} could not be downloaded"
        if prep_error == "video_too_large":
            return f"video {photo_id} is too large to send"
        if prep_error.startswith("db_error:"):
            detail = prep_error.split(":", 1)[1]
            return f"photo {photo_id} database error: {detail}"
        return f"photo {photo_id} {prep_error}"

    @staticmethod
    def _show_photo_on_kitchen_tv(uuid: str, caption: str) -> str:
        import subprocess

        trace(f"Switching Kitchen TV to PC mode")
        try:
            result = subprocess.run(
                ["./main.py", "setpcmode"],
                cwd=os.path.expanduser("~/Documents/Floxy/TheFrameAPI"),
                timeout=10,
                capture_output=True, text=True,
            )
            fine(f"setpcmode stdout: {result.stdout.strip()}")
            fine(f"setpcmode stderr: {result.stderr.strip()}")
            if result.returncode != 0:
                error(f"setpcmode exited with code {result.returncode}")
                return f"show_photo failed: setpcmode exited with code {result.returncode}: {result.stderr.strip()}"
        except Exception as e:
            error(f"Failed to set Kitchen TV to PC mode: {e}")
            return f"show_photo failed: could not switch Kitchen TV to PC mode: {e}"

        trace(f"Opening photo {uuid} in Photos app via AppleScript")
        applescript = f'''
            tell application "Photos"
                activate
                spotlight media item id "{uuid}"
            end tell
            tell application "System Events"
                tell process "Photos"
                    set frontmost to true
                    tell window 1
                        set value of attribute "AXFullScreen" to true
                    end tell
                end tell
            end tell
        '''
        try:
            result = subprocess.run(["osascript", "-e", applescript], timeout=10,
                                    capture_output=True, text=True)
            fine(f"osascript stdout: {result.stdout.strip()}")
            fine(f"osascript stderr: {result.stderr.strip()}")
            if result.returncode != 0:
                error(f"osascript exited with code {result.returncode}")
                return f"show_photo failed: AppleScript exited with code {result.returncode}: {result.stderr.strip()}"
        except Exception as e:
            error(f"Failed to open photo in Photos app: {e}")
            return f"show_photo failed: could not open photo in Photos app: {e}"

        info(f"Showing photo {uuid} on Kitchen TV: {caption}")
        return ""

    def cmd_search_photos(self, *args: str):
        self.background_command(self._run_search_photos, list(args))

    @staticmethod
    def _run_search_photos(*args: str) -> str:
        from datetime import datetime
        keywords = args[0].strip() if args else ""
        sort_by = args[1].strip() if len(args) > 1 else "relevance"
        keywords = keywords.replace("str = ", "").replace("str=", "").strip()
        sort_by = sort_by.replace("str = ", "").replace("str=", "").strip()
        if sort_by not in FloxyDatabase.SEARCH_SORT_OPTIONS:
            info(f"Invalid sort_by '{sort_by}', falling back to relevance")
            sort_by = "relevance"
        results = FloxyDatabase.get_instance().search_photos(keywords, limit=25, sort_by=sort_by)
        if not results:
            if keywords and keywords.strip() != "*":
                return f'No photos found matching "{keywords}".'
            return "No photos found."
        lines = []
        for r in results:
            date_str = ""
            if r.get("creation_date"):
                date_str = datetime.fromtimestamp(r["creation_date"]).strftime("%Y-%m-%d")
            location = r.get("location_text") or "Unknown location"
            pretty = r.get("pretty_score", 0)
            family = r.get("family_importance_score", 0)
            persons_tag = f' [{r["persons"]}]' if r.get("persons") else ""
            lines.append(f'{r["id"]}: {r["short_description"]}{Commands._media_tag(r)} ({location}, {date_str}) [pretty: {pretty}, family: {family}]{persons_tag}')
        info(f"Photo search for '{keywords}' returned {len(results)} results")
        return "\n".join(lines)

    def cmd_search_history(self, *args: str):
        self.background_command(self._run_search_history, list(args))

    @staticmethod
    def _run_search_history(*args: str) -> str:
        from datetime import datetime
        query = args[0].replace("str = ", "").replace("str=", "").strip() if args else ""
        sort = "relevance"
        if len(args) > 1:
            sort = args[1].replace("str = ", "").replace("str=", "").strip().lower()
            if sort not in FloxyDatabase.HISTORY_SORT_OPTIONS:
                info(f"Invalid sort '{sort}', falling back to relevance")
                sort = "relevance"
        if not query:
            return "No query provided for history search."

        db = FloxyDatabase.get_instance()

        entries = []
        seen_content = set()

        for r in db.search_history(query, sort=sort, limit=30):
            content = r["content"]
            if content in seen_content:
                continue
            seen_content.add(content)
            if len(content) > 600:
                content = content[:600] + "..."
            source_tag = f" ({r['source']})" if r.get("source") else ""
            location_tag = f" [{r['location']}]" if r.get("location") else ""
            date_str = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
            entries.append((
                r["timestamp"],
                r.get("fts_relevance", 0),
                f"[{date_str}]{location_tag} {r['role']}{source_tag}: {content}"
            ))

        photo_count = 0
        for r in db.search_photos(query, limit=5, sort_by="relevance"):
            photo_count += 1
            ts = r.get("creation_date", 0)
            date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
            location = r.get("location_text") or "Unknown location"
            pretty = r.get("pretty_score", 0)
            family = r.get("family_importance_score", 0)
            persons_tag = f' [{r["persons"]}]' if r.get("persons") else ""
            entries.append((
                ts,
                r.get("fts_relevance", 0),
                f"[{date_str}] photo#{r['id']}: {r['short_description']}{Commands._media_tag(r)} ({location}) [pretty: {pretty}, family: {family}]{persons_tag}"
            ))

        if not entries:
            return f'No history entries or photos found matching "{query}".'

        if sort == "oldest":
            entries.sort(key=lambda e: e[0])
        elif sort == "newest":
            entries.sort(key=lambda e: e[0], reverse=True)
        else:
            entries.sort(key=lambda e: e[1], reverse=True)

        info(f"History search for '{query}' returned {len(entries) - photo_count} history + {photo_count} photo results")
        return "\n".join(e[2] for e in entries)

    def cmd_show_history(self, *args: str):
        self.background_command(self._run_show_history, list(args))

    @staticmethod
    def _run_show_history(*args: str) -> str:
        from datetime import datetime
        if len(args) < 2:
            return "show_history failed: expected two parameters (from, to) in YYYY-MM-DD HH:MM format."
        from_str = args[0].strip()
        to_str = args[1].strip()

        try:
            from_dt = datetime.strptime(from_str, "%Y-%m-%d %H:%M")
            to_dt = datetime.strptime(to_str, "%Y-%m-%d %H:%M")
        except ValueError:
            return "show_history failed: timestamps must be in YYYY-MM-DD HH:MM format."

        from_ts = from_dt.timestamp()
        to_ts = to_dt.timestamp()

        db = FloxyDatabase.get_instance()
        results = db.get_history_range(from_ts, to_ts)

        if not results:
            return f'No history entries found between {from_str} and {to_str}.'

        lines = []
        seen_content = set()
        for r in results:
            content = r["content"]
            if content in seen_content:
                continue
            seen_content.add(content)
            if len(content) > 600:
                content = content[:600] + "..."
            source_tag = f" ({r['source']})" if r.get("source") else ""
            location_tag = f" [{r['location']}]" if r.get("location") else ""
            date_str = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{date_str}]{location_tag} {r['role']}{source_tag}: {content}")

        info(f"show_history({from_str} -> {to_str}) returned {len(lines)} entries")
        return "\n".join(lines)

    def _run_create_event(self, *args: str) -> str:
        try:
            if len(args) < 4:
                return "create_event failed: expected 4 parameters (title, date, start_time, end_time)"
            # If the LLM under-quoted and produced extra tokens, the title may have
            # been split at internal commas — re-join leading tokens into the title
            # and pull the last 3 tokens as date/start/end.
            title = ", ".join(a.strip() for a in args[:-3])
            date_str = args[-3].strip()
            start_time = args[-2].strip()
            end_time = args[-1].strip()
            CalDAVIntegration.create_calendar_event(title, date_str, start_time, end_time)
            return ""
        except Exception as e:
            return f"create_event failed: {e}"

    def cmd_create_event(self, *args: str):
        self.background_command(self._run_create_event, list(args))

    def cmd_load_calendar(self, *names: str):
        self.background_command(self._run_load_calendar, list(names))

    @staticmethod
    def _run_load_calendar(*names: str) -> str:
        """The named calendars over the window the prompt's calendar section covers."""
        names = [n for n in names if n and n.strip()]
        if not names:
            return "load_calendar failed: expected a calendar's name"
        try:
            return CalDAVIntegration.load_calendars(names)
        except Exception as e:
            return f"load_calendar failed: {e}"

    def cmd_add_shopping_list_item(self, parameter_string: str) -> str:
        return add_reminder_item(parameter_string)

    @staticmethod
    def _get_icloud_api() -> iCloudAPI:
        return iCloudAPI.get_instance()

    def cmd_find_phone(self, parameter_string: str):
        self.fire_and_forget(self._get_icloud_api().play_sound, parameter_string)

    def cmd_locate_user(self, parameter_string: str):
        self.background_command(self._get_icloud_api().get_location, parameter_string)

    def cmd_icloud_devices(self, parameter_string: str = ""):
        self.background_command(self._run_icloud_devices, "")

    def _run_icloud_devices(self, _="") -> str:
        """Every device the iCloud account sees, with its id and the icloud.devices entry it
        is configured in — the material for finishing the setup with config_set."""
        api = self._get_icloud_api()
        devices = api.get_icloud_devices()
        if not devices:
            info("icloud_devices: the account listed no devices")
            return ("iCloud listed no devices — the login may be incomplete (list_integrations shows the "
                    "iCloud state; a 2FA code is entered in the console's iCloud tab).")
        lines = []
        for d in devices:
            model = f" ({d['model']})" if d.get("model") else ""
            if d.get("configured_as"):
                status = f"configured as '{d['configured_as']}' (icloud.devices[{d['slot']}])"
            else:
                status = "not configured"
            lines.append(f"{d['name']}{model} — id {d['id']} — {status}")
        n = api.configured_device_count()
        info(f"icloud_devices: {len(lines)} device(s), {n} configured")
        return "\n".join(lines) + (f"\n\nicloud.devices has {n} entr{'y' if n == 1 else 'ies'}; the next one is "
                                    f"icloud.devices[{n}] (keys id and display_name).")

    def cmd_send_imessage(self, *args: str) -> str:
        if len(args) < 2:
            return "send_imessage failed: expected 'recipient, message' parameters"
        recipient = args[0].strip()
        # If the LLM didn't quote the body, trailing tokens are body fragments;
        # re-join them so commas in the message survive.
        body = ", ".join(a for a in args[1:]).strip()
        if not recipient or not body:
            return "send_imessage failed: expected 'recipient, message' parameters"

        # LLM emits literal "\n" escape sequences — turn them into real newlines.
        body = normalize_llm_escapes(body)
        body = f"{body}\n\n{self.AI_MESSAGE_SUFFIX}"
        from assistant.integration.integrations.IMessageIntegration import IMessageIntegration
        return IMessageIntegration.get_instance().send(recipient, body)

    def cmd_send_email(self, *args: str) -> str:
        if len(args) < 3:
            return "send_email failed: expected 'to, subject, body' parameters"
        to = args[0].strip()
        subject = args[1].strip()
        # Re-join trailing tokens so an under-quoted body with commas survives.
        body = ", ".join(a for a in args[2:]).strip()
        if not to or not subject or not body:
            return "send_email failed: expected 'to, subject, body' parameters"

        # LLM emits literal "\n" escape sequences — turn them into real newlines.
        body = normalize_llm_escapes(body)
        return ICloudEmailAccess.get_instance().send(to, subject, body)

    def cmd_fetch_camera_images(self, parameter_string: str):
        self.background_command(self._fetch_all_camera_images, parameter_string)

    @staticmethod
    def _fetch_arlo_snapshots() -> list:
        try:
            arlo = ArloCamera.get_instance()
            result = arlo.fetch_all_snapshots()
            if isinstance(result, list):
                return [b for b in result if not (b.get("type") == "text" and b.get("text", "").startswith("Here are"))]
            return []
        except Exception as e:
            error(f"Arlo snapshot fetch failed: {e}")
            return []

    @staticmethod
    def _fetch_ring_snapshot() -> list:
        try:
            ring = RingIntegration.get_instance()
            return ring.fetch_snapshot()
        except Exception as e:
            error(f"Ring snapshot fetch failed: {e}")
            return []

    @staticmethod
    def _fetch_all_camera_images(_params) -> object:
        from concurrent.futures import ThreadPoolExecutor

        results = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(Commands._fetch_arlo_snapshots),
                pool.submit(Commands._fetch_ring_snapshot),
            ]
            for f in futures:
                result = f.result()
                if isinstance(result, list):
                    results.extend(result)

        if not results:
            return "No camera images available."

        results.insert(0, {"type": "text", "text": "Here are the current camera images from around the house:"})
        return results

    def cmd_self_analysis(self, parameter_string: str):
        reason = CodeQualityMaintainer.unavailable_reason()
        if reason:
            info(f"self_analysis not started: {reason}")
            return reason

        def _run_and_deliver(params):
            try:
                report = CodeQualityMaintainer.analyze_code(params)
                trace("Received self-analysis report from Code Quality Maintainer")
                fine(f"Report: {report}")
                # The report is markdown; the history row holds it in the chat dialect like any
                # reply, so the Home tab draws it formatted. Since the console draws the history
                # row, the push is marked as one and reaches only the chats without a history view.
                html_report = notifier().markdown_to_html(report)
                aam = AutonomousActionManager.get_instance()
                prompt_manager = aam.prompt_manager if aam else None
                sent = notifier().send_message(f"Self-analysis report:\n\n{html_report}",
                                               in_history=prompt_manager is not None)
                # "[Self-analysis complete]" is the tag the LLM keys off; keep it first.
                delivery = "" if sent else f" ({NO_NOTIFICATION_CHANNEL})"
                if prompt_manager is not None:
                    prompt_manager.update_history(None, None, f"[Self-analysis complete]{delivery} {html_report}")
                else:
                    info("Self-analysis: no AutonomousActionManager yet, the report is not in the history")
            except Exception as e:
                error(f"Self-analysis failed: {e}")

        self.fire_and_forget(_run_and_deliver, parameter_string)

    def cmd_self_improvement(self, parameter_string: str):
        reason = CodeQualityMaintainer.unavailable_reason()
        if reason:
            info(f"self_improvement not started: {reason}")
            return reason

        def _run_and_deliver(params):
            try:
                report = CodeQualityMaintainer.improve_code(params)
                trace("Received self-improvement report from Code Quality Maintainer")
                fine(f"Report: {report}")
                html_report = notifier().markdown_to_html(report)
                sent = notifier().send_message(f"Self-improvement report:\n\n{html_report}")
                outcome = "report sent to user" if sent else f"{NO_NOTIFICATION_CHANNEL}; report: {report}"

                aam = AutonomousActionManager.get_instance()
                if aam and aam.prompt_manager:
                    aam.prompt_manager.update_history(None, None, f"[Self-improvement complete — {outcome}]")
            except Exception as e:
                error(f"Self-improvement failed: {e}")

        self.fire_and_forget(_run_and_deliver, parameter_string)

    def cmd_restart_server(self, parameter_string: str):
        def _run(_params):
            try:
                result = CodeQualityMaintainer.pull_and_restart()
                info(f"Restart server result: {result}")
                system_note("server", f"Server restart: {result}")
            except Exception as e:
                error(f"Restart server failed: {e}")

        self.fire_and_forget(_run, parameter_string)

    def cmd_execute(self, parameter_string: str) -> str:
        return home.active().execute(self._home_name(parameter_string))

    def cmd_search_music(self, parameter_string: str):
        self.background_command(self.music_manager.search_spotify, parameter_string)

    def cmd_play_music(self, *args: str):
        if len(args) < 2:
            return "play_music failed: expected 'zone, id' parameters"
        room = args[0].strip()
        id = args[1].strip()
        self.music_manager.play_in_zone(id, room)

    def cmd_play(self, parameter_string: str):
        self.music_manager.play(parameter_string)

    def cmd_pause(self, parameter_string: str):
        self.music_manager.pause(parameter_string)

    def cmd_next(self, parameter_string: str):
        self.music_manager.next_track(parameter_string)

    def cmd_volume_up(self, parameter_string: str):
        self.music_manager.adjust_volume(parameter_string, 4)

    def cmd_volume_down(self, parameter_string: str):
        self.music_manager.adjust_volume(parameter_string, -4)

    #  todo: this is not in the prompt, do we still want it?
    def cmd_sonos_enqueue_and_play(self, *args: str):
        if len(args) < 2:
            return "sonos_enqueue_and_play failed: expected 'zone, playlist' parameters"
        room = args[0].strip()
        playlist = args[1].strip()

        favorites = self.music_manager.get_favorites(room)
        if playlist in favorites:
            trace("Sonos favorite, playing.")
            self.music_manager.play_favorite(room, playlist)
        else:
            trace("Sonos playlist, playing.")
            self.music_manager.play_playlist(room, playlist, shuffle=True)

    def cmd_vault_add(self, *args: str) -> str:
        """Files a value the user has already said out loud into the vault, so every later
        request (and the history that already holds it) is redacted from here on."""
        if len(args) < 3:
            return "vault_add failed: expected 'kind, description, value' parameters"
        kind, description = args[0].strip().lower(), args[1].strip()
        # An unquoted value with commas in it arrives split — the rest of the args are the value.
        value = ", ".join(a.strip() for a in args[2:])
        if kind not in KINDS:
            return f"vault_add failed: unknown kind '{kind}', expected one of {', '.join(KINDS)}"
        if not description or not value:
            return "vault_add failed: description and value must not be empty"
        vault = Vault.get_instance()
        if not vault.enabled:
            error(f"vault_add: vault disabled — {vault.error}")
            return f"vault_add failed: the vault is disabled ({vault.error})"
        try:
            secret = vault.add(kind, description, value)
        except DuplicateSecret as e:
            info(f"vault_add: value already vaulted as '{e.existing_id}'")
            return f"vault_add: that value is already in the vault as {{{{secret:{e.existing_id}}}}}"
        except VaultError as e:
            return f"vault_add failed: {e}"
        info(f"vault_add: stored '{secret.id}' ({kind}) — redacted from the next request on")
        return ""

    def cmd_vault_request(self, *args: str) -> str:
        """Asks the user for a value through a masked prompt in the chat UIs and files it in
        the vault; the LLM only ever gets the placeholder back. The wait runs as a background
        command so its outcome reaches the LLM as a plain reply, not an error report."""
        if len(args) < 2:
            return "vault_request failed: expected 'kind, description' parameters"
        kind = args[0].strip().lower()
        # An unquoted description with commas in it arrives split — the rest of the args are the description.
        description = ", ".join(a.strip() for a in args[1:])
        if kind not in KINDS:
            return f"vault_request failed: unknown kind '{kind}', expected one of {', '.join(KINDS)}"
        if not description:
            return "vault_request failed: description must not be empty"
        vault = Vault.get_instance()
        if not vault.enabled:
            error(f"vault_request: vault disabled — {vault.error}")
            return f"vault_request failed: the vault is disabled ({vault.error})"
        # Captured now, on the command worker: by the time the bg worker runs, another
        # request may have replaced _current_context. The prompt opens only in this chat.
        ctx = self._current_context
        source = getattr(ctx, "source", None)
        conversation = getattr(ctx, "conversation", None)
        self.background_command(self._run_vault_request, (kind, description, source, conversation))

    def _run_vault_request(self, kind: str, description: str, source=None, conversation=None) -> str:
        """Runs on the bg command worker: blocks in the notifier until the user answered the
        prompt (or cancelled / timed out). The value only ever exists inside `accept`.
        *source*/*conversation* are the asking request's, so the prompt opens in that chat."""
        vault = Vault.get_instance()
        outcome: dict = {}

        def accept(value: str) -> str:
            try:
                secret = vault.add(kind, description, value)
            except DuplicateSecret as e:
                outcome["duplicate"] = e.existing_id
                info(f"vault_request: value already vaulted as '{e.existing_id}'")
                return f"Already in the vault as {e.existing_id}"
            except VaultError as e:
                outcome["reason"] = str(e)
                raise SecretRejected(str(e))
            outcome["secret"] = secret
            info(f"vault_request: stored '{secret.id}' ({kind}) — redacted from the next request on")
            return f"Stored as {secret.id} ({secret.masked})"

        decision = notifier().request_secret(title=f"{description} ({kind})", details="", accept=accept,
                                             source=source, conversation=conversation)
        if decision == "accepted":
            if "duplicate" in outcome:
                return (f"That value is already in the vault as {{{{secret:{outcome['duplicate']}}}}} — "
                        f"nothing new was stored.")
            secret = outcome["secret"]
            return (f"Stored as {{{{secret:{secret.id}}}}} ({kind}: {description}). Use that placeholder from "
                    f"now on; the value itself was never shown to you.")
        info(f"vault_request: {decision} for '{description}' ({kind}) — nothing stored")
        messages = {
            "cancel": "The user cancelled the prompt — nothing was stored.",
            "timeout": "The user did not answer the prompt in time — nothing was stored.",
            "rejected": f"The value was refused ({outcome.get('reason', 'it could not be stored')}) — "
                        "nothing was stored.",
            "unavailable": "No chat interface can show the prompt right now — ask the user to add the value "
                           "in the Vault tab of the console instead.",
        }
        return messages.get(decision, f"vault_request ended with '{decision}' — nothing was stored.")

    # --- self-setup: settings and integrations (plans/SELF_SETUP_PLAN.md) --------------------
    # The LLM never sees config.yaml: it gets key names, set / not set, and the example's text.
    # All four run in the background so their text reaches the LLM as a plain reply.

    _STATE_LABELS = {"ok": "ok", "warning": "warning", "error": "error", "not_configured": "not configured",
                     "disabled": "disabled", "needs_2fa": "needs 2FA", "rate_limited": "rate limited",
                     "unknown": "not checked yet"}
    # Integrations that are not Integration classes but follow the same switch, with their keys.
    @staticmethod
    def _integration_monitor():
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        return IntegrationMonitor.get_instance()

    @staticmethod
    def _keys_summary(keys: list[dict]) -> str:
        return ", ".join(f"{k['key']}: {'set' if k['set'] else 'not set'}" for k in keys)

    def cmd_list_integrations(self, parameter_string: str = ""):
        self.background_command(self._run_list_integrations, "")

    def _run_list_integrations(self, _="") -> str:
        statuses = self._integration_monitor().describe_all()
        if not statuses:
            info("list_integrations: nothing registered")
            return "No integrations are registered."
        lines = []
        for status in sorted(statuses, key=lambda s: s["name"].lower()):
            line = f"{status['name']} — {self._STATE_LABELS.get(status['state'], status['state'])}"
            if status["state"] == "disabled":
                line += f'; enable: enable_integration("{status["name"]}")'
            if status["last_error"] and status["state"] in ("warning", "error", "needs_2fa", "rate_limited"):
                line += f": {status['last_error'][:160]}"
            if status["last_check_time"]:
                line += f" (checked {self._age(status['last_check_time'])})"
            if status["keys"]:
                line += f" [{self._keys_summary(status['keys'])}]"
            if "feed" in status:
                from assistant.integration.FeedRunner import describe_feed_status
                # A disabled feed has no runner to describe, but it is still a feed.
                line += "; feed" if status["state"] == "disabled" else f"; feed: {describe_feed_status(status['feed'])}"
            line += "; restartable" if status["restartable"] else "; needs a server restart to change"
            if status.get("sensitive"):
                line += "; private data: enabling it ends the beta's transcript telemetry"
            lines.append(line)
        info(f"list_integrations: {len(lines)} integrations")
        return "\n".join(lines)

    @staticmethod
    def _age(epoch: float) -> str:
        seconds = max(0, int(time.time() - epoch))
        if seconds < 60:
            return f"{seconds}s ago"
        if seconds < 3600:
            return f"{seconds // 60} min ago"
        return f"{seconds // 3600} h ago"

    def cmd_config_help(self, parameter_string: str = ""):
        self.background_command(self._run_config_help, parameter_string.strip().strip('"\''))

    def _run_config_help(self, section: str = "") -> str:
        if not section:
            lines = []
            for name in ConfigSchema.sections():
                counts = ConfigStore.section_summary(name)
                lines.append(f"{name} ({counts})" if counts else name)
            info(f"config_help: listed {len(lines)} sections")
            return ("Sections of config.example.yaml (set = config.yaml holds a real value):\n" + "\n".join(lines)
                    + "\n\nconfig_help(section) shows one section with its comments.")
        text = ConfigSchema.section_text(section)
        if text is None:
            info(f"config_help: no section '{section}'")
            return f"No section '{section}' in config.example.yaml. Sections: {', '.join(ConfigSchema.sections())}"
        name = section.strip().lower()
        if not ConfigSchema.leaves(name):
            info(f"config_help: section '{section}' (documentation only, no keys)")
            return text
        info(f"config_help: section '{section}'")
        return f"{text}\n\nCurrently: {ConfigStore.summary([name])}"

    def cmd_config_request(self, parameter_string: str) -> str:
        """Asks the user for one setting through the masked prompt (the vault_request
        machinery with a config-writing accept) — the LLM gets 'saved', never the value."""
        key = ConfigSchema.normalize_key(parameter_string.strip().strip('"\''))
        if not key:
            return "config_request failed: expected a setting key such as email.password"
        try:
            ConfigStore.check_key(key)
        except ConfigStoreError as e:
            info(f"config_request: refused — {e}")
            return f"config_request failed: {e}"
        ctx = self._current_context
        source = getattr(ctx, "source", None)
        conversation = getattr(ctx, "conversation", None)
        self.background_command(self._run_config_request, (key, source, conversation), lane=self.CONFIG_LANE)

    def _run_config_request(self, key: str, source=None, conversation=None) -> str:
        outcome: dict = {}
        replaces = ConfigStore.is_set(key)

        def accept(value: str) -> str:
            try:
                ConfigStore.set_value(key, value)
            except ConfigStoreError as e:
                outcome["reason"] = str(e)
                raise SecretRejected(str(e))
            outcome["saved"] = True
            return f"Saved {key}"

        details = "This replaces the current value." if replaces else ""
        decision = notifier().request_secret(title=f"Setting {key}", details=details, accept=accept,
                                             source=source, conversation=conversation)
        if decision == "accepted":
            return f"{key} saved. {self._restart_hint(key)}"
        info(f"config_request: {decision} for '{key}' — nothing stored")
        messages = {
            "cancel": "The user cancelled the prompt — nothing was saved.",
            "timeout": "The user did not answer the prompt in time — nothing was saved.",
            "rejected": f"The value was refused ({outcome.get('reason', 'it could not be stored')}) — "
                        "nothing was saved.",
            "unavailable": "No chat interface can show the prompt right now — ask the user to open the console "
                           "or Telegram and try again.",
        }
        return messages.get(decision, f"config_request ended with '{decision}' — nothing was saved.")

    RESTART_HINT = "Restart the integration that uses it (restart_integration) for the change to take effect."
    TELEGRAM_RESTART_HINT = 'Run restart_integration("Telegram") for the bot to pick it up — no server restart.'
    # The chat-id lists are read live by the bot (TelegramTransport._live_chat_ids). The FTUE
    # assistant once restarted the bot from the Telegram chat that had just sent its chat id,
    # which killed that very conversation and left its spinner stuck.
    TELEGRAM_LIVE_HINT = "In force from the next message on — the bot reads it live, no restart needed."
    TELEGRAM_RESTART_KEYS = frozenset({"telegram_bot.bot_token", "telegram_bot.link_previews"})
    SERVER_RESTART_HINT = ("The server reads interfaces.* only at startup — offer restart_server "
                           "(developer toolkit) for the change to take effect.")
    # The FTUE assistant once saved api_keys.gemini, found no integration owning it in the
    # listing and concluded the key is read at startup — then offered a server restart mid-setup.
    # The consumers that matter for setup build their LLM on first use, so say so.
    MODEL_KEY_HINT = ("Model-provider key. Photo indexing, video descriptions and any integration enabled "
                      "from now on use it straight away — setup needs no restart.")
    MODEL_PROVIDER_KEYS = frozenset({"openai", "anthropic", "gemini", "grok", "openrouter"})
    DEVELOPER_RELEASE_HINT = ("Takes effect at the next restart_server or nightly update, which checks out master "
                              "(true) or release (false) before pulling — offer the restart if the user wants to "
                              "switch now.")
    TELEMETRY_HINT = ("Telemetry reads it for every event: false is in force from the next event on, "
                      "no restart needed. Switching telemetry.enabled back to true takes a restart_server.")

    @classmethod
    def _restart_hint(cls, key: str) -> str:
        if key == "developer_release":
            return cls.DEVELOPER_RELEASE_HINT
        if key.startswith("telemetry."):
            return cls.TELEMETRY_HINT
        if key.startswith("interfaces."):
            return cls.SERVER_RESTART_HINT
        if key.startswith("telegram_bot."):
            return cls.TELEGRAM_RESTART_HINT if key.split("[", 1)[0] in cls.TELEGRAM_RESTART_KEYS else cls.TELEGRAM_LIVE_HINT
        if key.startswith("api_keys.") and key.split(".", 1)[1] in cls.MODEL_PROVIDER_KEYS:
            return cls.MODEL_KEY_HINT
        return cls.RESTART_HINT

    def cmd_config_set(self, *args: str) -> str:
        """Writes a value the assistant states itself — a plain setting the user said in the chat,
        or a credential it already holds (an API key it just created in the browser). A credential
        the user has to supply preferably goes through config_request's masked prompt."""
        if len(args) < 2:
            return "config_set failed: expected (key, value), e.g. config_set(\"owner_name\", \"Anna\")"
        key = ConfigSchema.normalize_key(args[0].strip().strip('"\''))
        # An unquoted value with commas arrives split; re-join it like send_imessage does.
        value = ", ".join(args[1:]).strip().strip('"\'')
        try:
            ConfigStore.check_key(key)
        except ConfigStoreError as e:
            info(f"config_set: refused — {e}")
            return f"config_set failed: {e}"
        self.background_command(self._run_config_set, [key, value], lane=self.CONFIG_LANE)

    def _run_config_set(self, key: str, value: str) -> str:
        interface = ConfigStore.interface_flag(key)
        before = ConfigStore.interface_enabled(interface) if interface else None
        try:
            ConfigStore.set_value(key, value)
        except ConfigStoreError as e:
            info(f"config_set: refused — {e}")
            return f"config_set failed: {e}"
        if interface and ConfigStore.interface_enabled(interface) == before:
            # Telegram and console are on by default: writing "true" for them changes nothing,
            # and the restart hint would send the LLM off to restart_server for no reason.
            info(f"config_set: {key} was already {'on' if before else 'off'} in effect — no restart needed")
            return (f"{key} saved, but the {interface} interface was already {'on' if before else 'off'} "
                    f"(its default) — nothing changes and no restart is needed.")
        return f"{key} saved. {self._restart_hint(key)}"

    TELEGRAM_FEED = "Telegram Feed"

    def cmd_telegram_login(self, parameter_string: str):
        """Signs the Telegram Feed's account in from the chat: the code Telegram sends and
        the account's 2FA password come in through the masked prompt, so the LLM sees neither.
        Console only — Telegram invalidates a login code that is sent as a Telegram message,
        so the bot chat cannot carry it."""
        phone = parameter_string.strip().strip('"\'').replace(" ", "")
        if not phone.startswith("+") or not phone[1:].isdigit() or len(phone) < 8:
            return ('telegram_login failed: expected the phone number with its country code, '
                    'e.g. telegram_login("+46701234567")')
        ctx = self._current_context
        source = getattr(ctx, "source", None)
        if source != SOURCE_CONSOLE:
            info(f"telegram_login: refused from {source or 'no chat'} — console only")
            return ("telegram_login failed: it only works from the console chat — Telegram invalidates a login "
                    "code that is sent as a Telegram message, so the code cannot be typed into this chat. "
                    "Ask the user to open the console and say it there.")
        conversation = getattr(ctx, "conversation", None)
        self.background_command(self._run_telegram_login, (phone, source, conversation), lane=self.CONFIG_LANE)

    def _run_telegram_login(self, phone: str, source=None, conversation=None) -> str:
        monitor = self._integration_monitor()
        integration = monitor.find(self.TELEGRAM_FEED)
        if integration is None or not hasattr(integration, "login"):
            state = getattr(integration, "state", None)
            label = self._STATE_LABELS.get(getattr(state, "value", state), "not available")
            info(f"telegram_login: Telegram Feed is {label}, no login possible")
            if label == "disabled":
                return f'Telegram Feed is disabled — enable_integration("{self.TELEGRAM_FEED}") first.'
            return f"Telegram Feed is {label} — set telegram_feed.api_id and api_hash and enable it first."

        def ask(title: str, details: str, accept) -> str:
            return notifier().request_secret(title=title, details=details, accept=accept,
                                             source=source, conversation=conversation)

        failure = integration.login(phone, ask)
        if failure:
            info(f"telegram_login: {failure}")
            return failure
        result = monitor.restart(integration.name)
        return f"Telegram Feed signed in. {self._describe_restart(result['status'], result['restartable'])}"

    def cmd_restart_integration(self, parameter_string: str):
        name = parameter_string.strip().strip('"\'')
        if not name:
            return "restart_integration failed: expected an integration name from list_integrations"
        self.background_command(self._run_restart_integration, name, lane=self.CONFIG_LANE)

    def _run_restart_integration(self, name: str) -> str:
        monitor = self._integration_monitor()
        result = monitor.restart(name)
        if result is None:
            known = ", ".join(sorted(s["name"] for s in monitor.get_all_statuses()))
            return f"No integration named '{name}'. Known: {known}"
        return self._describe_restart(result["status"], result["restartable"])

    def cmd_enable_integration(self, parameter_string: str):
        """Adds the integration to enabled_integrations by name and rebuilds it — one call,
        so the LLM never juggles list indices (a second index write used to push the first
        name out)."""
        name = parameter_string.strip().strip('"\'')
        if not name:
            return "enable_integration failed: expected an integration name from list_integrations"
        self.background_command(self._run_enable_integration, name, lane=self.CONFIG_LANE)

    def cmd_disable_integration(self, parameter_string: str):
        name = parameter_string.strip().strip('"\'')
        if not name:
            return "disable_integration failed: expected an integration name from list_integrations"
        self.background_command(self._run_disable_integration, name, lane=self.CONFIG_LANE)

    def _run_enable_integration(self, name: str) -> str:
        return self._switch_integration(name, enable=True)

    def _run_disable_integration(self, name: str) -> str:
        return self._switch_integration(name, enable=False)

    def _switch_integration(self, name: str, enable: bool) -> str:
        """Edit enabled_integrations, then restart the integration so the list takes effect now."""
        verb = "enable" if enable else "disable"
        canonical = self._canonical_integration_name(name)
        if canonical is None:
            known = ", ".join(sorted(self._known_integration_names()))
            info(f"{verb}_integration: no integration named '{name}'")
            return f"No integration named '{name}'. Known: {known}"
        was_listed = ConfigManager.get_instance().integration_enabled(canonical)
        try:
            changed = (ConfigStore.enable_integration if enable else ConfigStore.disable_integration)(canonical)
        except ConfigStoreError as e:
            info(f"{verb}_integration: refused — {e}")
            return f"{verb}_integration failed: {e}"
        result = self._integration_monitor().restart(canonical)
        if was_listed != enable:
            Telemetry.get_instance().integration_switch(canonical, enable, result["status"]["state"] if result else None)
        else:
            trace(f"{verb}_integration: {canonical} was already {verb}d, no telemetry event")
        if result is None:
            return changed
        return f"{changed}; {self._describe_restart(result['status'], result['restartable'])}"

    def _known_integration_names(self) -> list[str]:
        return [s["name"] for s in self._integration_monitor().get_all_statuses()]

    def _canonical_integration_name(self, name: str) -> Optional[str]:
        """The name as list_integrations shows it for any spelling the config accepts."""
        from common.ConfigManager import integration_key
        wanted = integration_key(name)
        for known in self._known_integration_names():
            if integration_key(known) == wanted:
                return known
        return None

    def _describe_restart(self, status: dict, restartable: bool) -> str:
        if not restartable:
            return f"{status['name']} cannot be restarted from here — restart the server for a config change to apply."
        state = status["state"]
        line = f"{status['name']} restarted: {self._STATE_LABELS.get(state, state)}"
        if state == "disabled":
            line += f' — not in enabled_integrations; enable_integration("{status["name"]}") adds it'
        elif state == "not_configured":
            missing = [k["key"] for k in ConfigStore.describe_keys(status["config_keys"]) if not k["set"]]
            if missing:
                line += f" — not set: {', '.join(missing)}"
            elif status["last_error"]:
                line += f" — {status['last_error']}"
        elif state == "needs_2fa":
            if status["name"] == self.TELEGRAM_FEED:
                line += " — sign the account in with telegram_login(phone) from the console chat"
            else:
                line += " — the login must be completed in the console's tab for it (the code arrives by SMS or email)"
        elif status["last_error"]:
            line += f" — {status['last_error'][:300]}"
        return line

    def cmd_remember(self, parameter_string: str):
        MemoryFile.append(parameter_string)

    # The watch list and the procedures: the argument handling lives with the stores
    # (the memory curator's command block runs the same commands through them).
    def cmd_load_watch(self, *names: str):
        self.background_command(self._run_load_watch, list(names))

    @staticmethod
    def _run_load_watch(*names: str) -> str:
        """The named watch items in full, splatted like `_run_load_procedure`."""
        names = [n for n in names if n and n.strip()]
        if not names:
            return "load_watch failed: expected an item's name. Open: " + (", ".join(i.name for i in WatchList.items()) or "none")
        return "\n".join(WatchList.load(name) for name in names)

    def cmd_save_watch(self, *args: str) -> str:
        return WatchList.run("save_watch", list(args))

    def cmd_checked_watch(self, *args: str) -> str:
        return WatchList.run("checked_watch", list(args))

    def cmd_delete_watch(self, *args: str) -> str:
        return WatchList.run("delete_watch", list(args))

    def cmd_load_procedure(self, *names: str):
        self.background_command(self._run_load_procedure, list(names))

    @staticmethod
    def _run_load_procedure(*names: str) -> str:
        """The named procedures, each under its header. Splatted like every list-parameter
        background command (`function(*params)`)."""
        names = [n for n in names if n and n.strip()]
        if not names:
            return "load_procedure failed: expected a procedure name. Available: " + (", ".join(Procedures.names()) or "none")
        return "\n\n".join(Procedures.load(name) for name in names)

    def cmd_save_procedure(self, *args: str) -> str:
        return Procedures.run("save_procedure", list(args))

    def cmd_delete_procedure(self, *args: str) -> str:
        return Procedures.run("delete_procedure", list(args))

    def cmd_false_positive(self, parameter_string: str):
        info("We detected a false positive!")

        current_time = time.time()
        Commands.false_positive_timestamps.append(current_time)

        time_window_in_seconds = 10
        five_seconds_ago = current_time - time_window_in_seconds
        recent_count = sum(1 for ts in Commands.false_positive_timestamps if ts > five_seconds_ago)

        trace(f"False positives in last {time_window_in_seconds} seconds: {recent_count}")

        if recent_count >= 2:
            self.report_fp_flood()

    @staticmethod
    def report_fp_flood():
        if Commands.false_positive_reporter is None:
            info("No voice interface — false-positive flood report ignored.")
            return
        note("False positive flood! We're listening to gibberish - reporting to the voice interface.")
        Commands.false_positive_reporter()

    def cmd_turn_on(self, parameter_string: str) -> str:
        return home.active().turn_on(self._home_name(parameter_string))

    def cmd_turn_off(self, parameter_string: str) -> str:
        return home.active().turn_off(self._home_name(parameter_string))

    def cmd_dim(self, *args: str) -> str:
        if len(args) < 2:
            return "dim failed: expected 'name, level' parameters"
        try:
            level = int(float(args[1].strip().rstrip("%")))
        except ValueError:
            return f"dim failed: '{args[1].strip()}' is not a level"
        return home.active().dim(self._home_name(args[0]), level)

    def fire_and_forget(self, function, params):
        def wrapper():
            try:
                function(params)
            except Exception as e:
                error(f"Error in fire_and_forget thread: {e}")

        threading.Thread(target=wrapper).start()

    # --- background commands -------------------------------------------------------------
    # A command that has to wait on the outside world (a search, a page fetch, IMAP) is
    # handed to a lane instead of running on the command worker. Lanes:
    #   IO_LANE      — runs concurrently, capped by _IO_SLOTS across all Commands instances.
    #                  For independent HTTP / file work: an LLM reply that emits google(),
    #                  search_x() and search_email() together waits for the slowest one,
    #                  not the sum.
    #   BROWSER_LANE — strictly one at a time, in submission order. Every browser command
    #                  drives the one shared Chrome tab (open_url's tier-1 fetch included),
    #                  and interleaving two of them wedges both.
    #   CONFIG_LANE  — strictly one at a time, in submission order. A reply that emits
    #                  config_set("icloud.devices[0].id"), …[1].id and restart_integration
    #                  together only works in that order: each write reads config.yaml back
    #                  before the next, and the restart must see them all (2026-09-08 log:
    #                  run in parallel, seven of eight writes were refused or lost).
    # All lanes feed one batch: results are published together, in submission order, when
    # the last outstanding command of the batch finishes.
    IO_LANE = "io"
    BROWSER_LANE = "browser"
    CONFIG_LANE = "config"
    SERIAL_LANES = frozenset({BROWSER_LANE, CONFIG_LANE})
    MAX_PARALLEL_IO = 5
    _IO_SLOTS = threading.BoundedSemaphore(MAX_PARALLEL_IO)

    # Runs on: command worker thread. Hands the work to a lane thread.
    def background_command(self, function, params, lane: str = IO_LANE):
        echo, self._current_echo = self._current_echo, None
        with self.bg_lock:
            seq = self._bg_seq
            self._bg_seq += 1
            # _bg_outstanding is the source of truth for "a batch is in flight": the last
            # command out decrements it to zero under this lock and publishes there, so a
            # submit either joins the running batch or starts a fresh one — never lands
            # between the two.
            if self._bg_outstanding == 0:
                self.command_processor.background_command_start()
                # Start a fresh result set only when there is no reply still waiting to be
                # consumed. If a reply is pending (an earlier batch in this same turn already
                # published), keep accumulating onto it so the combined publish never drops
                # the earlier results.
                if self.command_processor.command_reply is None:
                    self._bg_results = {}
                    self.bg_results_contain_error = False
                self._bg_batch_started = time.time()
                self._bg_batch_count = 0
                self._bg_busy_seconds = 0.0
                self._bg_idle.clear()
            self._bg_outstanding += 1
            task = (seq, function, params, echo)
            if lane in self.SERIAL_LANES:
                self._serial_queues.setdefault(lane, queue.Queue()).put(task)
                if lane not in self._serial_workers:
                    worker = threading.Thread(target=self._serial_lane_worker, args=(lane,),
                                              name=f"Bg{lane.capitalize()}Worker", daemon=True)
                    self._serial_workers[lane] = worker
                    worker.start()
            else:
                threading.Thread(target=self._io_lane_run, args=(task,),
                                 name=f"BgCommand-{seq}", daemon=True).start()

    def wait_for_background(self, timeout: Optional[float] = None) -> bool:
        """Blocks until every background command of this instance has finished and the
        batch is published. True when idle, False on timeout."""
        return self._bg_idle.wait(timeout)

    def background_outstanding(self) -> int:
        with self.bg_lock:
            return self._bg_outstanding

    # Runs on: its own daemon thread, one per IO_LANE command.
    def _io_lane_run(self, task):
        with self._IO_SLOTS:
            self._run_one(*task)

    # Runs on: a serial lane's daemon thread. Drains that lane's queue in order and
    # retires when it is empty.
    def _serial_lane_worker(self, lane: str):
        tasks = self._serial_queues[lane]
        while True:
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                # Retire atomically: re-check under the lock so a concurrent
                # background_command() either enqueues before we retire (we keep going)
                # or spawns a fresh worker after we've dropped this one.
                with self.bg_lock:
                    if tasks.empty():
                        del self._serial_workers[lane]
                        return
                continue
            self._run_one(*task)
            tasks.task_done()

    # Runs on: a lane thread. Executes one command, completes its chat echo and files
    # its result; the last result of the batch triggers the publish.
    def _run_one(self, seq: int, function, params, echo):
        started = time.time()
        result = None
        failed = True
        try:
            # When the caller passes a list/tuple, splat as positional args so
            # multi-arg helpers (e.g. _run_send_email(to, subject, body)) work.
            if isinstance(params, (list, tuple)):
                result = function(*params)
            else:
                result = function(params)
            failed = False
        except Exception as e:
            # Never let a failing command take the batch down with it — that would leave
            # command_running_in_background stuck True and hang the request. Surface
            # the error to the LLM instead.
            error(f"Background command {getattr(function, '__name__', function)} failed: {e}")
            result = f"Command failed: {e}"
        finally:
            if echo is not None:
                self._complete_echo(echo, describe_completion(started, None if failed else result, failed=failed))
            self._finish_one(seq, result, failed, time.time() - started)

    def _finish_one(self, seq: int, result, failed: bool, elapsed: float):
        with self.bg_lock:
            self._bg_results[seq] = result
            if failed:
                self.bg_results_contain_error = True
            self._bg_batch_count += 1
            self._bg_busy_seconds += elapsed
            self._bg_outstanding -= 1
            if self._bg_outstanding > 0:
                return
            wall = time.time() - self._bg_batch_started
            if self._bg_batch_count > 1:
                info(f"{self._bg_batch_count} background commands done in {wall:.1f}s "
                     f"(would have taken {self._bg_busy_seconds:.1f}s one after another).")
            self._publish_bg_results()
            self.command_processor.background_command_done()
            self._bg_idle.set()

    @staticmethod
    def _complete_echo(echo, detail: str):
        try:
            notifier().complete_command_notification(echo, detail)
        except Exception as e:
            error(f"Failed to complete command notification: {e}")

    @property
    def accumulated_bg_results(self) -> list:
        """The batch's results so far, in submission order — the order the LLM emitted
        the commands, whatever order they finished in."""
        return [self._bg_results[k] for k in sorted(self._bg_results)]

    # Runs on: a lane thread, under bg_lock. Combines everything accumulated this turn and
    # publishes it as one reply. _bg_results holds all results for the turn
    # (background_command only resets it when no reply is pending), so the overwriting
    # set_command_reply here can't drop an earlier batch's results.
    def _publish_bg_results(self):
        cp = self.command_processor
        contains_error = self.bg_results_contain_error
        results = self.accumulated_bg_results
        list_results = [r for r in results if isinstance(r, list)]
        str_results = [r for r in results if isinstance(r, str) and r.strip()]
        # A sync command error appended earlier this round (append_command_reply) lives
        # only in command_reply, not in the batch — fold it into the publish so the
        # wholesale replace can't drop the error text or its framing.
        if cp.command_reply_contains_error and cp.command_reply is not None:
            contains_error = True
            if isinstance(cp.command_reply, list):
                list_results.insert(0, cp.command_reply)
            else:
                str_results.insert(0, cp.command_reply)
        if list_results:
            combined = []
            for lr in list_results:
                combined.extend(lr)
            if str_results:
                combined.append({"type": "text", "text": "\n\n".join(str_results)})
            cp.set_command_reply(combined, contains_error=contains_error)
        elif str_results:
            cp.set_command_reply("\n\n".join(str_results), contains_error=contains_error)

    def list_commands(self) -> list[str]:
        # Name check first: getattr on every attribute would evaluate the properties too, and
        # `web_access` builds the WebAccess singleton (and its Chrome loop thread) on first use.
        return [func.replace("cmd_", "", 1) for func in dir(self) if
                func.startswith("cmd_") and callable(getattr(self, func))]

    def is_sensitive(self, command_name: str) -> bool:
        return command_name in self.SENSITIVE_COMMANDS

    def approval_note(self, command_name: str) -> str:
        return self.APPROVAL_NOTES.get(command_name, "")

    def is_pure(self, command_name: str) -> bool:
        return command_name in self.PURE_COMMANDS

    # --- toolkits: the command references the LLM loads on demand (plans/TOOLKITS_PLAN.md) ---

    @staticmethod
    def _prompt_manager():
        """The live PromptManager, reached the way the bug-fix commands reach the history;
        None in a process without one (tests)."""
        aam = AutonomousActionManager.current()
        return aam.prompt_manager if aam is not None else None

    def cmd_load_toolkit(self, *names: str):
        self.background_command(self._run_load_toolkit, list(names))

    def _run_load_toolkit(self, *names: str) -> str:
        """The references of the named toolkits, each already in the LLM's history window
        answered with a pointer to it instead of the text again. Splatted like every
        list-parameter background command (`function(*params)`)."""
        names = [n for n in names if n and n.strip()]
        if not names:
            return "load_toolkit failed: expected a toolkit name. Available: " + ", ".join(Toolkits.names())
        prompt_manager = self._prompt_manager()
        in_window = prompt_manager.message_history.loaded_toolkits() if prompt_manager else {}
        # The autonomous check's rounds live in its own messages, never in the history: what
        # it is sent must not become the baseline the history's readers are diffed against.
        record = self._current_profile is not AUTONOMOUS
        parts, loaded, pointed, unknown = [], [], [], []
        for name in names:
            toolkit = Toolkits.by_name(name)
            if toolkit is None:
                unknown.append(name.strip())
                parts.append(f"No toolkit named '{name.strip()}'. Available: {', '.join(Toolkits.names())}")
            elif toolkit.name in in_window:
                # The reference is static, the carried data is not: a pointer back to the
                # reference, the data again as it is now.
                pointed.append(toolkit.name)
                at = time.strftime("%H:%M", time.localtime(in_window[toolkit.name]))
                parts.append(f"The {toolkit.name} toolkit was loaded at {at}: its reference is above in this "
                             f"conversation, use it from there." + self._carried_data(toolkit, prompt_manager, record))
            else:
                loaded.append(toolkit.name)
                parts.append(ToolkitPreloader.render_load(toolkit, prompt_manager, record=record))
        result = "\n\n".join(parts)
        info(f"load_toolkit: loaded {loaded or 'nothing'}, pointed back to {pointed or 'nothing'}, "
             f"unknown {unknown or 'none'} ({len(result)} chars, channel {self._current_profile.name}"
             f"{'' if record else ', baseline untouched'})")
        return result

    @staticmethod
    def _carried_data(toolkit, prompt_manager, record: bool = True) -> str:
        """The toolkit's carried sections (the home data) to send after its reference, current
        as of now (PromptManager.carried_data); empty for a toolkit that carries none, or
        without a PromptManager to take them from."""
        if not toolkit.sections or prompt_manager is None:
            return ""
        return "\n\n" + prompt_manager.carried_data(toolkit, record=record)

    @staticmethod
    def _short_url(u: str, max_len: int = 50) -> str:
        u = (u or "").strip()
        if len(u) <= max_len:
            return u
        try:
            parsed = urllib.parse.urlparse(u)
            host = parsed.netloc or u
            path = parsed.path or ""
            head = host + path
            if len(head) <= max_len:
                return head
            return head[:max_len]
        except Exception:
            return u[:max_len]

    # Returns a human-readable progress string for the given command, or None
    # if no describe_X method exists or the description is intentionally empty.
    # Accepts either a list[str] (canonical) or a str (legacy callers); the
    # describe_* methods themselves all take a single joined string for simplicity.
    def describe(self, command: str, params) -> Optional[str]:
        command = command.strip()
        f = getattr(self, "describe_" + command, None)
        if f is None:
            return None
        joined = ", ".join(params) if isinstance(params, list) else (params or "")
        try:
            text = f(joined)
            return text if text else None
        except Exception as e:
            error(f"describe_{command} raised: {e}")
            return None

    def echo_text(self, command: str, params) -> Optional[str]:
        """What a chat shows for a command call: its description, the call itself for a
        command without a describe_ method, None for one whose description is empty."""
        text = self.describe(command, params)
        if text is None and not hasattr(self, "describe_" + command.strip()):
            params_display = ", ".join(params) if isinstance(params, list) else str(params)
            return f"{command}({params_display})"
        return text

    @staticmethod
    def _first_argument(p: str) -> str:
        return p.split(",", 1)[0].strip().strip("\"'`")

    def describe_weather(self, p):                return "Checking the weather"
    def describe_home_automation_logs(self, p):   return "Reading home automation logs"
    def describe_server_logs(self, p):            return "Reading server logs"
    def describe_open_url(self, p):               return f"Reading {self._short_url(p)}"
    def describe_google(self, p):                 return f"Searching for {p.strip()}"
    def describe_search_x(self, p):               return f"Searching X for {p.strip()}" if p.strip() else "Searching X"
    def describe_browse_open(self, p):            return f"Browsing {self._short_url(p)}"
    def describe_browse_eval(self, p):            return "Inspecting the page"
    def describe_browse_snapshot(self, p):        return "Re-reading the page"
    def describe_browse_click_at(self, p):        return f"Clicking at {p.strip()}" if p.strip() else "Clicking"
    def describe_browse_extract(self, p):         return "Extracting page text"
    def describe_browse_network(self, p):         return "Reading the page's network traffic"
    def describe_browse_wait(self, p):
        pattern, timeout = WebAccess._parse_wait_spec(p)
        timeout = max(1.0, min(timeout, WebAccess._WAIT_MAX_S))
        return f"Waiting up to {timeout:.0f}s for '{pattern}'" if pattern else "Waiting for the page"
    def describe_browse_screenshot(self, p):      return "Taking a screenshot"
    def describe_send_screenshot(self, p):        return "Sending screenshot"
    def describe_browse_close(self, p):           return "Closing browser"
    def describe_latest_email(self, p):
        parts = [x.strip().strip('"').strip("'") for x in p.split(",")] if p else []
        folder = parts[0] if parts and parts[0] else ""
        unread = len(parts) > 1 and parts[1].lower() in self._UNREAD_TOKENS
        scope = "unread email" if unread else "email"
        return f"Checking {scope}" + (f" in {folder}" if folder else "")
    def describe_search_email(self, p):           return f"Searching email for {p.strip()}"
    def _friendly_email_subject(self, email_id):
        """Cached subject for an email id, trimmed and truncated for a progress
        notification, or None when no usable subject is known (cache miss,
        "(no subject)" placeholder, or lookup error). Never raises and never
        leaks the raw internal id, so callers can safely fall back to a generic
        message."""
        if not email_id:
            return None
        try:
            subject = self._email_provider.subject_for(email_id)
        except Exception as e:
            trace(f"_friendly_email_subject: subject lookup failed for '{email_id}': {e}")
            return None
        if subject and subject.strip() and subject.strip() != "(no subject)":
            s = subject.strip()
            if len(s) > 50:
                s = s[:50].rstrip() + "…"
            return s
        return None
    def describe_read_email(self, p):
        s = self._friendly_email_subject(p)
        return f"Reading email '{s}'" if s else "Reading email…"
    def describe_move_email(self, p):
        if not p:
            return "Moving email to folder"
        head, sep, tail = p.rpartition(",")
        folder = (tail if sep else p).strip()
        email_id = head.strip() if sep else ""
        s = self._friendly_email_subject(email_id)
        if s:
            return f"Moving email '{s}' to {folder}"
        return f"Moving email to {folder}"
    def describe_list_folders(self, p):           return "Listing email folders"
    def describe_schedule_wakeup(self, p):
        minutes, _, reason = p.partition(",")
        return f"Scheduling wakeup in {minutes.strip()} min" + (f": {reason.strip()}" if reason.strip() else "")
    def describe_caffeinate(self, p):             return "Staying awake"
    def describe_tesla_max_defrost(self, p):      return "Turning on Tesla max defrost"
    def describe_tesla_air_con_on(self, p):       return "Turning on Tesla A/C"
    def describe_tesla_air_con_off(self, p):      return "Turning off Tesla A/C"
    def describe_tesla_open_trunk(self, p):       return "Opening Tesla trunk"
    def describe_show_photo(self, p):             return f"Showing photo {p.strip()}"
    def describe_add_shopping_list_item(self, p): return f"Adding {p.strip()} to the shopping list"
    def describe_find_phone(self, p):             return "Pinging phone"
    def describe_locate_user(self, p):            return f"Locating {p.strip()}" if p.strip() else "Locating user"
    def describe_icloud_devices(self, p):         return "Listing iCloud devices"
    def describe_send_imessage(self, p):
        recipient = p.split(",", 1)[0].strip() if p else ""
        if not recipient:
            return "Sending iMessage"
        try:
            from assistant.integration.integrations.IMessageIntegration import IMessageIntegration
            imessage = IMessageIntegration.get_instance()
            handle, err = imessage.resolve_recipient(recipient)
            if handle and not err and handle.lower() != recipient.lower():
                display = imessage.display_name_for_handle(handle) or recipient
                return f"Sending iMessage to {display} ({handle})"
        except Exception as e:
            trace(f"describe_send_imessage: handle lookup failed for '{recipient}': {e}")
        return f"Sending iMessage to {recipient}"
    def describe_send_email(self, p):
        recipient = p.split(",", 1)[0].strip() if p else ""
        return f"Sending email to {recipient}" if recipient else "Sending email"
    def describe_fetch_camera_images(self, p):    return "Fetching camera images"
    def describe_self_analysis(self, p):          return "Running self-analysis"
    def describe_self_improvement(self, p):       return "Running self-improvement"
    def describe_restart_server(self, p):         return "Restarting server"
    def describe_execute(self, p):                return f"Running action {p.strip()}"
    def describe_search_music(self, p):           return f"Searching music for {p.strip()}"
    def describe_play(self, p):                   return f"Playing in {p.strip()}" if p.strip() else "Playing"
    def describe_pause(self, p):                  return f"Pausing in {p.strip()}" if p.strip() else "Pausing"
    def describe_next(self, p):                   return f"Skipping to next track in {p.strip()}" if p.strip() else "Skipping to next track"
    def describe_volume_up(self, p):              return f"Turning up the volume in {p.strip()}" if p.strip() else "Turning up the volume"
    def describe_volume_down(self, p):            return f"Turning down the volume in {p.strip()}" if p.strip() else "Turning down the volume"
    def describe_vault_add(self, p):
        # Never the value: the notification is exactly where a secret must not be echoed.
        parts = [x.strip() for x in p.split(",")]
        if len(parts) >= 2 and parts[1]:
            return f"Adding {parts[1]} to the vault ({parts[0]})"
        return "Adding a secret to the vault"
    def describe_vault_request(self, p):      return None  # the masked prompt itself says what is asked for
    def describe_list_integrations(self, p):  return "Listing integrations"
    def describe_config_help(self, p):        return f"Reading the settings for {p.strip()}" if p.strip() else "Listing the settings sections"
    def describe_config_request(self, p):     return f"Asking for setting {p.strip()}" if p.strip() else "Asking for a setting"
    def describe_telegram_login(self, p):     return "Signing the Telegram account in"
    def describe_config_set(self, p):
        key = p.split(",", 1)[0].strip().strip('"\'')
        return f"Setting {key}" if key else "Setting a value"
    def describe_restart_integration(self, p): return f"Restarting {p.strip()}" if p.strip() else "Restarting an integration"
    def describe_enable_integration(self, p):  return f"Enabling {p.strip()}" if p.strip() else "Enabling an integration"
    def describe_disable_integration(self, p): return f"Disabling {p.strip()}" if p.strip() else "Disabling an integration"
    def describe_remember(self, p):               return f"Saving to memory: {p.strip()}" if p.strip() else "Saving to memory"
    def describe_load_calendar(self, p):          return f"Looking at the calendar: {self._first_argument(p)}" if p.strip() else "Looking at a calendar"
    def describe_load_watch(self, p):             return f"Looking at: {self._first_argument(p)}" if p.strip() else "Looking at a watch item"
    def describe_save_watch(self, p):             return f"Watching: {self._first_argument(p)}" if p.strip() else "Saving a watch item"
    def describe_checked_watch(self, p):          return f"Checked: {self._first_argument(p)}" if p.strip() else "Marking a watch item checked"
    def describe_delete_watch(self, p):           return f"No longer watching: {self._first_argument(p)}" if p.strip() else "Deleting a watch item"
    def describe_load_procedure(self, p):         return f"Loading the {self._first_argument(p)} procedure" if p.strip() else "Loading a procedure"
    def describe_save_procedure(self, p):         return f"Saving the {self._first_argument(p)} procedure" if p.strip() else "Saving a procedure"
    def describe_delete_procedure(self, p):       return f"Deleting the {self._first_argument(p)} procedure" if p.strip() else "Deleting a procedure"
    def describe_false_positive(self, p):         return None  # noisy / dev-only — suppress
    def describe_load_toolkit(self, p):
        names = [n.strip().strip("\"'`") for n in p.split(",") if n.strip()]
        if not names:
            return "Loading a toolkit"
        if len(names) == 1:
            return f"Loading the {names[0]} toolkit"
        return f"Loading the {', '.join(names[:-1])} and {names[-1]} toolkits"
    def describe_turn_on(self, p):                return f"Turning on {p.strip()}"
    def describe_turn_off(self, p):               return f"Turning off {p.strip()}"

    def describe_dim(self, p):
        parts = [x.strip() for x in p.split(",")]
        if len(parts) >= 2:
            return f"Dimming {parts[0]} to {parts[1]}%"
        return f"Dimming {p.strip()}"

    def describe_browse_act(self, p):
        spec = (p or '').strip()
        ops = WebAccess._split_act_batch(spec)
        if len(ops) > 1:
            return ' · '.join(self._describe_one_act(op) for op in ops)
        return self._describe_one_act(spec)

    def _describe_one_act(self, spec):
        idx_str, _, value = spec.partition('=') if '=' in spec else (spec, '', None)
        idx_str = idx_str.strip()
        try:
            idx = int(idx_str)
        except ValueError:
            return f"Acting on {spec}"
        label = self.web_access.get_snapshot_label(idx)
        target = label if label else f"element {idx}"
        if value is not None:
            value = value.strip()
            if value.endswith('\\n'):
                return f"Filling {target} with '{value[:-2].rstrip()}' and pressing Enter"
            return f"Filling {target} with '{value}'"
        return f"Clicking on {target}"

    def describe_browse_key(self, p):
        spec = (p or '').strip()
        if ',' in spec:
            idx_str, _, key = spec.partition(',')
            key = key.strip()
            try:
                idx = int(idx_str.strip())
            except ValueError:
                return f"Pressing {spec}"
            label = self.web_access.get_snapshot_label(idx)
            target = label if label else f"element {idx}"
            return f"Pressing {key} on {target}"
        return f"Pressing {spec}"

    def describe_browse_inspect(self, p):
        spec = (p or '').strip()
        idx_str, _, attr = spec.partition(',') if ',' in spec else (spec, '', '')
        idx_str = idx_str.strip()
        attr = attr.strip()
        try:
            idx = int(idx_str)
        except ValueError:
            return f"Inspecting {spec}"
        label = self.web_access.get_snapshot_label(idx)
        target = label if label else f"element {idx}"
        if attr:
            return f"Reading '{attr}' of {target}"
        return f"Inspecting {target}"

    def describe_search_photos(self, p):
        kw = p.split(",", 1)[0].strip()
        return f"Searching photos for {kw}" if kw and kw != "*" else "Browsing photos"

    def describe_search_history(self, p):
        q = p.split(",", 1)[0].strip()
        return f"Searching history for {q}" if q else "Searching history"

    def describe_show_history(self, p):
        parts = [x.strip() for x in p.split(",", 1)]
        if len(parts) == 2:
            return f"Reading history from {parts[0]} to {parts[1]}"
        return "Reading history"

    def describe_create_event(self, p):
        title = p.split(",", 1)[0].strip()
        return f"Creating calendar event '{title}'" if title else "Creating calendar event"

    def describe_play_music(self, p):
        room = p.split(",", 1)[0].strip()
        return f"Playing music in {room}" if room else "Playing music"

    def describe_sonos_enqueue_and_play(self, p):
        parts = [x.strip() for x in p.split(",", 1)]
        if len(parts) == 2:
            return f"Playing {parts[1]} in {parts[0]}"
        return "Playing music"

    # Runs on: command worker thread (called from CommandProcessor._command_worker)
    def run(self, command: str, params: list[str]) -> str:
        command = command.strip()

        if command == "dim" and len(params) >= 2 and params[1].strip() == "0":
            trace("Changing dim command to turn_off since dim value is zero and we want clean data in stats.")
            command = "turn_off"
            params = [params[0]]

        params_display = ", ".join(params)
        note("Running command: " + command + "(" + params_display + ")")
        f = getattr(self, "cmd_" + command, None)
        if f is None:
            error(f"Unknown command: {command}")
            return f"Unknown command: {command}"

        try:
            if command in self.MULTI_PARAM_COMMANDS:
                result = f(*params)
            else:
                result = f(params[0] if params else "")
            return result if isinstance(result, str) else ""
        except Exception as e:
            error(f"Exception running command {command}({params_display}): {e}")
            return str(e)
