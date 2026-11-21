from dataclasses import dataclass
import os
from typing import Optional

from assistant.ChannelProfile import PHOTO_TARGET_SCREEN
from assistant.commands.CommandSet import CommandSet
from assistant.commands.Commands import MEDIA_NOT_DELIVERED
from assistant.Notifier import notifier
from common.ConfigManager import config
from common.FloxyDatabase import FloxyDatabase
from common.logging import trace, info, error, fine


@dataclass(frozen=True)
class MediaPayload:
    """One library asset ready to be sent: JPEG bytes for a still, an MP4 rendition for a video."""
    data: bytes
    caption: str
    uuid: str
    kind: str  # "photo" | "video"
    thumbnail: Optional[bytes] = None  # small JPEG poster frame, videos only


class PhotosCommands(CommandSet):
    """The photo library's commands: search the indexed photos and videos, and show some —
    in the chat that asked, or full screen on the server's display for a spoken request when
    `photos.show_on_screen` is on (`photos.screen_prepare_command` runs first, e.g. to switch
    the TV to the Mac's input); off, a spoken request's photo goes to the chat too."""
    MULTI_PARAM_COMMANDS = frozenset({"show_photo", "search_photos"})

    def cmd_show_photo(self, *args: str):
        route = self.commands.request_route()
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

        if self._shows_on_screen():
            return self._show_photo_on_screen(payload.uuid, payload.caption)

        if payload.kind == "video":
            sent = notifier().send_video(payload.data, caption=payload.caption, thumbnail=payload.thumbnail,
                                         **(route or {}))
        else:
            sent = notifier().send_photo(payload.data, caption=payload.caption, **(route or {}))
        if not sent:
            info(f"show_photo: no chat took {payload.kind} {photo_id}")
            return f"show_photo {MEDIA_NOT_DELIVERED}"
        info(f"Sent {payload.kind} {photo_id} to notifier: {payload.caption}")
        return ""

    def _run_show_photo_album(self, photo_ids: list[int], route: Optional[dict] = None) -> str:
        if self._shows_on_screen():
            info(f"{self.profile.name}: showing first photo only from album of {len(photo_ids)} (one screen)")
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
            info(f"show_photo: no chat took all of {len(album_photos)} photo(s) and {len(videos)} video(s)")
            return f"show_photo {MEDIA_NOT_DELIVERED}"

        total = len(album_photos) + len(videos)
        info(f"Sent {len(album_photos)} photos as album and {len(videos)} video(s) to notifier")
        if errors:
            return f"Sent {total} photos (skipped: {'; '.join(errors)})"
        return ""

    @staticmethod
    def _open_photos_db_connection():
        import sqlite3 as _sqlite3
        from integrations.photos.PhotoProcessor import PHOTOS_SQLITE_PATH

        return _sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)

    @staticmethod
    def _fetch_photo_asset_row(cursor, asset_guid: str):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        cursor.execute(PhotoProcessor.ASSET_FILE_SQL, (asset_guid,))
        return cursor.fetchone()

    # Telegram's bot upload limit; the 720p iCloud rendition of a clip under ~2 min fits.
    TELEGRAM_VIDEO_MAX_BYTES = 50 * 1024 * 1024

    def _prepare_photo_send_payload(self, photo_id: int, cursor=None) -> tuple[MediaPayload | None, str | None]:
        from integrations.photos.PhotoProcessor import PhotoProcessor
        from integrations.photos.ICloudPhotoDownloader import (
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
        from integrations.photos.ICloudPhotoDownloader import ICloudPhotoDownloader
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

    def _shows_on_screen(self) -> bool:
        """A spoken request's photo goes to the server's screen only when the install says it has
        one; without `photos.show_on_screen` it goes to the chat like everyone else's."""
        if self.profile.photo_target != PHOTO_TARGET_SCREEN:
            return False
        if config.get("photos.show_on_screen", False):
            return True
        info(f"{self.profile.name}: photos.show_on_screen is off, sending the photo to the chat instead of the screen")
        return False

    @staticmethod
    def _show_photo_on_screen(uuid: str, caption: str) -> str:
        import subprocess

        prepare = (config.get("photos.screen_prepare_command", "") or "").strip()
        if prepare:
            trace(f"Preparing the screen: {prepare}")
            try:
                result = subprocess.run(prepare, shell=True, timeout=10, capture_output=True, text=True)
                fine(f"screen_prepare_command stdout: {result.stdout.strip()}")
                fine(f"screen_prepare_command stderr: {result.stderr.strip()}")
                if result.returncode != 0:
                    error(f"screen_prepare_command exited with code {result.returncode}")
                    return (f"show_photo failed: photos.screen_prepare_command exited with code "
                            f"{result.returncode}: {result.stderr.strip()}")
            except Exception as e:
                error(f"screen_prepare_command failed: {e}")
                return f"show_photo failed: could not prepare the screen: {e}"
        else:
            trace("No photos.screen_prepare_command, opening the photo as the screen is")

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

        info(f"Showing photo {uuid} on the screen: {caption}")
        return ""

    def cmd_search_photos(self, *args: str):
        self.background_command(self._run_search_photos, list(args))

    def _run_search_photos(self, *args: str) -> str:
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
            lines.append(f'{r["id"]}: {r["short_description"]}{self.commands.media_tag(r)} ({location}, {date_str}) [pretty: {pretty}, family: {family}]{persons_tag}')
        info(f"Photo search for '{keywords}' returned {len(results)} results")
        return "\n".join(lines)

    def describe_show_photo(self, p):             return f"Showing photo {p.strip()}"
    def describe_search_photos(self, p):
        kw = p.split(",", 1)[0].strip()
        return f"Searching photos for {kw}" if kw and kw != "*" else "Browsing photos"
