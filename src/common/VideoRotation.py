"""Bakes a clip's display rotation into its pixels, for players that read the coded frames and drop the matrix.

On a Mac the re-encode is AVFoundation's; on any other system it is `ffmpeg`'s, which applies the
display matrix by itself when it re-encodes."""
import importlib
import os
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Optional

from common import Platform
from common.logging import error, info, trace

EXPORT_TIMEOUT_SECONDS = 180
# The frame fitted into 1280x1280 without ever growing — what the Mac's 720p preset does for
# either orientation — and even sides, which H.264 needs.
_FFMPEG_SCALE = "scale='min(1280,iw)':'min(1280,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"
_STATUS_COMPLETED = 3  # AVAssetExportSessionStatusCompleted

_FRAMEWORK_NAMES = {
    "AVFoundation": ("AVURLAsset", "AVMediaTypeVideo", "AVMutableVideoCompositionLayerInstruction",
                     "AVMutableVideoCompositionInstruction", "AVMutableVideoComposition", "AVAssetExportSession",
                     "AVAssetExportPreset1280x720", "AVFileTypeMPEG4"),
    "CoreMedia": ("CMTimeMake", "CMTimeRangeMake", "kCMTimeZero"),
    "Foundation": ("NSURL",),
    "Quartz": ("CGAffineTransformConcat", "CGAffineTransformMakeTranslation", "CGRectApplyAffineTransform",
               "CGRectMake"),
}
_frameworks_lock = threading.Lock()
_frameworks: Optional[SimpleNamespace] = None


def _load_frameworks() -> SimpleNamespace:
    """Every AVFoundation, CoreMedia, Foundation and Quartz name the export uses, resolved once.

    pyobjc binds a framework's names lazily, on first access, and that first lookup is not
    thread-safe: it checks a shared dict and pops from it, so two threads resolving the same name
    at once leave one of them with a bare KeyError. Two clips sent in one reply are re-encoded in
    parallel, and did exactly that (KeyError('CGAffineTransformConcat')); resolving under a lock,
    once, keeps the lazy lookups out of the export itself. Raises ImportError where the frameworks
    are missing (not a Mac); a failure is not cached, so the next call tries again."""
    global _frameworks
    with _frameworks_lock:
        if _frameworks is None:
            resolved = {}
            for module_name, names in _FRAMEWORK_NAMES.items():
                module = importlib.import_module(module_name)
                for name in names:
                    resolved[name] = getattr(module, name)
            _frameworks = SimpleNamespace(**resolved)
        return _frameworks


def _bake_with_ffmpeg(video_bytes: bytes, timeout: float) -> Optional[bytes]:
    """The same re-encode through ffmpeg, which rotates the frames by the display matrix and
    writes none when it re-encodes. None — the clip goes out as it is — when ffmpeg is missing,
    fails or takes longer than *timeout*."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="floxy-rotate-") as folder:
        source = os.path.join(folder, "source.mp4")
        target = os.path.join(folder, "upright.mp4")
        with open(source, "wb") as f:
            f.write(video_bytes)
        command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", source, "-vf", _FFMPEG_SCALE,
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-movflags", "+faststart", target]
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            error(f"VideoRotation: ffmpeg could not re-encode the clip ({type(e).__name__}: {e}), clip sent as it is")
            return None
        if result.returncode != 0 or not os.path.exists(target) or not os.path.getsize(target):
            detail = (result.stderr or b"").decode("utf-8", errors="replace").strip()[-300:]
            error(f"VideoRotation: ffmpeg failed (exit {result.returncode}: {detail}), clip sent as it is")
            return None
        with open(target, "rb") as f:
            upright = f.read()
    trace(f"VideoRotation: clip re-encoded by ffmpeg in {time.monotonic() - started:.1f}s "
          f"({len(video_bytes) / 1024 / 1024:.1f} -> {len(upright) / 1024 / 1024:.1f} MB)")
    return upright


def bake_rotation(video_bytes: bytes, timeout: float = EXPORT_TIMEOUT_SECONDS) -> Optional[bytes]:
    """*video_bytes* re-encoded so that the frames are stored the way the track's display matrix
    shows them and the matrix itself becomes identity. None when it cannot be done — AVFoundation
    missing (not a Mac), no video track, or an export that failed or timed out — so the caller
    sends the clip as it is.

    A phone's portrait clip is a landscape stream plus a 90° matrix; Telegram for iOS reads the
    matrix for the layout, then renders the unrotated raster into it. Only a clip whose matrix is
    not identity needs this, see `Mp4Metadata.mp4_is_rotated`."""
    if not Platform.is_mac():
        return _bake_with_ffmpeg(video_bytes, timeout)
    try:
        fw = _load_frameworks()
    except ImportError as e:
        info(f"VideoRotation: AVFoundation not available ({e}), clip sent as it is")
        return None

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="floxy-rotate-") as folder:
        source = os.path.join(folder, "source.mp4")
        target = os.path.join(folder, "upright.mp4")
        with open(source, "wb") as f:
            f.write(video_bytes)

        asset = fw.AVURLAsset.URLAssetWithURL_options_(fw.NSURL.fileURLWithPath_(source), None)
        tracks = asset.tracksWithMediaType_(fw.AVMediaTypeVideo)
        if not tracks:
            info("VideoRotation: no video track in the clip, sent as it is")
            return None
        track = tracks[0]
        natural = track.naturalSize()
        transform = track.preferredTransform()
        # The matrix maps the coded frame onto the display; the frame lands wherever its
        # translation says, so shift it back to the origin before rendering into a canvas of
        # the displayed size.
        shown = fw.CGRectApplyAffineTransform(fw.CGRectMake(0, 0, natural.width, natural.height), transform)
        transform = fw.CGAffineTransformConcat(transform, fw.CGAffineTransformMakeTranslation(-shown.origin.x, -shown.origin.y))
        width, height = round(abs(shown.size.width)), round(abs(shown.size.height))

        layer = fw.AVMutableVideoCompositionLayerInstruction.videoCompositionLayerInstructionWithAssetTrack_(track)
        layer.setTransform_atTime_(transform, fw.kCMTimeZero)
        instruction = fw.AVMutableVideoCompositionInstruction.videoCompositionInstruction()
        instruction.setTimeRange_(fw.CMTimeRangeMake(fw.kCMTimeZero, asset.duration()))
        instruction.setLayerInstructions_([layer])
        composition = fw.AVMutableVideoComposition.videoComposition()
        composition.setRenderSize_((width, height))
        frame_rate = track.nominalFrameRate() or 30
        composition.setFrameDuration_(fw.CMTimeMake(1000, round(frame_rate * 1000)))
        composition.setInstructions_([instruction])

        # The 720p preset fits either orientation (a 720x1280 canvas stays 720x1280) and caps the
        # bitrate, so a rendition near the chat's upload limit does not grow past it.
        export = fw.AVAssetExportSession.exportSessionWithAsset_presetName_(asset, fw.AVAssetExportPreset1280x720)
        if export is None:
            error("VideoRotation: AVFoundation refused to create an export session for the clip")
            return None
        export.setOutputURL_(fw.NSURL.fileURLWithPath_(target))
        export.setOutputFileType_(fw.AVFileTypeMPEG4)
        export.setVideoComposition_(composition)
        export.setShouldOptimizeForNetworkUse_(True)
        done = threading.Event()
        export.exportAsynchronouslyWithCompletionHandler_(done.set)
        if not done.wait(timeout):
            export.cancelExport()
            error(f"VideoRotation: export did not finish within {timeout:.0f}s, clip sent as it is")
            return None
        if export.status() != _STATUS_COMPLETED:
            err = export.error()
            error(f"VideoRotation: export failed (status {export.status()}, "
                  f"{err.localizedDescription() if err else 'no error given'}), clip sent as it is")
            return None
        with open(target, "rb") as f:
            upright = f.read()
    trace(f"VideoRotation: {round(natural.width)}x{round(natural.height)} frames re-encoded as {width}x{height} "
          f"in {time.monotonic() - started:.1f}s ({len(video_bytes) / 1024 / 1024:.1f} -> {len(upright) / 1024 / 1024:.1f} MB)")
    return upright
