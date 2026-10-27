import json
from pathlib import Path

import pytest

from integrations.podcasts.PodcastsConfig import PodcastsConfig


class DictConfig:
    """A ConfigManager stand-in answering from a flat dict of key paths."""

    def __init__(self, values: dict):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


@pytest.fixture
def show_dir(tmp_path) -> Path:
    path = tmp_path / "Output" / "Moonshots"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def values(tmp_path, show_dir) -> dict:
    return {
        "podcasts.shows": [{"name": "Moonshots", "playlist_url": "https://www.youtube.com/playlist?list=PLexample",
                            "output_folder": str(show_dir)}],
        "podcasts.state_dir": str(tmp_path / "state"),
        "podcasts.openai_api_key": "sk-test-example",
    }


@pytest.fixture
def config(values) -> PodcastsConfig:
    return PodcastsConfig(DictConfig(values))


def write_bundle(root: Path, slug: str = "moonshots-412", video_id: str = "vid412",
                 chapters: tuple = (("Opening", 0, "[0:00]  Hello there. >> Welcome to the show."),
                                    ("Fusion", 125, "[2:05]  The tokamak reached Q of 1.5 at Alderbank Labs."))) -> Path:
    bundle = root / slug
    (bundle / "chapters").mkdir(parents=True)
    meta = {"video_id": video_id, "slug": slug, "title": f"Robots Learn To Juggle | EP #{slug.split('-')[-1]}",
            "uploader": "Example Channel", "upload_date": "20261020", "duration_s": 3600,
            "video_url": f"https://www.youtube.com/watch?v={video_id}", "chapters": [],
            "transcript_source": "whisper+youtube-turns", "timings": {"download_s": 12.0, "transcribe_s": 200.0}}
    (bundle / "video_meta.json").write_text(json.dumps(meta))
    for index, (title, start, body) in enumerate(chapters):
        minutes, seconds = divmod(start, 60)
        (bundle / "chapters" / f"{index:02d}_{minutes}-{seconds:02d}.txt").write_text(
            f"CHAPTER: {title}\nTIME: {minutes}:{seconds:02d} - 59:00\n\n\n{body}\n")
    return bundle
