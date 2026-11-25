"""A stand-in for the Drama 3 model and its folders, for the Fish Local TTS tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from common import Platform
from interfaces.voice.tts import FishLocalModel as fish

SMALL = 1000   # bytes of a build in these tests


class FakeModel:
    """What mlx-audio's model is to FishLocalModel: `sample_rate` and a `generate` that yields
    results with `.audio`. Records every call; `pieces` per call, `fail` raises instead."""
    sample_rate = 44100

    def __init__(self, pieces=3, fail=None):
        self.pieces, self.fail = pieces, fail
        self.calls = []
        self.closed = 0

    def generate(self, text, ref_audio=None, ref_text=None, stream=False, first_chunk_frames=None):
        self.calls.append(SimpleNamespace(text=text, ref_audio=ref_audio, ref_text=ref_text, stream=stream))
        if self.fail:
            raise self.fail

        def results():
            try:
                for _ in range(self.pieces):
                    yield SimpleNamespace(audio=np.full(441, 0.2, dtype=np.float32))
            finally:
                self.closed += 1
        return results()

    def spoken(self):
        """The texts asked for besides the voice clip and the warm-up."""
        return [c.text for c in self.calls if c.text not in (fish.VOICE_TEXT, fish.WARMUP_TEXT)]


@pytest.fixture
def fish_home(tmp_path, monkeypatch):
    """A capable Mac with empty folders, small builds, a fake hub and a fake model:
    `home.model`, `home.hub` (the builds fetched), `home.fill(quant, n)` puts n bytes of a build
    on disk, `home.have(quant)` a complete one."""
    monkeypatch.setattr(fish, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(fish, "VOICE_DIR", tmp_path / "voice")
    monkeypatch.setattr(fish, "BUILDS", {quant: fish.Build(build.repo, build.revision, SMALL, build.min_memory_gb)
                                         for quant, build in fish.BUILDS.items()})
    monkeypatch.setattr(Platform, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(Platform, "memory_gb", lambda: 256)
    monkeypatch.setattr(fish, "free_gb", lambda: 500)
    home = SimpleNamespace(model=FakeModel(), hub=[], hub_error=None, tmp=tmp_path)

    def download(build, target):
        home.hub.append(build)
        if home.hub_error:
            raise home.hub_error
        (target / "weights.safetensors").write_bytes(b"0" * build.size)

    def fill(quant, size):
        folder = fish.model_dir(quant)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "weights.safetensors").write_bytes(b"0" * size)

    def have(quant=fish.DEFAULT_QUANT):
        fill(quant, SMALL)
        fish.marker(quant).write_text(fish.BUILDS[quant].revision)

    monkeypatch.setattr(fish, "_hub_download", download)
    monkeypatch.setattr(fish, "_mlx_load", lambda path: home.model)
    home.fill, home.have = fill, have
    yield home
    fish.FishLocalModel.reset()
