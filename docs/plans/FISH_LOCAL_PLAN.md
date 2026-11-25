# Fish Local TTS — Drama 3 on the server's own Mac

A second local voice beside Supertonic: Fish Audio's open-weight Drama 3, run through MLX on the
server Mac. Nothing spoken leaves the machine, and the voice performs delivery tags (`[sigh]`,
`<whisper>…</whisper>`, `[laugh]`). The integration downloads the model by itself the first time
it is switched on.

Status: built (steps 1–4), off by default, on trial on one install. Not yet the voice of anything.

## What was measured

On an M5 Ultra with 256 GB, `mlx-audio` 0.6.8, the community conversions
`mlx-community/fish-audio-drama-3-{4bit,8bit}`, model loaded, standalone (no LLM running beside it):

| | 4-bit | 8-bit |
|---|---|---|
| Download | about 22 GB | about 42 GB |
| Peak memory | 23.5–26.4 GB | 40.6–43.5 GB |
| Speed | 1.9–2.1x real time | about 1.6x |
| First audio, streamed | 155–250 ms | 200–510 ms |
| Load, files in the page cache | 1.7 s | — |
| A new reference voice | about 1.4 s once, then cached | about 1 s |

4-bit is the default: faster, 17 GB lighter, and the 8-bit gain is unproven by ear.

The spike (step 1), same machine, 4-bit, inside one process with the rest:

- The local LLM generating beside it leaves Drama 3 at 1.7–1.95x real time; continuous Whisper
  large-v2 transcription drops it to 0.8–0.9x.
- Whisper and Drama 3 run from two threads of one process without a lock between them (20 of
  20 generations beside 110 transcriptions).
- A first piece of 4 frames let playback run dry once (245 ms) under LLM load; it is 8.
- The first generation after a load is slow (0.6 s to first audio, seconds under load) and a
  reference voice costs about 3.5 s at its first use — the warm-up after the load pays both.
- Without a reference the voice changes between calls (113–272 Hz median pitch).

Not measured: a Mac with less memory, and every judgement of quality.

## Decisions

- **In the server's process**, like Supertonic. A dry run of `pip install mlx-audio==0.6.8` in
  Keep's venv moves one pin only (`sounddevice` 0.4.6 → 0.5.6); `mlx`, `numpy`, `transformers`
  and `mlx-whisper` stay. A worker process in a venv of its own is the fallback if step 1 shows
  MLX cannot run two models from two threads (Whisper for the local microphones and this) or
  that a memory failure takes the server down — it is a good deal more code (a second venv, a
  protocol, a supervisor), so it is not built on suspicion.
- **A TTS engine, where the others are**: `src/interfaces/voice/tts/FishLocalVoiceSynthesizer.py`,
  a `VoiceSynthesizer` and an `Integration` named `Fish Local TTS`, config section
  `tts.fish_local`. Off by default; `enable_integration("Fish Local TTS")` turns it on.
  Supertonic stays the default voice of local AI.
- **`mac_only`**, Apple Silicon, and a memory gate: refused with the reason under 48 GB of RAM
  for 4-bit (26 GB peak beside the LLM and the rest of the server). `tts.fish_local.quant`
  takes `3bit`, `4bit` (default) or `8bit`; the gate follows it (32 / 48 / 64 GB) — the
  3-bit and 8-bit thresholds come from the mlx-audio docs, not from a run of ours.
- **Every fact about the model in one file** (`FishLocalModel.py`): the repo per quant, the
  pinned revision of each, the sizes, the gates. The revision is pinned because the repo is a
  community conversion Fish neither released nor endorsed — an install loads the bytes that
  were tested, never whatever the repo holds that day.
- **The licence is the user's**: Drama 3 is under the Fish Audio Community License (commercial
  use free under USD 500,000 of annual revenue; offering it as a hosted API needs a licence of
  its own). Keep downloads the weights to the user's Mac and redistributes nothing. The config
  comment and the integration's description say so in a sentence.

## The pieces

### Download (`FishLocalModel.ensure`)

- Started on a background thread when the engine is first constructed while enabled, and by
  `restart_integration`. `huggingface_hub.snapshot_download(repo, revision=…, local_dir=…)` into
  `~/.cache/keep/fish_local/<repo>` — outside the checkout, so a reinstall of Keep does not
  fetch 22 GB again, and a folder of its own, so its size is the progress (the hub's shared
  cache keeps its blobs elsewhere). An interrupted download resumes file by file: the files that
  arrived whole are kept, the ones cut off start again, and their part files are deleted first
  (the hub leaves them behind — 6.8 GB after a kill at 39 % — and they would count as progress). A complete one leaves
  `.keep-revision` in the folder: with it a start asks the hub nothing, so the voice comes up
  without a network.
- Before it starts: free disk at least what is left to fetch plus 8 GB, else not configured
  with the numbers — as on a Mac with too little memory, set by the engine's constructor,
  which then starts nothing; `restart_integration` asks again.
- Progress is that folder's size against the repo's total, logged every 5 % and carried by
  `check_health` as a `HealthWarning` ("downloading, 43 %") — WARNING in the monitor, never
  counted towards ERROR. A failed download is retried at the next health check, with the
  reason logged.
- Then load and one warm-up sentence with the reference voice, so the voice is cached before
  the first reply. A restart of the server loads from disk on the same thread.

### Until it is ready

`VoiceSynthesizer.ready()` (new, `True` in the base): `VoiceFactory.create_voice` skips an
engine that answers `False` and logs why, exactly as it skips a missing key. So during the
download, and after a failed load, replies go on through the next engine — Supertonic on a
local install — and switch over by themselves.

`ENGINES` gets `FishLocalVoiceSynthesizer` in front of `SupertonicVoiceSynthesizer`: with both
on, the better voice speaks.

### Synthesis

- The model takes a whole text; the LLM streams one. The clause and sentence batching
  Supertonic has (`_pop_speakable`: the first unit at the earliest clause mark, the rest
  sentence by sentence) moves to a shared `ClauseBatcher` both engines use, with the
  44.1 → 24 kHz int16 conversion beside it.
- Per unit: `model.generate(text, ref_audio, ref_text, stream=True, first_chunk_frames=8)`,
  each piece converted and queued as it arrives.
- One model per process and one generation at a time: `FishLocalModel` owns one thread that
  loads the model and then speaks the queued texts in turn, so a second dialog waits its turn
  and no lock is held across a generation. `close()` sets `is_closed`, checked between
  pieces — a superseded reply stops within one piece (about 0.3 s of audio).
- `SUPPORTS_EMOTION_TAGS = True`. An unmatched `<whisper>` cut by the batcher is closed or
  dropped before the call (the model validates its text).
- `cache_variant()` is the quant, so the phrase cache keeps one recording per build.

### The voice

Without a reference the model picks another voice at every call, so a reference is always
passed, and nothing is shipped for it:

- default: the first load reads a fixed neutral text (`VOICE_TEXT`, about 12 s) without a
  reference, saves the clip and its transcript as `cache/fish_local/voice.wav` / `voice.txt`,
  and speaks in that voice from then on — one voice per install. Deleting the clip and
  `restart_integration` makes another;
- `tts.fish_local.voice` / `voice_text`: the user's own clip and what is said in it.

`voice_id()` carries the clip's time of writing, so the phrase cache never answers in a
voice that was replaced.

### Config

```yaml
tts:
  fish_local:
    quant: 4bit      # 3bit | 4bit | 8bit — about 18 / 22 / 42 GB to download
    # voice: ~/voices/mine.wav        # a reference clip to speak in the voice of
    # voice_text: "What the clip says."
```

`config.example.yaml`: this section, the engine list in the comment over
`enabled_integrations`, and the engines named at `tts.emotions`.

### Requirements

`requirements-voice.txt`: `mlx-audio==0.6.8 ; sys_platform == "darwin"`, and `sounddevice`
to 0.5.6. Every `mlx_audio` import is inside the load, so Linux and a Mac without the
integration import nothing of it.

### Tests (`tests/unit/interfaces/voice/tts/test_fish_local_voice_synthesizer.py`)

Over a fake model whose `generate` yields arrays: the factory's fall-through while not ready and
the switch once it is; the download's states (too little disk, progress as a warning, a failure
retried, a resume); the gate per quant; a tag cut across units; a kill between pieces; the
turns. The unit conftest refuses `snapshot_download` and the model loader for real, and
`test_no_server_relaunch.py` holds the refusal — no test fetches or loads 22 GB.

## Steps

1. **Spike, before any integration code** — go or no-go. In a branch, with `mlx-audio` in
   Keep's real venv on the M5 Ultra:
   - speed and first audio while the local LLM streams a reply on the same GPU (under 1.2x
     real time is a no-go for 4-bit: playback would stall);
   - Whisper transcribing on a second thread during a generation (decides in-process or worker);
   - the server's microphones and speakers on `sounddevice` 0.5.6;
   - by ear: the tags of `res/voice_emotions.md`, Swedish, 4-bit against 8-bit, a voice with
     and without a reference, sentence joins.
2. `ClauseBatcher` out of Supertonic, `VoiceSynthesizer.ready()`, the factory's skip — no
   change in behaviour, tests green.
3. `FishLocalModel` (download, gates, load) with its tests.
4. `FishLocalVoiceSynthesizer`, config, requirements, the conftest refusals.
5. A day as the voice of the M5 Ultra install, then the `setup` toolkit's one line about it.

## Open

- **A chosen default voice**: every install rolls its own, so no two sound alike and none
  was picked by ear. A clip recorded for Keep and shipped in `res/` would settle that; a clip
  made by Drama 3 itself raises whether a model output is ours to ship under its licence.
- **Swedish**: `PREFERRED` keeps Gemini for `sv`; a local install has no Gemini, so Swedish
  would go to this engine. Whether that is better than Supertonic's Swedish is step 1's call.
- **Memory when idle**: the model stays loaded (about 24 GB). Unloading after a quiet hour
  costs a 1.7 s load on the next reply; left out until somebody misses the memory.
- **Smaller Macs**: S2 Pro has an MLX build of 6.7 GB that would suit a 32 GB machine, with no
  published Mac numbers. A second quant list in `FishLocalModel` if it is wanted.
