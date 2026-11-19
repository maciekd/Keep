"""Tunable constants for local mic outposts.

Ported verbatim from Tools/LocalSTTTests/whisper_turbo_streaming_test.py — the
defaults here mirror that script's argparse defaults and module constants so the
local outposts behave identically to the hand-run test. Change values here, not
in the call sites.
"""

from common.ConfigManager import config

# ── Audio capture ─────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
FRAME = 1600                       # samples per audio frame = 100 ms at 16 kHz
FRAME_DURATION = FRAME / SAMPLE_RATE
AUDIO_QUEUE_TIMEOUT = 0.5          # seconds to block waiting on the capture queue

# ── Capture-drop detection ────────────────────────────────────────────────────
# CoreAudio silently drops input samples when the audio callback is starved of the
# GIL/CPU (heavy Whisper decodes + per-frame VAD across mics) — the gap the user
# hears in a saved recording. PortAudio's own overflow status flag is UNRELIABLE on
# macOS (verified: stayed silent through a deliberate 1.3 s drop), so we detect drops
# from the hardware ADC capture clock instead: consecutive callbacks should advance
# time_info.inputBufferAdcTime by exactly FRAME_DURATION, so any excess is audio the
# device captured but never delivered. The excess equals the size of the gap.
CAPTURE_DROP_MIN_GAP = 0.1         # ADC-clock hole (s) at/above this counts as a dropped-audio gap
CAPTURE_DROP_WARMUP_FRAMES = 2     # ignore the first N callbacks after (re)open (stream-start settling)
CAPTURE_DROP_STARTUP_GRACE = 60.0  # ignore drops in the first N s after a mic starts: model-load/init CPU+GIL contention starves the callback at boot, so these are false positives, not real recording gaps
CAPTURE_DROP_SUMMARY_INTERVAL = 60.0  # min seconds between cumulative per-mic drop summaries
RESOURCE_MONITOR_INTERVAL = 60.0   # seconds between 'resmon' process/audio resource snapshots (leak hunt)

# ── Privacy ───────────────────────────────────────────────────────────────────
# The mics transcribe everything said in the room, but only a wake-word sequence is the
# assistant's business: an utterance that opens with the wake word (whether or not the
# fire survives the hallucination gate) and the follow-ups inside the listen window it
# opened. False keeps the words of every other transcript out of the logs — a log line
# carries its length instead. True logs all of it in clear, for hunting a missed wake word.
LOG_OVERHEARD_TRANSCRIPTS = False


def transcript_for_log(text, addressed=False):
    """`text` quoted for a log line, or only its length when it is overheard speech and
    LOG_OVERHEARD_TRANSCRIPTS is off. `addressed` says the transcript belongs to a
    wake-word sequence; a caller that cannot know (the Whisper worker, which decodes
    before the gate) leaves it False."""
    if addressed or LOG_OVERHEARD_TRANSCRIPTS or not text:
        return f"'{text or ''}'"
    return f"<{len(text)} chars, not logged>"

# ── Wake word ─────────────────────────────────────────────────────────────────
# Single source of truth: config.yaml `wake_word` (also drives the prompt identity
# and /outpost_settings). WAKE_WORDS is matched case-insensitively against transcripts;
# `wake_word_aliases` lists extra spellings Whisper commonly mishears it as (e.g.
# "michel" for "Michelle") so near-misses still trigger.
WAKE_WORD = config.get('wake_word', 'Michelle')
WAKE_WORDS = tuple(w.lower() for w in (WAKE_WORD, *config.get('wake_word_aliases', [])))

# Words Scribe renders a cleanly spoken wake word as — near-homophones that are NOT the wake
# word and never trigger on their own, but on the Scribe VERIFY of a Whisper wake-word fire
# they mean "Scribe heard the sound and misspelt it", not "Scribe heard no wake word" (see
# gate_scribe_verify / near_wake_word_token). Evidence-backed, one entry per observed
# rendering: "Mission complete" for "Michelle good night" (WakeWordTest ×2), "Missions to
# kitchen temperature" for "Michelle, what's the kitchen temperature" (dev log 2026-08-03),
# "We shall volume down" for "Michelle volume down" (WakeWordTest 2026-08-27). An entry may
# span several words; it must be something ordinary speech rarely OPENS with — check the
# suite negatives and prod Scribe vetoes before adding one (as of 2026-08-27, 341 suite
# negatives + 212 prod vetoes: none opens with any entry). config.yaml
# `wake_word_near_misses` overrides.
_DEFAULT_NEAR_MISSES = ['mission', 'missions', 'we shall']
_UNSET = object()   # config.get raises on a missing key unless given a non-None default
_configured_near_misses = config.get('wake_word_near_misses', _UNSET)
WAKE_WORD_NEAR_MISSES = tuple(
    ' '.join(w.lower().split()) for w in (_DEFAULT_NEAR_MISSES if _configured_near_misses is _UNSET
                        else _configured_near_misses))
if _configured_near_misses is _UNSET and WAKE_WORD.lower() != 'michelle':
    # The defaults are Scribe renderings of "Michelle" specifically. Under another wake word
    # they are just ordinary words, and any sentence opening with one would corroborate a
    # Whisper fire instead of vetoing it.
    from common.logging import note
    note(f"wake_word is '{WAKE_WORD}' but wake_word_near_misses is not set — the default "
         f"{_DEFAULT_NEAR_MISSES} is tuned for 'Michelle'. Set wake_word_near_misses in "
         f"config.yaml (an empty list disables the near-miss rescue).")

# A wake word only counts when it starts within the first N characters of the
# transcript — it has to *open* the utterance, not merely appear somewhere in it.
# Mirrors the Android outpost (MainActivity.kt: `indexOf(wakeWord) in 0..9`), which
# anchors for the same reason: otherwise an overheard sentence that happens to name
# the assistant mid-flow (e.g. "Ska du berätta för Michelle vad du såg?") triggers a
# false command. Widened 10→15 to admit a short hallucinated lead-in Whisper sometimes
# prepends (e.g. "Thank you, Michelle …"). A fire whose prefix contains spoken words
# (not just punctuation) must ALSO clear WAKE_WORD_PREFIX_CONFIDENCE_MIN_PCT below.
# A wake word past the anchor still counts when it OPENS ITS OWN SENTENCE — the text
# before it ends with sentence-final punctuation ("What are their names? Michelle,
# pause the music.", Aug 29 2026 prod miss): the speaker finished addressing someone
# else, then addressed us. Mid-sentence mentions stay overheard. Such a fire always has
# a spoken prefix, so the prefix confidence floor below guards it too.
WAKE_WORD_ANCHOR_CHARS = 15

# A wake-word fire with spoken words BEFORE the wake word ("I'm Michelle…",
# "Okay, Michelle…") is almost always a hallucinated or overheard non-address: in
# 2 months of prod fires (Jun 9 – Aug 5 2026), 49 of 600 had a spoken prefix; 48
# were junk (priming echoes like "I'm Michelle", overheard chatter — including the
# 2026-08-04 "Okay, Michelle, I've run out of stuff." and 2026-08-05 "I love you,
# Michelle." false activations) and topped out at 53.2% confidence, while the single
# genuine one ("Who's here? Michelle, open the pool." — the pool actually opened)
# scored 56.6%. Genuine addresses otherwise decode with the wake word flush at the
# front (209/210 on WakeWordTest, 551/600 prod fires). So a prefixed fire must clear
# this floor; 55 sits between the observed junk ceiling and the one genuine rescue.
# Leading punctuation/quotes don't count as a prefix — only alphanumeric speech does.
WAKE_WORD_PREFIX_CONFIDENCE_MIN_PCT = 55.0

# ── Whisper (MLX) ─────────────────────────────────────────────────────────────
# large-v2 beat large-v3-turbo and large-v3 on English command accuracy in the STT
# harness (Tools/STTTest): ~+6pp exact-match over turbo on its own, and it leverages the
# command-priming prompt far more (primed-phrase pass 36%→61% vs turbo 33%→43%). Costs
# ~0.4x RTF (vs turbo's 0.29x) — still comfortably under real-time. Full fp16, not quantized.
DEFAULT_MODEL_REPO = "mlx-community/whisper-large-v2-mlx"

# Decode settings tuned for low-latency, per-utterance independence. The latter
# (condition_on_previous_text=False) also makes every transcribe() call fully
# self-contained, which is what lets one shared model serve many mics safely.
DECODE_KWARGS = dict(temperature=0.0, condition_on_previous_text=False)

# Example-style priming: reinforces the wake word, the command cadence, and the
# expected domain vocabulary so Whisper spells/parses them the way Floxy expects.
#
# ONE phrase per configured language. This matters beyond spelling: Whisper continues
# in the language of its prompt, so a single-language (English) prompt makes it
# *translate* other languages instead of transcribing them — verified on real audio,
# Swedish "vad är klockan?" came back as "what is the clock?". Priming each configured
# language gives every utterance same-language context, which keeps Swedish Swedish
# while still spelling the wake word right (dropping the prompt fixes the translation
# but then the wake word is mis-heard). A pure-English config primes English only.
#
# Kept as separate phrases (not one blob) because Whisper periodically hallucinates a
# priming phrase back verbatim during silence — we match transcripts against these and
# drop any exact echo (see is_priming_echo in LocalOutpost), so each language's phrase
# must also be its own entry here.
#
# The English phrase is an example-command prompt built from floxy.db's real
# VoiceToVoice command vocabulary. Measured in the STT harness (Tools/STTTest), it
# roughly halves English error vs. a plain one-sentence prompt: exact-match pass jumped
# on every command cluster (e.g. "volume down", "Model X" stopped decoding as "Rolex")
# with NO regression on non-primed utterances. ~80 tokens, well under Whisper's 224-token
# prompt cap. Caveat: is_priming_echo only drops a *verbatim* echo of the whole phrase,
# so a single command hallucinated out of it during silence is not echo-filtered — the
# risk metrics (no_speech/logprob/compression) remain the backstop for that.
_PRIMING_BY_LANG = {
    "en": (f"You are {WAKE_WORD}, our AI home assistant. Examples of commands we say to you: "
           f"turn off the wardrobe lights, "
           f"dim the office table lamp, "
           f"volume up, "
           f"pause the music, "
           f"next song, "
           f"defrost the Model X, "
           f"add milk to the shopping list, "
           f"show me the camera feeds. "
           f"Next sentence will be a query to you: "),
    "sv": f"Du är min personliga AI-hemassistent, {WAKE_WORD}. Nästa mening är en förfrågan från mig: ",
}


def _configured_priming_phrases(languages=None):
    """One priming phrase per language subtag, in order and de-duplicated. Languages
    without a template fall back to the English phrasing (the wake word still primes;
    only the cadence sentence is English). `languages` is a list of BCP-47 locales;
    None means the configured set."""
    from interfaces.voice.tts.VoiceSynthesizer import VoiceSynthesizer
    if languages is None:
        languages = VoiceSynthesizer.configured_languages()
    phrases = []
    for tag in languages:
        phrase = _PRIMING_BY_LANG.get(tag.split("-")[0].lower(), _PRIMING_BY_LANG["en"])
        if phrase not in phrases:
            phrases.append(phrase)
    return tuple(phrases)


def build_initial_prompt(languages=None):
    """The Whisper initial prompt for a language set (every language's priming phrase
    joined), built exactly as production builds INITIAL_PROMPT. Production uses the
    configured set (the INITIAL_PROMPT constant below); the STT harness calls this with
    the run's languages so it primes precisely as a deployment of just those languages
    would — e.g. an `--lang en` run gets the same prompt a `language: en-US` server has."""
    return " ".join(_configured_priming_phrases(languages))


INITIAL_PROMPT_PHRASES = _configured_priming_phrases()
INITIAL_PROMPT = " ".join(INITIAL_PROMPT_PHRASES)

# ── Endpointing ───────────────────────────────────────────────────────────────
SILENCE = 0.4                      # trailing silence that ends an utterance (s)  make sure this is lower than whisper processing time and we should not introduce latency
# The longer somebody has been speaking, the longer a pause has to be before it counts as
# the end (Endpointer.patience): SILENCE up to PATIENCE_FROM seconds of utterance, growing
# in a straight line to SILENCE_LONG at PATIENCE_FULL. Placed from a month of production
# segments (pre-roll included, 535 wake-word commands): 93% end under 4.5 s and keep the
# 0.4 s endpoint untouched; the ones above are 9+ words, and the pauses inside a long
# sentence measured 1.1–1.4 s — each of which the fixed 0.4 s used to end the utterance on,
# so the reply to its first half started while the speaker was still going.
SILENCE_LONG = 2.0                 # trailing silence that ends a long utterance (s)
PATIENCE_FROM = 4.5                # utterance seconds up to which SILENCE applies
PATIENCE_FULL = 6.0                # utterance seconds from which SILENCE_LONG applies
SPEC_SILENCE = 0.15                # trailing quiet before a speculative decode (s); must be < SILENCE.
                                   # Raised from 0.05 so a speculative only fires on a pause long enough
                                   # (>= ~0.2 s quiet) to plausibly be the utterance end — not on every
                                   # 0.1 s inter-word micro-gap. Firing on micro-gaps produced a storm of
                                   # uninterruptible decodes that held the GIL while the mic was still
                                   # capturing, starving the audio callback and dropping recorded audio.
                                   # The end-of-utterance pause still crosses this on its way to SILENCE,
                                   # so the speculative→final overlap is preserved.
                                   # Measured on the RMS ENERGY gate, not silero: silero's offset scores
                                   # decay slowly (~0.3–0.5 s of 'speech' after the acoustic end), while
                                   # energy collapses within one frame — so the energy gate starts the
                                   # speculative during silero's hangover instead of after it, shaving
                                   # that hangover off the user-perceived STT time. Silero still owns the
                                   # real endpoint (SILENCE) and speech onset. See LocalOutpost._endpoint_loop.
SPEC_STALE_SPEECH_GRACE = 0.6      # An energy-triggered speculative fires while silero still reports
                                   # speech. Up to this many seconds of post-fire silero 'speech' is
                                   # treated as normal offset decay; beyond it, silero was likely hearing
                                   # genuinely soft speech (below the energy gate) that the speculative's
                                   # audio lacks — the spec is cancelled at VAD stop so the final decodes
                                   # the full utterance rather than promoting a truncated transcript.
MIN_SPEECH = 0.3                   # ignore speech segments shorter than this (s)
MAX_SEGMENT = 20.0                 # force-endpoint after this many seconds (s)
PRE_ROLL = 1.0                     # audio retained before speech onset (s).
                                   # Raised 0.3→1.0 after the STT harness (Tools/STTTest)
                                   # showed the wake word being dropped on a class of clips:
                                   # Silero marks speech-start slightly AFTER "Michelle"
                                   # begins (its soft "M-" onset is below the VAD threshold),
                                   # and a 0.3s pre-roll didn't reach back far enough, so the
                                   # trim sliced into the wake word and Whisper omitted it
                                   # ("where is my Model X" instead of "Michelle, where is…").
                                   # Reaching ~1.0s back recovers every such clip with zero
                                   # regressions on the test set (English exact-match 82→83.5%,
                                   # matching the untrimmed-audio ceiling). Pre-roll is buffered
                                   # historical audio, so this adds no user-facing latency, only
                                   # a little more audio per decode. WATCH (deployed for
                                   # observation): a longer lead-in can also pull in unrelated
                                   # prior audio on an always-on mic (tail of a previous
                                   # sentence, TV) — dial back if that shows up in practice. A
                                   # future onset-hysteresis (back-extend to a lower energy
                                   # threshold, bounded by the real pre-speech silence) would
                                   # catch the onset without a fixed long grab.
SPECULATIVE = True                 # overlap a speculative decode with the silence wait

# ── Utterance assembly (re-joining silence-split fragments) ───────────────────
# The outpost endpoints on trailing silence, so one spoken command interrupted by a
# pause longer than SILENCE arrives as several disjoint Whisper segments. The server's
# transcription voting expects *cumulative growing* phrases (as the phones' streaming
# STT sends) — disjoint fragments confuse it (it answers half the sentence). So the
# outpost re-joins consecutive fragments into one running phrase and re-sends it
# cumulatively, as long as the next fragment begins within CONCAT_MAX_GAP of the
# previous one ending. A longer gap is treated as a new, separate command. The total
# tolerated mid-sentence pause ≈ SILENCE + CONCAT_MAX_GAP. See LocalOutpost._on_transcript.
# A long utterance is rarely split at all: its endpoint waits longer (SILENCE_LONG above).
CONCAT_MAX_GAP = 2.5               # max inter-fragment silence to treat segments as one utterance (s)

# ── VAD ───────────────────────────────────────────────────────────────────────
VAD_BACKEND = "silero"             # "silero" (neural) or "rms" (energy gate)
VAD_THRESHOLD = 0.5                # speech-probability threshold for silero

# RMS auto-calibration. Used for the main VAD when VAD_BACKEND == "rms", and ALWAYS
# for the shadow energy gate that times the speculative decode (see SPEC_SILENCE).
# The startup calibration only SEEDS the floor/threshold; from then on RmsVAD tracks
# the room adaptively with a fast-attack raise rule and a quiet-window settle rule
# (see RmsVAD), so a kitchen whose background noise shifts — including post-reply
# speaker hiss — re-calibrates itself within seconds instead of pegging 'loud'.
RMS_CALIBRATION_SECONDS = 1.0
RMS_NOISE_FLOOR_FALLBACK = 0.005
RMS_THRESHOLD_MULTIPLIER = 1.5     # speech threshold = floor × this. Was 3.0 with the old
                                   # minimum-statistics floor (which sat at ambient's very bottom
                                   # and needed the headroom); the percentile floor anchors near
                                   # ambient's TOP, so 50% over it already means a real sound.
RMS_THRESHOLD_FLOOR = 0.0005       # degenerate-silence guard, NOT an ambience-relative tuning
                                   # knob: it only exists so a near-zero floor (dead channel, mic
                                   # gain accidentally at zero) can't collapse the threshold to
                                   # where electrical noise reads as speech. Sized below any real
                                   # room (measured floors 0.0014-0.0052 across a day, 2026-08-12)
                                   # so floor x MULTIPLIER is always the operative threshold in
                                   # normal operation. Was 0.006 (arbitrary) — that sat at ~half
                                   # of PEAK speech RMS on the quiet USB mics (speech peaks
                                   # 0.008-0.016, soft word tails 0.003-0.007), permanently
                                   # overrode the adaptive floor, and clipped fading endings
                                   # ("...lights on") ~0.5s early — inflating no_speech on
                                   # speculative clips past the risk gate.
RMS_FLOOR_RAISE_SECONDS = 1.0      # raise window: ambience rose when (almost) all of it sits above
                                   # the floor — lift the floor to its low percentile (fast attack)
RMS_FLOOR_SETTLE_SECONDS = 5.0     # settle window: mostly sub-threshold → re-anchor the floor near
                                   # its top (hug the quiet room from above)
RMS_FLOOR_PERCENTILE = 0.9         # outlier robustness for BOTH rules. Settle arms on and anchors
                                   # to the window's 90th percentile (a few clicks can neither hold
                                   # the floor up nor disarm the settle); raise triggers on the
                                   # complementary 10th percentile (one spuriously low frame can't
                                   # block a legitimate raise).

# ── GPU keep-warm ─────────────────────────────────────────────────────────────
# Metal drops the GPU into a low-power state after decode inactivity; the next
# Whisper decode then pays a ~2.2–2.9x cold penalty (measured 2026-08-11/12 with
# gpu-idle probe: warm floor ~0.6s → 1.4–1.8s cold, one decode restores the floor).
# The decay threshold VARIES with machine power state — ≥120s idle stayed warm on
# an attended machine, but 60–120s went cold on an unattended one — so the cadence
# stays under the worst observed bound. A tiny silence decode after this much
# decode-idle keeps the first real query of a quiet period at the warm floor
# (~1% GPU duty). Only submitted when no mic is mid-utterance, so a real decode
# can never queue behind a keep-warm that just started (see LocalOutpostManager).
KEEPWARM_IDLE_SECONDS = 45.0

# ── Confidence / hallucination risk thresholds (Whisper standard) ─────────────
# A transcript past ANY of these is considered risky and is NOT sent to the server.
# Confidence here is exp(mean per-token logprob) expressed as a percent — i.e. the
# transcript's average per-token probability.

# A soft-spoken speaker decodes around logprob -0.8 (~45%), so the floor sits just below that.
RISK_CONFIDENCE_MIN_PCT = 40.0            # confidence (%) below this is risky (≈ logprob -0.9)
WAKE_WORD_RISK_CONFIDENCE_MIN_PCT = 32.0  # looser confidence floor when a wake word opens the
                                   # clip: a short "Michelle, …" utterance gives Whisper little
                                   # context, so a cleanly-heard wake word averages low (32–64% on
                                   # the WakeWordTest set). Standard floor is 40; this stays below it
                                   # because Scribe verifies every Whisper wake-word fire (fusion).
                                   # Set to 32 (not lower): genuine fires bottom out at 32% conf, while
                                   # the "Michelle is a very long box." hallucinated-prefix false fire
                                   # sits at 31% — so 32 is the highest floor that drops that FP at zero
                                   # genuine loss. Above 32 it's pure recall bleed (no further FP dies
                                   # until 45%, which would cost ~17 genuine wake words). The
                                   # no_speech/compression gates still guard silence hallucinations.
# A wake-word fire that is BOTH low-confidence AND breathy (high no_speech) is the
# signature of a CORRELATED cross-engine hallucination: Whisper (forced to English) and
# Scribe (pinned to English) BOTH invent "Michelle <command>" out of a foreign/noise
# utterance, so the fusion gate (which only catches a single engine hallucinating) can't
# veto it. Neither single risk floor separates it from a genuine short command — a real
# "Michelle, turn on the TV" reaches no_speech 0.19, and a genuinely unsure command
# bottoms out at conf 32% — but the two NEVER co-occur in a genuine fire: a breathy
# genuine clip is still confident (the WakeWordTest breathy fires all sit ≥51%), and an
# unsure genuine clip still sounds like speech (low-conf fires top out at no_speech 0.16).
# Only the hallucination lands in both-bad. So a wake-word clip past BOTH floors at once
# is dropped; a clip tripping only one rides the normal floors above. Measured on
# WakeWordTest this kills the surviving Swedish FP ("vi ska inte sitta här då" →
# "Michelle, take it all out." conf 37% / no_speech 0.196) at ZERO genuine loss, the
# thresholds centered in both gaps (nearest genuine: +6pp conf, +0.01 no_speech).
# Widened 2026-08-25 from (0.17, 45) to (0.10, 42) to also catch the Aug 20–23 prod FPs
# "käkat just nu" / "köket just nu" (conf 38.2/40.8, no_speech 0.132/0.123 — "Michelle,
# could you stop / get this" in BOTH engines). Cost on WakeWordTest: ONE genuine fire,
# "Michelle good night" (conf 37.6 / ns 0.159, Whisper hears "Michelle, come on."). The
# margin is THIN: the nearest surviving genuine fire ("Michelle volume down 5") sits at
# conf 42.2 / ns 0.134 — 1.4pp above the ceiling — so re-sweep (experiments/CLAUDE.md,
# /tmp/wwt_cache_en.pkl) before touching either number or changing the Whisper model/prompt.
WAKE_WORD_HALLUCINATION_NO_SPEECH_MIN = 0.10   # no_speech must exceed this AND…
WAKE_WORD_HALLUCINATION_CONFIDENCE_MAX_PCT = 42.0  # …conf must be below this for the combined veto

# NOTE (2026-08-11): a "correlated low confidence" verify veto (whisper <50% AND raw scribe
# <68%, targeting the 2026-08-05 "vad är det som låter" FP at 47%/66%) was tried and REMOVED:
# Scribe's confidence jitters several pp between decodes of the same clip, so the one-clip-deep
# margins didn't survive a re-run — a genuine fire drifted into the veto corner (Scribe 72%→59%)
# while the FP drifted out of it. Don't re-add a cross-engine confidence-corner veto unless the
# corner is measured across MULTIPLE runs with margins that beat the observed jitter.

RISK_NO_SPEECH_MAX = 0.20          # no_speech_prob above this is risky. Tightened 0.6→0.20:
                                   # genuine wake-word fires top out at no_speech 0.19 on the
                                   # WakeWordTest set (highest: "Michelle, turn off the TV." 0.193),
                                   # so 0.6 was far looser than real speech ever needs. 0.20 drops
                                   # two hallucinated-prefix false activations ("Michelle, we are so
                                   # cool…" 0.36, "Michelle, you're almost at the point" 0.22) at zero
                                   # genuine loss on the set. DELIBERATELY AGGRESSIVE — the margin to
                                   # the top genuine fires (0.193/0.186) is ~0.01, chosen because
                                   # false activations are high-cost. Verified English-only; a noisier
                                   # room (TV behind a "turn off the TV") could nudge a real command
                                   # over, trading a rare miss for fewer false fires. Loosen if genuine
                                   # wake words start getting risk-dropped in the logs.
RISK_COMPRESSION_MAX = 1.3         # compression_ratio above this is risky. Genuine
                                   # wake-word commands top out around 0.9 (longer, varied
                                   # speech scores even lower); short repeat hallucinations
                                   # like "I'm Michelle. I'm Michelle. I'm Michelle." land at
                                   # ~2.2. The old 2.4 floor let those slip through, so 1.3
                                   # sits in the empty gap — well above real speech, below the
                                   # repeats — without an extra n-gram dedup check.
RISK_TEMPERATURE_MAX = 0.0         # temperature above this is risky

# ── STT engine names ──────────────────────────────────────────────────────────
# Tag each vote with the engine that produced it so the server (and Home tab) can
# attribute a phrase to Whisper vs Scribe. Carried as the `engine` /stream param.
ENGINE_WHISPER = "whisper"
ENGINE_SCRIBE = "scribe"

# ── Scribe (ElevenLabs Scribe v2 parallel-vote STT) ───────────────────────────
# Every utterance is transcribed by Scribe v2 (cloud) alongside the local Whisper
# decode; its transcript is posted to the server as an extra vote. Scribe scores
# higher than Whisper on its own (89.7% vs 83.3% exact-match in Tools/STTTest), so
# when the two disagree its candidate gives the server's voting a stronger option.
# No outpost-side selection — the server's existing confidence-weighted voting picks.
SCRIBE_ENABLED = True
SCRIBE_MODEL = "scribe_v2"
SCRIBE_TIMEOUT = 5                  # cloud round-trip budget (s); mean observed ~0.76s

# Gate Scribe behind Whisper's wake-word detection instead of voting on every utterance.
# True (new): Scribe runs ONLY after Whisper accepts a command — once, on that final audio
#   — and on a wake-word send it doubles as a verifier. If Scribe doesn't hear the wake word
#   the request is a Whisper false positive and gets killed (see LocalOutpost._on_scribe_transcript
#   and the server's /false_positive route). This stops paying Scribe for all-day background
#   speech that never contained the wake word.
# False (old): Scribe fires on every VAD-endpointed segment (speculative + final) as a pure
#   parallel vote, regardless of wake word. Kept as a rollback path.
SCRIBE_VERIFY_WAKE_WORD = True

# Skip Scribe entirely when Whisper is already confident. Two independent justifications:
#
# 1. Wake-word veto (empirical). Two months of prod logs (2026-06-15 → 2026-08-11, 793
#    wake-word fires, 75 Scribe vetoes): the veto rate is a steep function of Whisper
#    confidence — 40% of fires below 40% conf were vetoed, 11% in 40–50%, and NOT ONE fire
#    at ≥52% was ever vetoed (highest veto: 51.4%; genuine fires run to 72.8%). The two
#    vetoes above 50% were themselves FALSE vetoes — Scribe heard the wake word behind a
#    spoken lead-in ("…let's go out. Michelle, what's the pool temperature?") and the
#    anchor rule refused it — so up there a veto is more likely to kill a genuine command
#    than a hallucination. WakeWordTest (529 clips) agrees: the skip admits zero new false
#    activations — the only fusion-mode FPs are correlated cross-engine hallucinations
#    that Scribe fires on too, so verification never blocked them anyway.
#
# 2. Transcript selection (mathematical). The server's vote picks Whisper iff
#    whisper_conf × SCRIBE_CONF_CALIBRATION ≥ scribe_conf, and Scribe's conf never exceeds
#    1.0 — so at whisper_conf ≥ 1/1.65 ≈ 60.6% Whisper wins the vote NO MATTER what Scribe
#    says, making the skip provably identical to running the vote. 61 sits just above that
#    bound. (A 60 floor was measurably worse: on the STTTest fusion cache it lost one
#    borderline clip — "room" vs "pool temperature" at w_conf 60.006%, where Scribe's
#    0.9999 still won the vote.) If SCRIBE_CONF_CALIBRATION changes, re-derive: the
#    lossless bound is 100/calibration.
#
# What the skip buys per high-confidence fire (~25% of prod wake-word fires sit ≥61%):
#   - no Scribe API call (cost), and none of the false-veto risk from (1);
#   - the latency win: the send carries scribe_skipped=1 so the server's verdict gate
#     (MAX_WAIT_FOR_SCRIBE_VERDICT) doesn't hold the reply for a verdict that is never
#     coming. Measured over the same two months, of the 206 replies that reached the gate:
#     57% waited 0s (verdict beat the reply), 28% held mean 0.23s / max 0.9s, and 15%
#     burned the FULL 1.5s cap (Scribe never answered — network hiccup/lag). The skip
#     erases that whole tail for confident fires.
#
# The wake-word margin in (1) is CALIBRATED TO THE CURRENT WHISPER MODEL (conf =
# exp(mean per-token logprob) shifts across versions) — re-run the veto-vs-confidence log
# analysis after any model change. The 100/calibration bound in (2) is model-independent.
SCRIBE_SKIP_CONFIDENCE_PCT = 61.0

# Scribe and Whisper both report confidence as exp(mean per-word/token logprob) on a
# 0..1 scale, but Scribe's numbers run systematically higher: measured offline,
# whisper_conf × ~1.728 ≈ scribe_conf. We send scribe_conf / this so the two land on the
# SAME scale and the server's `conf*80` scorer compares them fairly — it picks Scribe iff
# scribe_conf > whisper_conf × this. Whisper's own sends are untouched.
#
# Set just below the raw 1.728 ratio: the STTTest fusion sweep (Tools/STTTest, imports this
# very constant) finds the server scorer picks the right transcript ~2 clips/204 more often
# at 1.65 — it favours Scribe a touch more — and 1.65 sits centered in a flat [1.62,1.70]
# plateau. Side effect on the drop-floor below: only raw Scribe conf in ~[0.495,0.518] flips
# from dropped to kept, so re-run WakeWordTest after changing this.
SCRIBE_CONF_CALIBRATION = 1.65

# Disagreement veto on a Scribe VERIFY: both engines heard the wake word but their COMMAND
# texts share nothing AND neither engine is sure of its own. On every genuine WakeWordTest
# fire where the engines disagree, Whisper botched the command and Scribe nailed it at raw
# conf 0.92–1.0 ("hold your jaw" / "volume down") — or Whisper was sure (≥ 56%) and Scribe
# wrong ("where's my Model X" / "we must not move", Scribe 0.53). On the correlated Swedish
# hallucinations BOTH engines guess — different, mediocre English ("go out the door" /
# "you want something or what?": Whisper 43%, Scribe 0.69). Similarity is the Jaccard
# overlap of letter bigrams over the de-spaced lowercase command texts ("goodnight" ==
# "good night"); ≤ 0.20 means "nothing in common". Sweep (2026-08-27, _disagree.py): both
# remaining FPs caught at 0 of 208 genuine fires lost; nearest genuine fires sit at
# Whisper 56% (model x) and Scribe 0.92 (volume_down_5). Prod (421 in-band Whisper fires
# with a Scribe vote, Jun–Aug 2026): ~20 real false activations vetoed, ~2 uncertain
# ("what do you think?" / "Wood" at Scribe 0.77).
SCRIBE_DISAGREE_MAX_SIMILARITY = 0.20
SCRIBE_DISAGREE_MAX_SCRIBE_CONF = 0.80
SCRIBE_DISAGREE_MAX_WHISPER_CONF = 0.50

# Scribe RESCUE of a risk-dropped Whisper wake-word fire (gate_scribe_rescue). The risk
# gate's no_speech ceiling misfires on speech over background music: the Aug 29 2026
# "pause the music" incidents decode PERFECTLY in both engines (Whisper 57–58% conf,
# Scribe raw 1.0, identical text) but no_speech 0.23–0.51 kills the fire before Scribe is
# consulted. So a risky drop whose transcript still opens FLUSH with the wake word and a
# command, at Whisper conf ≥ the floor below, is held for Scribe: it fires only if Scribe
# positively confirms — verify stands, raw Scribe conf ≥ the floor, AND the commands agree
# (shared content word or bigram similarity ≥ the floor). Margins on the 2026-08-31 full
# run (556 deduped clips): the rescued fires sit at Whisper 57–58% / Scribe 1.0 with exact
# agreement; the nearest agreeing negatives are the "you okay" prod FP (Whisper 37% — 13pp
# under the floor, and Whisper-MLX is deterministic) and "vad är det som låter 2"
# (Whisper 47%, Scribe 0.71 — 24pp under the Scribe floor). No negative is close on both
# axes at once. Scribe's conf drifts run to run, so a failed rescue is the fail-safe
# direction: the fire stays dropped, exactly as before the rescue existed. The flush
# requirement is load-bearing — it excludes a 59%-conf priming-echo variant ("you are
# Michelle. Next sentence…") that a blanket relaxation would admit.
WAKE_WORD_RESCUE_WHISPER_CONFIDENCE_MIN_PCT = 50.0
WAKE_WORD_RESCUE_SCRIBE_CONFIDENCE_MIN = 0.95
WAKE_WORD_RESCUE_SIMILARITY_MIN = 0.5

# Drop a Scribe vote locally if its calibrated confidence is below this. The server
# both ignores sub-0.3 queries AND kills any running request matching that phrase as a
# false positive (floxy_ai_server.py), so a low-confidence Scribe echo of Whisper's
# running phrase must never reach it.
SCRIBE_MIN_CONFIDENCE = 0.3

# Keyterm priming — the single biggest accuracy lever for Scribe (exact-match pass
# 62.0% → 86.8% on the English test set): short distinctive command-domain terms Scribe
# otherwise mangles ('Model X' → 'MobX'), biased toward only when the audio supports it.
# ≤5 words / <50 chars each, none of the `<>{}[]\` characters. Shared with the harness
# (Tools/STTTest/backends.py ScribeBackend).
SCRIBE_KEYTERMS = [
    WAKE_WORD, "Model X", "Model 3", "Model Y",
    "kitchen table lamp", "wardrobe lights", "office table lamp",
    "shopping list", "camera feeds", "smooth jazz", "smooth beats",
    "volume up", "volume down", "next song", "pause the music",
    "pool temperature", "good night", "dim",
]

# Scribe reports the detected language as ISO-639-3 ('eng', 'swe'), but the outpost routes
# on the ISO-639-1 subtags in lang_by_subtag ('en', 'sv'). Map the languages Floxy may run.
ISO639_3_TO_1 = {
    "eng": "en", "swe": "sv", "nor": "no", "dan": "da", "fin": "fi",
    "deu": "de", "fra": "fr", "spa": "es", "ita": "it", "nld": "nl",
    "por": "pt", "pol": "pl", "rus": "ru",
}

# ── Mute failsafe ─────────────────────────────────────────────────────────────
# The server mutes outposts ("stop") while Floxy speaks and unmutes them
# ("listen"/"wakeword") when the reply ends. If that end-of-reply broadcast never
# arrives (server killed/errored mid-reply), a muted mic would stay muted forever:
# it drops every transcript and, since it can't produce the next query, nothing ever
# unmutes it. As a failsafe each mic auto-reverts to wake-word listening after this
# long muted. Must comfortably exceed the longest realistic reply so it never unmutes
# mid-reply (which would let the mic capture Floxy's own voice as a query).
MUTE_WATCHDOG_TIMEOUT = 90.0          # seconds muted with no unmute → auto-recover

# ── Capture-stream failsafe ───────────────────────────────────────────────────
# The PortAudio input stream can silently stop delivering frames (USB/CoreAudio
# hiccup, device sleep/reconfigure) without raising or flagging a status — the
# capture loop then blocks on an empty queue forever and the mic goes deaf with no
# error and no recovery until the whole process restarts. If no audio frame arrives
# for this long while the outpost should be capturing, treat the stream as dead and
# reopen it. Frames arrive every 100 ms, so this is ~100 consecutive missed frames.
STREAM_SILENCE_TIMEOUT = 10.0         # seconds with no audio frames → reopen the stream
# When the reopen itself keeps failing (device gone from the bus, coreaudiod wedge),
# retrying every second is pointless and actively harmful: on 2026-06-11 four hours of
# 1 Hz reopen attempts against a vanished device left coreaudiod spamming overload
# messages and degraded the whole USB/audio stack. Back off exponentially instead —
# the manager's recovery loop owns the real fix (power-cycle + PortAudio refresh).
STREAM_REOPEN_BACKOFF_MAX = 60.0      # cap on the exponential reopen-retry delay

# ── Mic auto-recovery (manager level) ─────────────────────────────────────────
# A background loop in LocalOutpostManager watches every mic and brings back one that
# has gone offline (a USB drop, or a wedged/deaf capture stream) without needing a full
# server restart. Each tick it checks two independent offline signals:
#   - frame staleness: a mic silent longer than RECOVERY_STALE_TIMEOUT;
#   - bus presence: a mic gone from the uhubctl USB scan for RECOVERY_OFFBUS_GRACE.
# The second exists because frame staleness alone cannot see a removed device when
# another input device is present: macOS reroutes the open stream to the default
# input, so frames keep flowing while the outpost mirrors the wrong mic (observed
# 2026-06-10 when the miniDSP was unplugged and its outpost cloned the USB CODEC).
# Recovery then escalates:
#   1. power-cycle the mic's USB port via uhubctl (the software equivalent of a replug);
#   2. once it re-enumerates on the bus, refresh PortAudio (its device list is cached for
#      the life of the process) and re-attach the outpost.
RECOVERY_TICK_INTERVAL = 5.0          # seconds between recovery-loop checks
RECOVERY_STALE_TIMEOUT = 20.0         # mic silent this long → offline (> STREAM_SILENCE_TIMEOUT so
                                      # the outpost's own stream-reopen gets first crack)
RECOVERY_OFFBUS_GRACE = 10.0          # mic absent from the USB scan this long → offline (debounces
                                      # the brief disappearance during a normal re-enumeration)
USB_POWER_CYCLE_COOLDOWN = 60.0       # min seconds between power-cycles of the same port
USB_POWER_CYCLE_OFF_SECONDS = 2       # uhubctl off-duration (-d) during a cycle
# A power-cycle that works brings the device back within seconds, so consecutive cycles
# with no re-enumeration in between mean cycling cannot fix this drop: the device's own
# firmware is hung and the hub only switches the data path, not VBUS (2026-06-11: ~230
# fruitless cycles over 4 h until a reboot). Give up after this many, alert loudly, and
# re-arm only when the device reappears on the bus.
USB_POWER_CYCLE_MAX_ATTEMPTS = 3      # consecutive no-effect cycles → stop cycling + alert
UHUBCTL_PATHS = ("/opt/homebrew/bin/uhubctl", "/usr/local/bin/uhubctl", "uhubctl")

# ── Follow-up listen timeouts (mirror the Android outpost) ────────────────────
# After entering listen mode (wake word detected, or server "listen" because the AI
# asked a question), revert to requiring the wake word if the user goes quiet.
LISTEN_FALSE_POSITIVE_TIMEOUT = 2.5   # no first utterance after entering listen → revert
                                      # (the old cross-mic hazard behind the earlier 1.5s value is
                                      #  gone: a bare wake word now arms only the mic that heard it,
                                      #  never every mic — the server no longer broadcasts "listen"
                                      #  on wake-word detection. So this can be generous for slow
                                      #  speakers who pause after the wake word without reopening the
                                      #  "ambient speech on another mic taken as the command" hole.)
LISTEN_NO_NEW_WORDS_TIMEOUT = 5.0     # no further utterance after the last one → revert
LISTEN_MAX_DURATION = 15.0            # hard cap on a single listen session → revert. Was 60;
                                      # tightened 2026-08-05: in 57 days of server logs every
                                      # genuine window-dependent follow-up landed within ~6s
                                      # (later ones re-spoke the wake word), while a hallucinated
                                      # bare wake in a noisy room let chatter hold the window open
                                      # via VAD until a ≥40%-conf fragment leaked as a query at
                                      # +20.6s (Swedish-kitchen incident). 15s keeps all observed
                                      # genuine use and bounds the chatter-harvest exposure.

# ── Server endpoint ───────────────────────────────────────────────────────────
SERVER_HOST = "127.0.0.1"          # local outposts always talk to the in-process server
SEND_TIMEOUT = 60.0                # seconds; a /stream call stays open while Floxy speaks

# Request source reported to the server. Like the Android/iOS apps, local mic
# outposts are voice interactions — this makes the server treat them as
# VoiceToVoice (spoken replies, fast model) rather than TextChat. Sent explicitly
# because our HTTP client's User-Agent isn't recognized as a mobile app.
OUTPOST_SOURCE = "VoiceToVoice"

# Location reported to the server: the room the server machine's mics are in
# (`local_outposts.location`), one value for all of them until outposts carry their own
# per-device location. Unset, an outpost query lands as "unknown location".
OUTPOST_LOCATION = config.get('local_outposts.location', "") or None

# Mic names containing any of these substrings (case-insensitive) are skipped during
# discovery — they're virtual/aggregate/loopback devices, not real microphones.
VIRTUAL_MIC_NAME_HINTS = (
    "aggregate", "virtual", "blackhole", "soundflower", "loopback",
    "ishowu", "background music", "multi-output", "ca-channel",
    "microsoft teams audio", "iphone microphone",
)

# Cap on how many discovered mics become outposts. 1 = use only the first usable
# mic (handy for testing); set to None to run an outpost on every discovered mic.
MAX_OUTPOSTS = None

