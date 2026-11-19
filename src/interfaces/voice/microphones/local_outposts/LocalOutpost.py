"""A single local microphone outpost.

Owns one mic's audio stream, its own VAD, and the VAD-based endpoint state machine
(ported from the LocalSTTTests whisper script). On each detected utterance it hands
the audio to the shared Whisper worker; the worker's callback gates the transcript
(mute / risk / wake word) and pushes accepted text to the server via the client.

State across mics is independent — each outpost has its own VAD (Silero carries RNN
state per stream) and endpoint buffers. The Whisper model is the only shared piece.
"""

import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np

from .constants import (
    SAMPLE_RATE, FRAME, FRAME_DURATION, AUDIO_QUEUE_TIMEOUT,
    SILENCE, SPEC_SILENCE, SPEC_STALE_SPEECH_GRACE, MIN_SPEECH, MAX_SEGMENT,
    SPECULATIVE, CONCAT_MAX_GAP,
    SCRIBE_DISAGREE_MAX_SIMILARITY, SCRIBE_DISAGREE_MAX_SCRIBE_CONF,
    SCRIBE_DISAGREE_MAX_WHISPER_CONF,
    VAD_BACKEND, VAD_THRESHOLD, WAKE_WORDS, WAKE_WORD_NEAR_MISSES, WAKE_WORD_ANCHOR_CHARS,
    INITIAL_PROMPT_PHRASES,
    RISK_CONFIDENCE_MIN_PCT, WAKE_WORD_RISK_CONFIDENCE_MIN_PCT,
    WAKE_WORD_PREFIX_CONFIDENCE_MIN_PCT,
    SCRIBE_CONF_CALIBRATION, SCRIBE_MIN_CONFIDENCE, SCRIBE_VERIFY_WAKE_WORD,
    SCRIBE_SKIP_CONFIDENCE_PCT, ISO639_3_TO_1,
    WAKE_WORD_RESCUE_WHISPER_CONFIDENCE_MIN_PCT, WAKE_WORD_RESCUE_SCRIBE_CONFIDENCE_MIN,
    WAKE_WORD_RESCUE_SIMILARITY_MIN,
    ENGINE_WHISPER, ENGINE_SCRIBE,
    RMS_CALIBRATION_SECONDS, RMS_NOISE_FLOOR_FALLBACK,
    RMS_THRESHOLD_MULTIPLIER, RMS_THRESHOLD_FLOOR,
    LISTEN_FALSE_POSITIVE_TIMEOUT, LISTEN_NO_NEW_WORDS_TIMEOUT, LISTEN_MAX_DURATION,
    MUTE_WATCHDOG_TIMEOUT, STREAM_SILENCE_TIMEOUT, STREAM_REOPEN_BACKOFF_MAX,
    CAPTURE_DROP_MIN_GAP, CAPTURE_DROP_WARMUP_FRAMES, CAPTURE_DROP_STARTUP_GRACE,
    CAPTURE_DROP_SUMMARY_INTERVAL,
    transcript_for_log,
)
from .Endpointer import Endpointer, STARTED, STOPPED
from .VAD import make_vad, rms, RmsVAD
from .WhisperWorker import metrics_risky, wake_word_hallucination_risky, format_metrics
from common.logging import info, trace, note, fine, error
from common.audio_log import save_pcm_wav


# Sentence-final punctuation (plus trailing quotes/brackets) directly before a wake word:
# the previous sentence is COMPLETE, so the wake word opens its own sentence — an address,
# not a mid-sentence mention (see _addressing_wake_word).
_SENTENCE_BOUNDARY_RE = re.compile(r"[.?!…][)\]\"'»…]*\s*$")


def _addressing_wake_word(text: str):
    """(index, wake_word) of the earliest wake word that ADDRESSES us, or None.

    A wake word addresses us when it *opens* the utterance — starts within
    WAKE_WORD_ANCHOR_CHARS of the front — or opens its own sentence (the text before it
    ends with sentence-final punctuation: "What are their names? Michelle, pause the
    music.", the Aug 29 2026 prod miss). A wake word buried mid-sentence is an overheard
    mention ("…berätta för Michelle vad du såg?"), never an address."""
    lower = text.lower()
    best = None
    for wake_word in WAKE_WORDS:
        needle = wake_word.lower()
        idx = lower.find(needle)
        while idx >= 0:
            if idx < WAKE_WORD_ANCHOR_CHARS or _SENTENCE_BOUNDARY_RE.search(text[:idx]):
                if best is None or idx < best[0]:
                    best = (idx, wake_word)
                break
            idx = lower.find(needle, idx + 1)
    return best


def strip_wake_word(text: str):
    """Mirror the Android outpost: find the addressing wake word, return (found, rest).

    `rest` is everything after the wake word, trimmed. Matches case-insensitively.
    The wake word counts when it opens the utterance (within WAKE_WORD_ANCHOR_CHARS,
    Android's `indexOf(wakeWord) in 0..9`) or opens its own sentence after completed
    speech — see _addressing_wake_word. Returns (False, text) otherwise.
    """
    match = _addressing_wake_word(text)
    if match is None:
        return False, text
    idx, wake_word = match
    return True, text[idx + len(wake_word):].strip()


def wake_word_prefix(text: str) -> str:
    """The text before the addressing wake word ('' when none addresses us).
    Lets the caller distinguish a flush fire ("Michelle, …") from one behind a
    spoken lead-in ("Okay, Michelle, …" — and any sentence-boundary fire), which
    needs a stricter confidence floor (WAKE_WORD_PREFIX_CONFIDENCE_MIN_PCT)."""
    match = _addressing_wake_word(text)
    return text[:match[0]] if match else ""


def deep_wake_word_index(text: str) -> int:
    """Position of the earliest wake word that appears past the anchor window, or -1
    if none. Used purely to log the "saw the wake word mid-sentence, ignored it"
    case — the transcripts that strip_wake_word now deliberately lets through."""
    lower = text.lower()
    idxs = [i for i in (lower.find(w) for w in WAKE_WORDS)
            if i >= WAKE_WORD_ANCHOR_CHARS]
    return min(idxs) if idxs else -1


# Punctuation/whitespace left over after stripping the wake word (e.g. the comma in
# "Michelle, what's the pool temperature?" → ", what's …").
_LEADING_SEPARATORS = " \t\n\r,.;:·-–—"


def clean_query(text: str) -> str:
    """Tidy a query before sending: drop leading separators, strip, and capitalize
    the first letter (leaving the rest of the casing intact)."""
    cleaned = text.lstrip(_LEADING_SEPARATORS).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _normalize_for_match(text: str) -> str:
    """Reduce a transcript to its bare words (lowercase, alphanumerics only) so
    punctuation/casing differences don't defeat an exact phrase comparison."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def doubled_wake_word(text: str) -> bool:
    """True when the same wake word appears twice in a row ("Michelle Michelle, …").
    Nobody addresses the assistant that way — on WakeWordTest the doubling appears in
    0 of 210 genuine fires and only in hallucinated decodes of non-command speech
    (Swedish household talk force-decoded as English, both engines primed with the
    wake word) — so it marks a false fire, whichever engine produced it."""
    tokens = _normalize_for_match(text).split()
    wake = {_normalize_for_match(w) for w in WAKE_WORDS}
    return any(a == b and a in wake for a, b in zip(tokens, tokens[1:]))


# Normalized priming phrases. Whisper hallucinates the prompt back during silence — either
# the whole phrase, or (it continues from where the prompt left off) just its LEADING run,
# e.g. the identity sentence "You are Michelle, our AI home assistant." That chunk opens with
# the wake word, so unless we recognise it as an echo it false-fires (observed in prod). So we
# treat any transcript that is a word-prefix of a priming phrase as an echo, not only a
# verbatim whole-phrase match.
_PRIMING_PHRASES_NORM = tuple(_normalize_for_match(p) for p in INITIAL_PROMPT_PHRASES)

# A prefix shorter than this many words is too generic to attribute to the prompt; the
# identity sentence behind the real false fires is 7 words ("you are michelle our ai home
# assistant"). Full-phrase echoes match regardless of this floor.
_PRIMING_ECHO_MIN_PREFIX_WORDS = 3


def is_priming_echo(text: str) -> bool:
    """True when the transcript is a Whisper priming-phrase echo — the whole phrase parroted
    back, or just a leading run of it (the prompt's identity sentence and on) — a silence
    hallucination that must never reach the server. The prefix is matched word-for-word from
    the start of a priming phrase, so only a literal echo of the prompt's opening qualifies,
    never a genuine command (no command begins 'you are Michelle, our AI home assistant…')."""
    norm = _normalize_for_match(text)
    if not norm:
        return False
    if any(phrase == norm for phrase in _PRIMING_PHRASES_NORM):
        return True
    if len(norm.split()) < _PRIMING_ECHO_MIN_PREFIX_WORDS:
        return False
    prefix = norm + " "
    return any(phrase.startswith(prefix) for phrase in _PRIMING_PHRASES_NORM)


def _has_words(text) -> bool:
    return any(c.isalnum() for c in (text or ""))


def scribe_verify_agreement_veto(whisper_query, scribe_rest):
    """Bare-echo veto for a Scribe VERIFY of a Whisper wake-word fire: Whisper heard
    "<wake word> <command>" but Scribe heard ONLY the bare wake word — background chatter
    that merely names the assistant, not an address. Zero of the 208 genuine fused fires on
    WakeWordTest have this shape (when Whisper hears a command, Scribe always hears one
    too); it kills the 2026-08-09 prod FP (Whisper "Michelle is done." / Scribe "Michelle").

    Both sides are compared on WORD content, not string emptiness: a genuine bare
    "Michelle!" decodes to a punctuation-only whisper query ("!" survives clean_query's
    separator strip), which is not a command — so a wordless whisper_query never vetoes,
    keeping genuine bare-wake-word addresses (bare in BOTH engines) alive.

    Returns a human-readable veto reason, or None when the verify stands."""
    if _has_words(whisper_query) and not _has_words(scribe_rest):
        return "scribe heard only the bare wake word, no command behind it"
    return None


def command_similarity(a: str, b: str) -> float:
    """How alike two command texts are, 0..1: Jaccard overlap of letter bigrams over their
    de-spaced lowercase letters, so "goodnight" and "good night" are identical and
    punctuation is ignored. Two empty texts are identical; one empty text is nothing alike."""
    a, b = (re.sub(r"[^a-zåäö]", "", (t or "").lower()) for t in (a, b))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ga = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    gb = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    return len(ga & gb) / len(ga | gb)


# "please" is deliberately NOT here: two engines that both heard "please" heard an address
# ("one minute, please" / "volume really please" for "Michelle, when will it rain, please").
_FUNCTION_WORDS = frozenset(
    "the a an to of is it my me you i and or on in for that this what can could do "
    "be am are was your we us they he she".split())


def shares_content_word(a: str, b: str) -> bool:
    """True when the two texts have a real word in common (not a function word like
    "the"/"you"): two engines that both heard "today" heard the same utterance, however
    differently they rendered the rest."""
    words = lambda t: {w for w in re.findall(r"[a-zåäö']+", (t or "").lower())
                       if len(w) >= 3 and w not in _FUNCTION_WORDS}
    return bool(words(a) & words(b))


def matching_command_prefix(whisper_query, scribe_rest):
    """When Scribe's command text STARTS with exactly Whisper's command (word-for-word,
    normalized), the engines agree on the address and anything Scribe heard after it is
    background bleed — Scribe endpoints longer than Whisper and can capture overheard
    speech behind a genuine command ("Michelle, pause the music. Ta kort på det numret…",
    Aug 29 2026 prod miss: the Swedish tail tripped the foreign-language veto on a fire
    Scribe had confirmed word-for-word). Returns Scribe's rest truncated to the agreed
    command (so the tail never becomes part of the vote), or None when the prefix
    doesn't match."""
    wq = _normalize_for_match(whisper_query)
    sq = _normalize_for_match(scribe_rest)
    if not wq or not (sq == wq or sq.startswith(wq + " ")):
        return None
    # Truncate with the SAME tokenizer the comparison used: _normalize_for_match
    # splits on ASCII [a-z0-9], where a Unicode \w+ walk counts words differently
    # ("släck" is one \w+ token but two normalized tokens), letting background
    # bleed survive the cut on non-ASCII commands.
    n = len(wq.split())
    tokens = list(re.finditer(r"[a-z0-9]+", scribe_rest.lower()))
    if len(tokens) < n:
        return scribe_rest
    # The last ASCII token may end mid-word ("sätt på" -> "sätt p"): keep the rest of
    # the word's letters so a trailing å/ä/ö isn't chopped off the agreed command.
    end = tokens[n - 1].end()
    while end < len(scribe_rest) and scribe_rest[end].isalpha():
        end += 1
    return scribe_rest[:end]


def scribe_verify_disagreement_veto(whisper_query, scribe_rest, whisper_conf, scribe_conf):
    """Disagreement veto for a Scribe VERIFY: both engines heard the wake word, but their
    command texts share nothing AND neither engine is sure — the correlated hallucination
    shape (Swedish household speech force-decoded as "Michelle <something>" by both engines,
    each guessing a different something). A genuine address where one engine botched the
    command has the OTHER one confident, so it stands — and two renderings that share a
    real word ("what would you like to drink today?" / "will it rain today?") are one
    utterance heard twice, never a disagreement. See SCRIBE_DISAGREE_* in
    constants.py for the sweep. `whisper_conf` is Whisper's 0..1 clip confidence,
    `scribe_conf` Scribe's raw 0..1 confidence; a missing one disables the veto.

    Never fires without a command on both sides (bare fires have their own veto).
    Returns a veto reason, or None when the verify stands."""
    if whisper_conf is None or scribe_conf is None:
        return None
    if not _has_words(whisper_query) or not _has_words(scribe_rest):
        return None
    if shares_content_word(whisper_query, scribe_rest):
        return None
    sim = command_similarity(whisper_query, scribe_rest)
    if (sim <= SCRIBE_DISAGREE_MAX_SIMILARITY and scribe_conf < SCRIBE_DISAGREE_MAX_SCRIBE_CONF
            and whisper_conf < SCRIBE_DISAGREE_MAX_WHISPER_CONF):
        return (f"engines disagree on the command (similarity {sim:.2f}) and neither is sure "
                f"(whisper {whisper_conf:.0%}, scribe {scribe_conf:.0%})")
    return None


def near_wake_word_token(text: str) -> str | None:
    """The opening word(s) of `text` when they are one of the WAKE_WORD_NEAR_MISSES — a
    rendering Scribe is known to give a cleanly spoken wake word ("Mission complete" for
    "Michelle good night", "We shall volume down" for "Michelle volume down"); None
    otherwise. Only the opening counts: a near-miss buried in the sentence is no evidence
    the utterance was an address. Entries may span several words. Used solely by the
    Scribe verify path — a near-miss never triggers a fire on its own."""
    tokens = re.findall(r"[a-zåäö]+", text.lower())
    for miss in WAKE_WORD_NEAR_MISSES:
        n = len(miss.split())
        if n and tokens[:n] == miss.split():
            return miss
    return None


def _after_opening_tokens(text: str, n: int) -> str:
    """`text` with its first `n` words (and any punctuation up to them) removed."""
    m = re.match(r"\W*" + r"\w+\W*" * n, text)
    return text[m.end():].strip() if m else ""


def is_foreign_transcript(text, lang_detector) -> bool:
    """True when the language arbiter is confident the transcript is a non-command
    (background) language we don't accept — so a wake word in it is overheard household
    speech, not an address, and the fire must be vetoed. Guards a missing detector.
    The wake word is English, so strip it before classifying — mirroring
    LocalOutpost._arbitrate_language — otherwise on a short clip ("Michelle, ja") the
    English wake word dilutes the foreign signal below threshold and the veto misses."""
    if lang_detector is None:
        return False
    _, rest = strip_wake_word(text)
    return lang_detector.is_foreign(rest.strip() or text)


# ── The wake-word / hallucination gate, as pure decisions ──
# LocalOutpost._on_transcript and _on_scribe_transcript apply these to live decodes; the
# offline harness (Tools/WakeWordTest/detect.py) applies the SAME functions to recorded
# clips, so the harness can never drift from what production actually does. Keep every
# decision here and only the side effects (logging, sends, state) in the methods.

@dataclass
class WhisperGateVerdict:
    found: bool             # a wake word opens the transcript
    rest: str               # command text after the wake word (the whole text when not found)
    drop: str | None        # why the transcript is dropped; None when it passes the gate
    quiet: bool = True      # a routine drop (empty/echo/risky → fine/trace) vs a wake-word veto (info)
    conf_min_pct: float = RISK_CONFIDENCE_MIN_PCT   # the confidence floor the risk gate applied
    rescue: bool = False    # a risky drop Scribe may still rescue (see gate_scribe_rescue)


def gate_whisper_transcript(text, metrics, lang_detector=None) -> WhisperGateVerdict:
    """Decide whether a Whisper transcript survives the hallucination / wake-word gate.

    A passing verdict with found=True is a wake-word fire; found=False is a follow-up (or,
    while awaiting a wake word, nothing — that state decision stays in _on_transcript)."""
    text = (text or "").strip()
    if not text:
        return WhisperGateVerdict(False, "", "empty transcript")
    if is_priming_echo(text):
        return WhisperGateVerdict(False, "", "priming-phrase echo (Whisper hallucination)")

    # Wake-word clips get their own confidence floor (WAKE_WORD_RISK_CONFIDENCE_MIN_PCT):
    # a one-word "Michelle" averages low on context alone. It's currently set equal to
    # the standard floor — a looser value let too many false positives through on
    # Whisper v2 — but the per-clip override stays wired up. The no_speech/compression
    # gates still catch silence-hallucinated wake words.
    found, rest = strip_wake_word(text)
    conf_min_pct = WAKE_WORD_RISK_CONFIDENCE_MIN_PCT if found else RISK_CONFIDENCE_MIN_PCT

    def verdict(drop=None, quiet=True):
        return WhisperGateVerdict(found, rest, drop, quiet, conf_min_pct)

    if metrics_risky(metrics, conf_min_pct):
        # The no_speech ceiling misfires on speech over background music ("Michelle, pause
        # the music" during playback: conf 57–58%, no_speech 0.23–0.51). A risky drop whose
        # transcript still opens FLUSH with the wake word and a command, at decent
        # confidence and clear of every hallucination signature, is marked rescueable: the
        # caller holds it for Scribe, which fires it only on a positive confirmation
        # (gate_scribe_rescue). Everything else drops exactly as before.
        conf_pct = metrics["conf"] * 100 if metrics and metrics.get("conf") is not None else 0.0
        rescueable = (found and _has_words(rest)
                      and not any(c.isalnum() for c in wake_word_prefix(text))
                      and conf_pct >= WAKE_WORD_RESCUE_WHISPER_CONFIDENCE_MIN_PCT
                      and not wake_word_hallucination_risky(metrics)
                      and not doubled_wake_word(text)
                      and not is_foreign_transcript(text, lang_detector))
        return WhisperGateVerdict(found, rest, "risky transcript (likely hallucination)",
                                  True, conf_min_pct, rescue=rescueable)
    if not found:
        return verdict()

    # A wake-word fire that is both low-confidence AND breathy is a correlated
    # cross-engine hallucination (both engines invent the wake word from foreign/noise
    # audio) that the standard floors and the Scribe fusion gate can't catch alone.
    if wake_word_hallucination_risky(metrics):
        return verdict("low-confidence AND breathy (correlated hallucination signature)", quiet=False)
    # A doubled wake word is another hallucination signature (see doubled_wake_word);
    # confidence can't catch it — the Aug 2026 prod decodes carried it at up to 100%.
    if doubled_wake_word(text):
        return verdict("doubled wake word (hallucination signature)", quiet=False)
    # A wake word behind spoken words ("Okay, Michelle…", "I'm Michelle…") is almost
    # always overheard chatter or a priming echo, not an address — measured in prod,
    # 48 of 49 such fires were junk (see WAKE_WORD_PREFIX_CONFIDENCE_MIN_PCT). Only a
    # decisively confident transcript may fire from behind a prefix; punctuation-only
    # lead-ins (a stray quote or dash) don't count as one.
    prefix = wake_word_prefix(text)
    if any(c.isalnum() for c in prefix):
        conf_pct = metrics["conf"] * 100 if metrics and metrics.get("conf") is not None else 0.0
        if conf_pct < WAKE_WORD_PREFIX_CONFIDENCE_MIN_PCT:
            return verdict(f"spoken prefix '{prefix.strip()}' before the wake word at only "
                           f"{conf_pct:.1f}% confidence "
                           f"(< {WAKE_WORD_PREFIX_CONFIDENCE_MIN_PCT:.0f}% prefix floor)",
                           quiet=False)
    # A wake word inside a confidently non-command-language transcript (overheard
    # household speech, e.g. Swedish in an English-only home) is a hallucinated address,
    # not a command. Only checked when a wake word opens the clip — a continuation already
    # belongs to a session an accepted-language wake word opened.
    if is_foreign_transcript(text, lang_detector):
        return verdict("transcript is a non-command language (overheard background speech), "
                       "not an address", quiet=False)
    return verdict()


def should_skip_scribe(confidence, query) -> bool:
    """At high Whisper confidence Scribe has nothing left to veto — in two months of prod
    no COMMAND fire at ≥52% conf was ever vetoed (see SCRIBE_SKIP_CONFIDENCE_PCT). A bare
    wake-word fire (empty query) is always verified regardless: a spurious bare fire arms
    an unguarded 15s listen window (the 2026-08-05 prod FP)."""
    return (bool(query) and confidence is not None
            and confidence * 100 >= SCRIBE_SKIP_CONFIDENCE_PCT)


@dataclass
class ScribeVerifyVerdict:
    found: bool             # Scribe heard the wake word (anywhere — see the anchor note)
    rest: str               # Scribe's command text after the wake word
    veto: str | None        # why Whisper's fire is refuted; None when the verify stands
    echo: bool = False      # Scribe parroted a priming phrase: no veto, but no vote either
    near_miss: str | None = None  # Scribe opened with a near-homophone of the wake word
                            # ("Mission complete"): the fire stands, but Scribe evidently
                            # misheard the opening, so its command text casts no vote


def gate_scribe_verify(text, whisper_query, lang_detector=None,
                       whisper_conf=None, scribe_conf=None) -> ScribeVerifyVerdict:
    """Decide whether Scribe's transcript confirms or refutes a Whisper wake-word fire for
    `whisper_query` (the cleaned command Whisper sent; '' for a bare wake word).
    `whisper_conf` / `scribe_conf` are the engines' 0..1 confidences for the
    disagreement veto (a missing one disables it).

    Scribe is the more accurate engine, so if it does NOT confirm the wake word — heard no
    speech at all (empty), a non-command language, or simply a different phrase with no
    wake word — Whisper's detection is a false positive."""
    text = (text or "").strip()
    if not text:
        # An empty transcript means Scribe ran and heard no speech at all — a network hiccup
        # never reaches here (ScribeWorker._run swallows the error before the callback).
        # "Heard nothing" refutes a wake-word fire as strongly as "no wake word".
        return ScribeVerifyVerdict(False, "", "empty transcript (Scribe heard no speech)")
    if is_priming_echo(text):
        return ScribeVerifyVerdict(False, "", None, echo=True)

    found, rest = strip_wake_word(text)
    # When VERIFYING a Whisper wake-word fire the anchor (wake word must OPEN the clip)
    # does not apply: Scribe and Whisper endpoint differently, so Scribe routinely captures
    # extra leading speech that pushes the wake word past the anchor — e.g. Whisper's
    # "Michelle, what's the pool temperature?" vs Scribe's "…let's go out. Michelle, what's
    # the pool temperature?". That deeper match is Scribe CONFIRMING the wake word, not
    # disagreeing, so it must not veto. Promote a deep wake word to a found one (and strip
    # from it, so the vote is the command, not the preamble).
    if not found:
        deep_idx = deep_wake_word_index(text)
        if deep_idx != -1:
            found, rest = strip_wake_word(text[deep_idx:])

    # Scribe heard no wake word — but if its transcript OPENS with one of the words it is
    # known to misspell the wake word as ("Mission complete" for "Michelle good night", see
    # WAKE_WORD_NEAR_MISSES) the wake-word sound was there: Whisper heard the exact wake word
    # flush at the front and cleared every hallucination veto, and a hallucinated fire has
    # Scribe hearing unrelated speech instead. Only a COMMAND fire is rescued this way
    # and only when Scribe heard words behind the near-miss too: a bare wake-word fire arms
    # an unguarded listen window, and a near-miss with nothing behind it is the bare-echo
    # shape, so both keep the strict veto.
    near_miss = None
    if not found:
        near_miss = near_wake_word_token(text)
        if near_miss and _has_words(whisper_query):
            after = _after_opening_tokens(text, len(near_miss.split()))
            if _has_words(after):
                found, rest = True, after
            else:
                near_miss = None
        else:
            near_miss = None

    if not found:
        return ScribeVerifyVerdict(False, rest, f"no wake word in '{text}'")
    # Scribe heard the wake word AND its command opens with exactly Whisper's command:
    # the engines agree on the address, so trailing background bleed must not trip the
    # foreign-language veto — and must not become part of the vote either (truncate).
    agreed = matching_command_prefix(whisper_query, rest)
    if agreed is not None:
        rest = agreed
    elif is_foreign_transcript(text, lang_detector):
        return ScribeVerifyVerdict(
            True, rest, f"a non-command language (overheard background speech) in '{text}'")
    # A doubled wake word in Scribe's decode ("Michelle Michelle …") refutes the fire —
    # a hallucination signature, never a genuine address (see doubled_wake_word). Both
    # Aug 2026 doubled prod decodes were Scribe's, on speech with no wake word at all.
    if doubled_wake_word(text):
        return ScribeVerifyVerdict(
            True, rest, f"doubled wake word (hallucination signature) in '{text}'")
    # Scribe heard the wake word, but a bare wake-word echo of a Whisper command still
    # refutes the fire (see scribe_verify_agreement_veto).
    veto = scribe_verify_agreement_veto(whisper_query, rest)
    if veto:
        return ScribeVerifyVerdict(True, rest, f"{veto} in '{text}'")
    # A near-miss opening already marks Scribe's command text as untrustworthy (no vote),
    # so only an exact wake-word hearing is held to the disagreement veto.
    if near_miss is None:
        veto = scribe_verify_disagreement_veto(whisper_query, rest, whisper_conf, scribe_conf)
        if veto:
            return ScribeVerifyVerdict(True, rest, f"{veto} in '{text}'")
    return ScribeVerifyVerdict(True, rest, None, near_miss=near_miss)


def gate_scribe_rescue(text, whisper_query, lang_detector=None,
                       scribe_conf=None) -> ScribeVerifyVerdict:
    """Decide whether Scribe's transcript RESCUES a Whisper wake-word fire the risk gate
    dropped (WhisperGateVerdict.rescue — flush wake word + command at conf ≥ 50% whose
    no_speech/compression tripped, the background-music shape).

    Stricter than a verify: a verify only needs to not-refute a fire Whisper already made,
    a rescue must POSITIVELY confirm one Whisper wasn't sure enough to make. On top of the
    full verify gate (foreign / doubled / bare-echo vetoes) Scribe must be near-certain
    (raw conf ≥ WAKE_WORD_RESCUE_SCRIBE_CONFIDENCE_MIN) and the two commands must agree
    (shared content word or bigram similarity ≥ WAKE_WORD_RESCUE_SIMILARITY_MIN) — and the
    weaker forms of confirmation (a priming echo, a near-miss opening) that merely leave a
    verify standing are no evidence at all here. A refused rescue simply stays dropped, so
    Scribe's run-to-run confidence drift fails safe. `fired` means the rescue stands;
    `rest` is Scribe's command text, which the rescued send carries (Scribe is the engine
    that was actually sure)."""
    verdict = gate_scribe_verify(text, whisper_query, lang_detector)
    if verdict.veto:
        return verdict
    if verdict.echo:
        return ScribeVerifyVerdict(False, "", "a priming-phrase echo is no positive "
                                              "confirmation — rescue refused")
    if verdict.near_miss:
        return ScribeVerifyVerdict(True, verdict.rest,
                                   f"near-miss opening '{verdict.near_miss}' is no positive "
                                   f"confirmation — rescue refused")
    if scribe_conf is None or scribe_conf < WAKE_WORD_RESCUE_SCRIBE_CONFIDENCE_MIN:
        conf_str = f"{scribe_conf:.0%}" if scribe_conf is not None else "n/a"
        return ScribeVerifyVerdict(True, verdict.rest,
                                   f"scribe confidence {conf_str} below the "
                                   f"{WAKE_WORD_RESCUE_SCRIBE_CONFIDENCE_MIN:.0%} rescue floor")
    if not (shares_content_word(whisper_query, verdict.rest)
            or command_similarity(whisper_query, verdict.rest)
            >= WAKE_WORD_RESCUE_SIMILARITY_MIN):
        return ScribeVerifyVerdict(True, verdict.rest,
                                   f"commands disagree ('{whisper_query}' vs "
                                   f"'{verdict.rest}') — rescue refused")
    return verdict


class _StreamStalled(Exception):
    """Raised by the capture loop when the audio stream has delivered no frames for
    STREAM_SILENCE_TIMEOUT — signals the supervisor to reopen the (silently dead) stream."""


class LocalOutpost:
    """Captures one local microphone, endpoints utterances with its VAD and hands the audio to Whisper.

    The worker's callback gates the transcript (mute / risk / wake word) and pushes accepted
    text to the server via the client. Each outpost owns its own VAD and endpoint state;
    the Whisper model is the only shared piece.
    """

    def __init__(self, device_index, device_name, worker, client, default_lang,
                 lang_by_subtag, lang_detector=None, scribe_worker=None,
                 vad_backend=VAD_BACKEND, vad_threshold=VAD_THRESHOLD):
        self.device_index = device_index
        self.device_name = device_name
        self.label = f"{device_name:<30}"
        self._worker = worker
        self._client = client
        self._scribe = scribe_worker           # ElevenLabs Scribe v2, runs in parallel with Whisper
        self.default_lang = default_lang        # full locale used when detection is unavailable
        self.lang_by_subtag = lang_by_subtag    # {"en": "en-US", "sv": "sv-SE"}: detected → /stream locale
        # Language hint sent to Scribe: pin the single configured language so it's primed for it;
        # leave None (auto-detect) when several are configured, mirroring Whisper's forced-vs-auto.
        self._scribe_lang_hint = default_lang if len(lang_by_subtag) == 1 else None
        self._lang_detector = lang_detector     # text-based arbiter for Whisper's flaky audio label
        self._vad_backend = vad_backend
        self._vad_threshold = vad_threshold

        # Cross-thread control flags (set by the manager from the command listener).
        self.muted = False
        self._muted_at = None   # time.time() when muted; drives the stuck-mute watchdog
        self.awaiting_wakeword = True
        # True while this mic is mid-utterance (VAD start → utterance reset). Read by
        # the manager's keep-warm gate so a keep-warm decode never starts while a real
        # decode may be imminent. Cross-thread read of a bool — benign.
        self.capturing_speech = False

        # Cross-fragment utterance accumulator. The outpost endpoints on silence, so one
        # command split by a pause > SILENCE arrives as several disjoint Whisper segments;
        # we re-join them into the cumulative growing phrase the server expects (mirrors
        # the phones' streaming STT). See _on_transcript / CONCAT_MAX_GAP.
        self._acc_text = ""             # running joined text of the current utterance
        self._acc_had_wake_word = False  # this utterance opened with the wake word
        self._acc_sent_wake_word = False  # already sent the wake-word reset for this utterance
        self._acc_last_stop_at = None    # vad_stop_at of the last appended fragment

        # Follow-up listen timers (mirror Android): revert to wake word if the user
        # goes quiet after the AI asks a question.
        self._timer_lock = threading.Lock()
        self._no_words_timer = None
        self._max_listen_timer = None

        self._audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stop = threading.Event()
        self._stream = None
        self._last_frame_at = 0.0   # time.time() of the most recent audio frame; drives the stream watchdog
        self._thread = None
        self._vad = None

        # Capture-drop detection. The audio callback compares the hardware ADC capture
        # clock across buffers to spot samples the device captured but never delivered
        # (a gap in the recording). It only does arithmetic + a lock-guarded counter bump
        # — the actual error() is emitted from the endpoint thread, never the callback
        # (its print I/O could stall capture and cause the very drops we're watching for).
        self._drop_lock = threading.Lock()
        self._last_adc = None           # previous callback's inputBufferAdcTime
        self._last_adc_frames = None    # previous callback's frame count
        self._cb_count = 0              # callbacks since the stream (re)opened (warm-up skip)
        self._pending_drops = []        # [(time.time(), gap_seconds)] awaiting an error() from the endpoint thread
        self._pending_status = []       # PortAudio status flags awaiting a log line
        self._started_at = 0.0          # time.time() when this mic started; drops within CAPTURE_DROP_STARTUP_GRACE of it are suppressed
        self._pending_suppressed = []   # [gap_seconds] startup-grace drops awaiting a trace() from the endpoint thread
        self._suppressed_drop_count = 0  # cumulative drops swallowed by the startup grace
        self._drop_count = 0            # cumulative dropped-audio gaps for this mic
        self._total_dropped_s = 0.0     # cumulative seconds of audio lost at capture
        self._drops_at_last_summary = 0
        self._last_drop_summary_at = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        self._vad = make_vad(self._vad_backend, self._vad_threshold, label=self.label)
        # Shadow energy gate for speculative timing: energy reacts within one frame of
        # the speaker going quiet, while silero's offset scores decay for ~0.3–0.5s — so
        # the speculative fires on the energy gate and silero keeps owning onset and the
        # real endpoint. When the main VAD is already RMS the two are the same instance.
        self._energy_vad = self._vad if self._vad.name == "rms" else RmsVAD(label=self.label)
        self._started_at = time.time()   # opens the CAPTURE_DROP_STARTUP_GRACE window before the stream can fire callbacks
        self._open_stream()
        self._thread = threading.Thread(target=self._run, name=f"LocalOutpost-{self.device_index}", daemon=True)
        self._thread.start()
        info(f"[{self.label}] started (device #{self.device_index}, VAD={self._vad.name}).")

    def stop(self):
        self._stop.set()
        self._cancel_listen_timers()
        self._close_stream()
        info(f"[{self.label}] stopped.")

    def is_alive(self):
        """True while the capture thread is running and the outpost hasn't been told to
        stop. The manager's recovery loop uses this to spot a mic whose thread has died."""
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def metrics(self):
        """Snapshot of this mic's capture-queue depth and drop counters for the resource
        monitor. A climbing _audio_q means the endpoint thread is draining slower than the
        callback fills it — the back-pressure that precedes the capture-drop storm."""
        with self._drop_lock:
            pending, drops, lost = len(self._pending_drops), self._drop_count, self._total_dropped_s
        return {
            "name": self.device_name,
            "q": self._audio_q.qsize(),
            "pending_drops": pending,
            "drops": drops,
            "lost_s": lost,
            "alive": self.is_alive(),
        }

    def seconds_since_last_frame(self):
        """Seconds since the audio callback last delivered a frame. The callback fires even
        while muted, so a healthy mic stays near 0; a value climbing past STREAM_SILENCE_TIMEOUT
        means the stream has gone silent (device asleep, unplugged, or wedged). Drives the
        manager's offline detection."""
        return time.time() - self._last_frame_at

    def _open_stream(self):
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=FRAME, device=self.device_index, callback=self._audio_callback,
        )
        self._stream.start()
        # Prime the watchdog clock so a slow first frame can't trip it during startup.
        self._last_frame_at = time.time()
        # The ADC clock restarts with a fresh stream — drop the old baseline so the first
        # callback after a (re)open just re-primes it instead of reading a giant false gap.
        self._last_adc = None
        self._last_adc_frames = None
        self._cb_count = 0

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        now = time.time()
        self._last_frame_at = now
        self._detect_capture_drop(time_info, frames, status, now)
        self._audio_q.put(indata[:, 0].copy())

    def _detect_capture_drop(self, time_info, frames, status, now):
        """Spot audio the device captured but never delivered — a gap in the recording.
        PortAudio should advance inputBufferAdcTime by exactly frames/SAMPLE_RATE each
        callback; any excess is samples dropped at the device because this callback was
        starved (CPU/GIL contention). The excess is the gap size. Runs in the PortAudio
        callback, so it only records — the endpoint thread emits the error() (see
        _drain_capture_drops): logging here could itself stall capture."""
        adc = getattr(time_info, "inputBufferAdcTime", None)
        self._cb_count += 1
        prev_adc, prev_frames = self._last_adc, self._last_adc_frames
        if adc is not None:
            self._last_adc, self._last_adc_frames = adc, frames
        # Need a previous buffer to compare; skip the post-(re)open settling callbacks.
        if (adc is None or prev_adc is None
                or self._cb_count <= CAPTURE_DROP_WARMUP_FRAMES):
            if status:
                with self._drop_lock:
                    self._pending_status.append(str(status))
            return
        gap = (adc - prev_adc) - prev_frames / SAMPLE_RATE
        with self._drop_lock:
            if status:
                self._pending_status.append(str(status))
            if gap >= CAPTURE_DROP_MIN_GAP:
                if self._started_at and now - self._started_at < CAPTURE_DROP_STARTUP_GRACE:
                    # Boot-time model loading hammers the CPU/GIL and starves this callback,
                    # producing ADC-clock holes that aren't real recording gaps. Swallow them.
                    self._suppressed_drop_count += 1
                    self._pending_suppressed.append(gap)
                else:
                    self._drop_count += 1
                    self._total_dropped_s += gap
                    self._pending_drops.append((now, gap))

    def _drain_capture_drops(self):
        """Emit an error() for every capture drop the audio callback recorded since the
        last tick, plus any PortAudio status flag. Called from the endpoint thread so the
        callback never does blocking log I/O. Also emits a throttled cumulative summary."""
        with self._drop_lock:
            drops = self._pending_drops
            statuses = self._pending_status
            suppressed = self._pending_suppressed
            count, total_s = self._drop_count, self._total_dropped_s
            supp_count = self._suppressed_drop_count
            if drops:
                self._pending_drops = []
            if statuses:
                self._pending_status = []
            if suppressed:
                self._pending_suppressed = []
        for gap in suppressed:
            fine(f"[{self.label}] capture drop ignored — ~{gap * 1000:.0f}ms ADC-clock hole "
                 f"within the {CAPTURE_DROP_STARTUP_GRACE:.0f}s startup grace (boot-time "
                 f"model-load contention starves the callback; not a real recording gap). "
                 f"{supp_count} suppressed since this mic started.")
        for _, gap in drops:
            info(f"[{self.label}] AUDIO DROP — ~{gap * 1000:.0f}ms of audio lost at capture "
                 f"(device captured it but the callback was starved, so PortAudio never "
                 f"delivered it; recordings around now have a gap). "
                 f"Cumulative for this mic: {count} drops, {total_s:.1f}s lost.")
        for s in statuses:
            trace(f"[{self.label}] PortAudio status flag: {s} (unreliable on macOS — the "
                  f"ADC-clock check above is the authoritative drop signal).")
        self._maybe_log_drop_summary(count, total_s)

    def _maybe_log_drop_summary(self, count, total_s):
        """Throttled cumulative capture-drop summary — only when new drops have occurred
        and at most once per CAPTURE_DROP_SUMMARY_INTERVAL."""
        if count == self._drops_at_last_summary:
            return
        now = time.time()
        if now - self._last_drop_summary_at < CAPTURE_DROP_SUMMARY_INTERVAL:
            return
        self._last_drop_summary_at = now
        self._drops_at_last_summary = count
        note(f"[{self.label}] capture-drop summary: {count} drops totalling {total_s:.1f}s "
             f"of audio lost since this mic started.")

    # ── command handling (called from the manager) ───────────────────────────
    def on_mute(self):
        # Floxy is speaking — pause the follow-up timers so they don't fire mid-reply;
        # the post-reply listen/wakeword broadcast sets the next state.
        self.muted = True
        self._muted_at = time.time()
        self._cancel_listen_timers()
        self._reset_accumulator()  # Floxy speaking ends the utterance; don't join across it

    def on_listen(self):
        """Server expects a follow-up — accept the next utterance without a wake word,
        but only for a limited window (Android parity)."""
        self.muted = False
        self._muted_at = None
        self.awaiting_wakeword = False
        self._begin_listen_window()

    def on_wakeword(self):
        """Back to idle — require the wake word again before sending."""
        self.muted = False
        self._muted_at = None
        self.awaiting_wakeword = True
        self._cancel_listen_timers()
        self._reset_accumulator()

    def _check_mute_watchdog(self):
        """Failsafe against a stuck mute. The server unmutes us ('listen'/'wakeword')
        when a reply ends; if that broadcast never arrives (server killed/errored
        mid-reply) a muted mic stays muted forever — it drops every transcript and,
        because it can't produce the next query, nothing ever unmutes it. After too
        long muted, auto-revert to wake-word listening so a single dropped unmute can't
        permanently wedge the mic. Runs on the capture thread (ticks ~every frame)."""
        if not self.muted or self._muted_at is None:
            return
        muted_for = time.time() - self._muted_at
        if muted_for > MUTE_WATCHDOG_TIMEOUT:
            note(f"[{self.label}] still muted {muted_for:.0f}s after the server said 'stop' "
                 f"with no unmute — auto-reverting to wake-word listening (failsafe).")
            self.on_wakeword()

    def _check_stream_watchdog(self):
        """Failsafe against a silently dead capture stream. PortAudio can stop calling
        the audio callback (USB/CoreAudio hiccup, device sleep/reconfigure) without
        raising or flagging a status — the queue then runs dry, the endpoint loop just
        keeps timing out on an empty get(), and the mic goes deaf forever with no error.
        Frames arrive every ~100 ms; if none has for STREAM_SILENCE_TIMEOUT, raise so the
        supervisor reopens the stream. Runs on the capture thread (ticks ~every frame).
        Note: while muted the callback still fires, so a normal mute can't trip this."""
        if time.time() - self._last_frame_at > STREAM_SILENCE_TIMEOUT:
            raise _StreamStalled()

    # ── follow-up listen timeout (mirrors Android's three timers) ─────────────
    def _arm_timer(self, attr, delay, reason):
        timer = threading.Timer(delay, self._listen_timed_out, (reason,))
        timer.daemon = True
        setattr(self, attr, timer)
        timer.start()

    def _begin_listen_window(self):
        """Entered listen mode: arm the false-positive (first utterance) and max-duration
        timers. Resets any in-flight timers."""
        with self._timer_lock:
            self._cancel_timers_locked()
            self._arm_timer("_no_words_timer", LISTEN_FALSE_POSITIVE_TIMEOUT, "no follow-up spoken")
            self._arm_timer("_max_listen_timer", LISTEN_MAX_DURATION, "max listen duration reached")

    def _note_followup_utterance(self):
        """An utterance was accepted while listening: reset the between-words timer
        (leave the max-duration cap running)."""
        with self._timer_lock:
            if self.awaiting_wakeword:
                return  # not in a listen window
            if self._no_words_timer is not None:
                self._no_words_timer.cancel()
            self._arm_timer("_no_words_timer", LISTEN_NO_NEW_WORDS_TIMEOUT, "no further follow-up spoken")

    def _note_speech_started(self):
        """VAD heard the user speaking inside a listen window: hold the between-words
        timer so it can't fire mid-utterance. A slow-to-transcribe reply that began
        in time must not be cut off; the STOP (or the accepted transcript) re-arms it."""
        with self._timer_lock:
            if self.awaiting_wakeword:
                return  # not in a listen window
            if self._no_words_timer is not None:
                self._no_words_timer.cancel()
                self._no_words_timer = None

    def _note_speech_ended(self):
        """VAD ended an utterance inside a listen window: re-arm the between-words
        timer so a transcript that gets dropped (risky / no wake word) or never
        arrives still eventually closes the window."""
        with self._timer_lock:
            if self.awaiting_wakeword:
                return
            if self._no_words_timer is not None:
                self._no_words_timer.cancel()
            self._arm_timer("_no_words_timer", LISTEN_NO_NEW_WORDS_TIMEOUT, "no further follow-up spoken")

    def _listen_timed_out(self, reason):
        if self.awaiting_wakeword:
            return
        trace(f"[{self.label}] follow-up listen timed out ({reason}) — requiring wake word again.")
        self.awaiting_wakeword = True
        self._cancel_listen_timers()

    def _cancel_timers_locked(self):
        for attr in ("_no_words_timer", "_max_listen_timer"):
            timer = getattr(self, attr)
            if timer is not None:
                timer.cancel()
                setattr(self, attr, None)

    def _cancel_listen_timers(self):
        with self._timer_lock:
            self._cancel_timers_locked()

    # ── capture + endpoint loop ───────────────────────────────────────────────
    def _calibrate_rms(self, vad):
        info(f"[{self.label}] calibrating ambient noise (~{RMS_CALIBRATION_SECONDS:.0f}s)…")
        noise = []
        t_end = time.time() + RMS_CALIBRATION_SECONDS
        while time.time() < t_end:
            try:
                noise.append(rms(self._audio_q.get(timeout=AUDIO_QUEUE_TIMEOUT)))
            except queue.Empty:
                break
        floor = float(np.median(noise)) if noise else RMS_NOISE_FLOOR_FALLBACK
        vad.noise_floor = floor
        vad.threshold = max(floor * RMS_THRESHOLD_MULTIPLIER, RMS_THRESHOLD_FLOOR)
        info(f"[{self.label}] noise floor {floor:.4f} → energy speech threshold {vad.threshold:.4f}")

    def _run(self):
        """Supervisor: calibrate once, then run the endpoint loop — restarting it if it
        ever crashes, so one bad frame (a VAD/torch/numpy hiccup) can't silently kill the
        mic's capture thread for the rest of the process. A frame drought (the PortAudio
        stream died without erroring) raises _StreamStalled, which we recover from by
        reopening the stream rather than restarting the whole process."""
        if self._energy_vad.threshold is None:
            self._calibrate_rms(self._energy_vad)
        reopen_failures = 0
        while not self._stop.is_set():
            try:
                self._endpoint_loop()
            except _StreamStalled:
                if reopen_failures == 0:
                    note(f"[{self.label}] no audio for {STREAM_SILENCE_TIMEOUT:.0f}s — capture stream "
                         f"looks dead, reopening it.")
                try:
                    self._close_stream()
                    self._open_stream()
                    reopen_failures = 0
                    info(f"[{self.label}] capture stream reopened.")
                except Exception as e:
                    reopen_failures += 1
                    delay = min(2.0 ** (reopen_failures - 1), STREAM_REOPEN_BACKOFF_MAX)
                    error(f"[{self.label}] failed to reopen capture stream: {e} — "
                          f"retry #{reopen_failures} in {delay:.0f}s.")
                    from .AudioDiagnostics import capture_open_failure
                    capture_open_failure(self.label, self.device_name, e)
                    self._stop.wait(delay)
            except Exception as e:
                error(f"[{self.label}] capture/endpoint loop crashed — restarting it: {e}")
                trace(traceback.format_exc())
                try:
                    self._vad.reset()
                except Exception:
                    pass
                time.sleep(0.5)

    def _endpoint_loop(self):
        # A crash or stream stall mid-utterance re-enters here with fresh locals but
        # would leave self.capturing_speech stuck True — which silently disables the
        # GPU keep-warm until this mic completes an utterance. Fresh entry = no
        # utterance in progress.
        self.capturing_speech = False
        ep = Endpointer()         # pre-roll / captured frames / silero-quiet run — owns the endpoint
        energy_quiet_run = 0.0    # consecutive quiet per the energy gate — times the speculative
        energy_heard = False      # energy gate heard at least one loud frame this utterance
        silero_speech_after_spec = 0.0  # silero 'speech' seconds since the speculative fired
        spec_fired = False
        spec_id = None
        spec_identify_future = None
        spec_trigger = "vad"      # which gate started the winning decode's quiet run ("rms"/"vad")
        speech_started_at = 0.0   # wall-clock of this utterance's first loud frame
        utt_drop_baseline = 0.0   # cumulative dropped-seconds at this utterance's onset

        def reset_utterance():
            nonlocal energy_quiet_run, energy_heard, \
                silero_speech_after_spec, spec_fired, spec_id, spec_identify_future, spec_trigger
            ep.reset()
            energy_quiet_run, energy_heard, silero_speech_after_spec = 0.0, False, 0.0
            spec_fired, spec_id, spec_identify_future = False, None, None
            spec_trigger = "vad"
            self.capturing_speech = False
            self._vad.reset()

        while not self._stop.is_set():
            self._check_mute_watchdog()
            self._check_stream_watchdog()
            self._drain_capture_drops()
            try:
                frame = self._audio_q.get(timeout=AUDIO_QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            # While Floxy is speaking, drop audio and any half-captured utterance so
            # we never flush the server's own voice back as a query.
            if self.muted:
                if ep.has_audio:
                    if spec_id is not None:
                        self._worker.cancel(spec_id)
                    reset_utterance()
                    ep.clear_pre_roll()
                continue

            loud = self._vad.is_speech(frame)
            # Score EVERY frame on the energy gate, not just in-utterance ones: its
            # adaptive noise-floor tracker (see RmsVAD) needs the quiet between
            # utterances, which is most of the day. Silero's verdict rides along as
            # speech_hint so the tracker never adapts to the user's own voice — it
            # models the room, not the speaker. Same instance as the main VAD when
            # the backend is RMS — don't score the frame twice.
            energy_loud = (loud if self._energy_vad is self._vad
                           else self._energy_vad.is_speech(frame, speech_hint=loud))

            event = ep.feed(frame, loud)
            if event == STARTED:
                self.capturing_speech = True
                speech_started_at = time.time()
                utt_drop_baseline = self._total_dropped_s
                fine(f"[{self.label}] ▶ VAD start "
                      f"(score={self._vad.last_score:.2f} thr={self._vad.threshold:.2f})")
                self._note_speech_started()
                # Tell the server we're talking again the instant the pause ends —
                # long before this fragment's transcript is ready — so an in-flight
                # incomplete request keeps waiting instead of speaking over the
                # continuation (and muting the mic mid-sentence).
                if self._should_signal_speech_active(speech_started_at):
                    self._client.notify_speech_active(self.label)
            elif ep.in_speech:
                # Cancelling a speculative is symmetric with what fired it. An
                # energy-fired spec dies when ENERGY resumes (speech restarted); silero
                # staying loud over it is just offset decay, handled by the stale-speech
                # grace at VAD stop. A silero-fallback spec fired precisely because the
                # energy gate was reading sustained noise as loud, so energy-loud frames
                # are its normal environment — it dies only when SILERO hears speech
                # again (the pre-hybrid rule). Cross-cancelling silero-fired specs on
                # energy-loud killed every fallback spec ~100ms after birth whenever
                # post-reply hiss sat above the energy threshold (2026-08-12 tests).
                if energy_loud:
                    energy_heard = True
                    energy_quiet_run = 0.0
                    if spec_fired and spec_trigger == "rms":
                        if spec_id is not None:  # energy resumed — the speculative is stale
                            fine(f"[{self.label}] ✗ speculative cancelled — energy resumed "
                                  f"({ep.speech_secs:.2f}s captured at cancel)")
                            self._worker.cancel(spec_id)
                            spec_id = None
                        spec_fired = False
                        silero_speech_after_spec = 0.0
                else:
                    energy_quiet_run += FRAME_DURATION
                if loud:
                    if spec_fired and spec_trigger == "vad":
                        if spec_id is not None:  # silero heard speech again — stale
                            fine(f"[{self.label}] ✗ speculative cancelled — silero heard speech "
                                  f"again ({ep.speech_secs:.2f}s captured at cancel)")
                            self._worker.cancel(spec_id)
                            spec_id = None
                        spec_fired = False
                        silero_speech_after_spec = 0.0
                    elif spec_fired:
                        silero_speech_after_spec += FRAME_DURATION
                silence_run = ep.silence_run
                speech_secs = ep.speech_secs
                # Longest quiet run either gate has seen: energy leads during silero's
                # offset decay; the silero run is the fallback for an utterance too soft
                # for the energy gate to have heard at all.
                quiet_run = max(silence_run, energy_quiet_run) if energy_heard else silence_run

                # Once per pause, at the frame a short command would have ended on: a long
                # utterance is given a longer pause before it counts as the end.
                if event != STOPPED and SILENCE <= silence_run < SILENCE + FRAME_DURATION:
                    fine(f"[{self.label}] … still listening after {silence_run:.2f}s of quiet — "
                          f"{speech_secs - silence_run:.1f}s spoken, the utterance ends on "
                          f"{ep.silence_needed:.2f}s")

                if (SPECULATIVE and not spec_fired
                        and quiet_run >= SPEC_SILENCE
                        and speech_secs >= MIN_SPEECH):
                    spec_audio = ep.audio()
                    spec_id = self._worker.submit_speculative(spec_audio, label=self.label)
                    spec_fired = True   # don't retry this pause; spec_id is None if the worker skipped it
                    # Which gate led the quiet run that fired this speculative — surfaced on
                    # the Home tab's Silence detect row when this decode wins the utterance.
                    spec_trigger = "rms" if energy_heard and energy_quiet_run > silence_run else "vad"
                    if spec_id is not None:
                        fine(f"[{self.label}] speculative decode fired after {quiet_run:.2f}s quiet "
                              f"(energy {energy_quiet_run:.2f}s / silero {silence_run:.2f}s, "
                              f"{speech_secs:.2f}s speech)")
                    else:
                        fine(f"[{self.label}] speculative skipped (another decode already in flight) "
                              f"after {quiet_run:.2f}s quiet — final will decode fresh")
                    # Start speaker identification now too, on this same audio, so it
                    # overlaps the remaining silence wait instead of delaying the GET.
                    # Once per utterance — if speech resumes the speaker is unchanged,
                    # so the early sample stays valid and we keep the original future.
                    if spec_identify_future is None:
                        spec_identify_future = self._client.identify_async(spec_audio, self.label)
                        fine(f"[{self.label}] speaker identification started early (speculative).")
                    # Fire Scribe on the utterance-so-far now, for its ~350ms head start so its
                    # vote lands closer to Whisper's. Uncancellable cloud call: if speech resumes,
                    # the in-flight call still completes and votes — a partial that loses to the
                    # complete final vote on the server's length/recency scoring. Cost accepted.
                    # Old always-vote path only (SCRIBE_VERIFY_WAKE_WORD=False); the new gated path
                    # fires Scribe once from _on_transcript after Whisper accepts the command.
                    if self._scribe is not None and not SCRIBE_VERIFY_WAKE_WORD:
                        self._scribe.submit(
                            spec_audio, self._scribe_lang_hint,
                            lambda text, conf, language, t=time.time() - quiet_run:
                                self._on_scribe_transcript(text, conf, language, t),
                            label=self.label)

                if event == STOPPED:
                    reason = "max-segment" if speech_secs >= MAX_SEGMENT else "trailing silence"
                    fine(f"[{self.label}] ⏹ VAD stop ({reason}) — {speech_secs:.2f}s speech, "
                          f"{silence_run:.2f}s silero-quiet / {energy_quiet_run:.2f}s energy-quiet")
                    # An energy-triggered speculative survived silero reporting speech for
                    # longer than offset decay explains — that was likely genuinely soft
                    # speech its audio lacks, so promoting it would truncate the transcript.
                    # Decode fresh instead: correctness over the head start.
                    if spec_id is not None and silero_speech_after_spec >= SPEC_STALE_SPEECH_GRACE:
                        fine(f"[{self.label}] cancelling speculative — silero heard "
                              f"{silero_speech_after_spec:.2f}s of speech after it fired "
                              f"(>= {SPEC_STALE_SPEECH_GRACE:.1f}s grace, soft speech not decay); "
                              f"final will decode fresh")
                        self._worker.cancel(spec_id)
                        spec_id = None
                    audio = ep.audio()
                    vad_stop_at = time.time()
                    # When the user actually went quiet, per whichever gate noticed first —
                    # the energy gate reacts within a frame, so this includes silero's slow
                    # offset decay in the measured (and displayed) silence-detect time. This
                    # is the reference for user-perceived STT latency; vad_stop_at keeps its
                    # meaning for the fragment-concat gap logic.
                    speech_ended_at = vad_stop_at - quiet_run
                    # Audio dropped at capture between this utterance's onset and now — so the
                    # saved WAV is missing that much in the middle. Attributed to the utterance
                    # for the per-recording warning in _on_transcript.
                    utt_dropped = max(0.0, self._total_dropped_s - utt_drop_baseline)
                    self._worker.submit_final(
                        audio,
                        lambda text, metrics, audio, language, timing=None, t=vad_stop_at,
                               s=speech_started_at, fut=spec_identify_future, d=utt_dropped,
                               e=speech_ended_at, tr=spec_trigger:
                            self._on_transcript(text, metrics, audio, language, t, fut, s, d, e,
                                                timing, tr),
                        spec_id=spec_id, label=self.label)
                    # Scribe votes on the same complete utterance Whisper's final decodes — when
                    # they agree the server gets a strong count=2 vote for that phrase. Old
                    # always-vote path only (SCRIBE_VERIFY_WAKE_WORD=False); the new gated path
                    # fires Scribe from _on_transcript once Whisper has accepted the command.
                    if self._scribe is not None and not SCRIBE_VERIFY_WAKE_WORD:
                        self._scribe.submit(
                            audio, self._scribe_lang_hint,
                            lambda text, conf, language, t=speech_ended_at:
                                self._on_scribe_transcript(text, conf, language, t),
                            label=self.label)
                    reset_utterance()
                    self._note_speech_ended()

    def _resolve_lang(self, detected):
        """Map Whisper's detected primary subtag (e.g. 'sv') to the configured full
        locale (e.g. 'sv-SE') — that locale drives the reply language and TTS voice.
        In practice `detected` is already one of the configured subtags: WhisperWorker's
        restricted-detection pass re-picks any out-of-set auto-detect back into the
        allowed set before we ever see it (see _apply_restricted_detection), so foreign
        speech is no longer dropped here — it's coerced to the nearest configured locale
        and the risk-metric gate in _on_transcript drops genuinely bad audio instead.
        Returns None only defensively (subtag somehow outside the configured set); with
        detection unavailable (None), fall back to the primary."""
        if not detected:
            return self.default_lang
        return self.lang_by_subtag.get(detected.split("-")[0].lower())

    def _arbitrate_language(self, text, detected):
        """Cross-check Whisper's per-clip audio language `detected` against a text
        classification of the transcript, returning the language to actually route on.
        Whisper transcribes correctly but mislabels short clips — e.g. an English wake word
        prefixing a Swedish command pulls the audio label to 'en'. The text arbiter trusts
        the decoded words when they confidently disagree (see TextLanguageDetector). The
        wake word itself is English, so strip it before classifying — otherwise it biases
        the very thing we're trying to correct."""
        if self._lang_detector is None or not self._lang_detector.active:
            return detected
        _, rest = strip_wake_word(text)
        lid_text = rest.strip() or text
        # A clip awaiting a wake word with none in it is dropped downstream, so an override
        # here never routes anything — keep it at fine() rather than note().
        return self._lang_detector.arbitrate(lid_text, detected,
                                             important=self._addressed(text))

    def _addressed(self, text) -> bool:
        """Whether a transcript belongs to a wake-word sequence — it opens with the wake
        word, or lands in the listen window one opened. Anything else is overheard speech,
        whose words stay out of the logs (see LOG_OVERHEARD_TRANSCRIPTS)."""
        return bool(text) and (not self.awaiting_wakeword or strip_wake_word(text)[0])

    def _is_foreign(self, text):
        return is_foreign_transcript(text, self._lang_detector)

    def _should_signal_speech_active(self, speech_started_at) -> bool:
        """True when a speech onset looks like the continuation of an utterance we're
        mid-sending — we're in a listen window, have an accumulator, and the onset is
        within CONCAT_MAX_GAP of the last fragment. Gating this tightly (rather than
        pinging on every onset) keeps room noise / radio from holding open every in-flight
        server request. The capture loop only reaches here when not muted."""
        return (not self.awaiting_wakeword and bool(self._acc_text)
                and self._acc_last_stop_at is not None
                and speech_started_at - self._acc_last_stop_at <= CONCAT_MAX_GAP)

    def _reset_accumulator(self):
        """Drop the cross-fragment utterance accumulator — the next transcript starts a
        fresh utterance (new wake-word command, a gap too long to be one sentence, or
        Floxy started speaking). Touched from the worker thread (_on_transcript) and the
        command thread (on_mute/on_wakeword); the race is benign — worst case a fragment
        starts a fresh utterance instead of joining."""
        self._acc_text = ""
        self._acc_had_wake_word = False
        self._acc_sent_wake_word = False
        self._acc_last_stop_at = None

    def _save_recording_async(self, text, audio, utt_dropped_s, whisper_span=None,
                              addressed=True):
        """Write the debug WAV on a background thread — the synchronous write sat on the
        transcript hot path and added ~0.4s of user-facing latency per query. `audio` is
        a fresh per-utterance array that is never mutated after the VAD stop, so handing
        the reference off is safe. `whisper_span` is how many leading samples Whisper's
        winning decode actually saw when that was less than the whole utterance; it rides
        inside the WAV so WakeWordTest can replay each engine on its prod span."""
        from common.logging import disable_file_logging
        if disable_file_logging:
            return

        def _save():
            try:
                pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                metadata = ({"whisper_span_samples": int(whisper_span), "decode": "spec"}
                            if whisper_span is not None else None)
                path = save_pcm_wav("log/audio", text, pcm, sample_rate=SAMPLE_RATE,
                                    metadata=metadata)
                if utt_dropped_s >= CAPTURE_DROP_MIN_GAP:
                    # The WAV is named after its transcript, so the path is the text.
                    error(f"[{self.label}] saved recording {transcript_for_log(path, addressed)} "
                          f"is INCOMPLETE — ~{utt_dropped_s:.1f}s of audio was dropped at "
                          f"capture during this utterance, leaving a gap in the middle.")
            except Exception as e:
                trace(f"[{self.label}] failed to save raw audio: {e}")
        threading.Thread(target=_save, name=f"WavSave-{self.device_index}", daemon=True).start()

    # ── transcript gate (runs on the shared worker thread) ────────────────────
    def _on_transcript(self, text, metrics, audio, language=None, vad_stop_at=None,
                       identify_future=None, vad_start_at=None, utt_dropped_s=0.0,
                       speech_ended_at=None, timing=None, spec_trigger="vad"):
        addressed = self._addressed(text)
        shown = transcript_for_log(text, addressed)
        if self.muted:
            trace(f"[{self.label}] dropping transcript — muted while Floxy speaks: {shown}")
            return
        if not text:
            trace(f"[{self.label}] empty transcript, nothing to send.")
            return

        language = self._arbitrate_language(text, language)
        lang = self._resolve_lang(language)
        if lang is None:
            trace(f"[{self.label}] dropping transcript — detected language '{language}' not in "
                  f"configured {sorted(self.lang_by_subtag)}: {shown}")
            return

        # STT latency for this mic, measured from when the user actually stopped
        # speaking (speech_ended_at = VAD stop minus the trailing-silence wait), so the
        # console shows what the user perceives — silence-detection wait included.
        # Reported to the server on the actual query send, not set here directly:
        # several mics transcribe the same utterance and the slowest would otherwise
        # clobber the value. The server records it only for the transcription it picks
        # as the winner (is_best_transcription), so the console shows the latency of
        # the mic that actually drove the response.
        stt_ms = None
        stt_wait_ms = None
        stt_trigger = None
        stt_ref = speech_ended_at if speech_ended_at is not None else vad_stop_at
        if stt_ref is not None:
            now = time.time()
            stt_ms = round((now - stt_ref) * 1000)
            breakdown = ""
            if timing is not None:
                # Where the user-perceived STT time went: endpoint = waiting out trailing
                # silence until the winning decode was submitted (~SPEC_SILENCE for a
                # promoted speculative, ~SILENCE for a fresh final); queue = that decode
                # waited behind another mic's job on the single GPU thread; infer = the
                # Whisper decode itself; deliver = decode done → this stamp (includes
                # holding an early speculative until the VAD stop confirmed the utterance).
                endpoint = timing["submitted_at"] - stt_ref
                queue = timing["started_at"] - timing["submitted_at"]
                infer = timing["done_at"] - timing["started_at"]
                deliver = now - timing["done_at"]
                stt_wait_ms = round(endpoint * 1000)
                # Which gate started the winning decode: the energy gate's early fire
                # only counts when the speculative's decode actually produced this text —
                # a fresh final always started at silero's VAD stop.
                stt_trigger = spec_trigger if timing.get("decode") == "spec" else "vad"
                breakdown = (f" (endpoint {endpoint:.2f}s + queue {queue:.2f}s + "
                             f"infer {infer:.2f}s + deliver {deliver:.2f}s, {stt_trigger})")
            fine(f"[{self.label}] STT done {stt_ms / 1000:.2f}s after speech end{breakdown}: "
                 f"{shown}")

        # The WAV holds the full utterance — that is what Scribe verifies below — but
        # Whisper may have decoded less of it: a promoted speculative ran on the
        # utterance-so-far (cut at the energy gate's short quiet), and decoding the longer
        # final span offline gives a different transcript and confidence — three of the
        # four Aug 2026 prod false activations were promoted speculatives that WakeWordTest
        # could not reproduce from the full clip ("…at the dentist?" 33.4% in prod vs
        # "…want to answer?" 29%). The speculative span is always a prefix of the utterance
        # (the capture buffer only grows), so its length is enough for the harness to
        # replay Whisper on exactly the prod span and Scribe on the whole clip.
        decoded_audio = (timing or {}).get("audio")
        whisper_span = None
        if decoded_audio is not None and len(decoded_audio) != len(audio):
            whisper_span = len(decoded_audio)
            fine(f"[{self.label}] Whisper decoded a {whisper_span / SAMPLE_RATE:.2f}s speculative "
                 f"span of the {len(audio) / SAMPLE_RATE:.2f}s utterance — recording the span "
                 f"in the saved WAV.")
        self._save_recording_async(text, audio, utt_dropped_s, whisper_span, addressed)

        # The hallucination / wake-word gate itself is pure (gate_whisper_transcript) so the
        # offline harness runs the identical decision; only the logging lives here.
        verdict = gate_whisper_transcript(text, metrics, self._lang_detector)
        found, rest = verdict.found, verdict.rest
        fine(f"[{self.label}] risk check for {shown}: "
             f"{format_metrics(metrics, verdict.conf_min_pct)}")
        if verdict.drop:
            if verdict.rescue and self._scribe is not None and SCRIBE_VERIFY_WAKE_WORD:
                # A confident, flush wake-word fire the risk gate dropped (background music
                # inflating no_speech) — hold it for Scribe instead of dropping outright:
                # it fires late iff Scribe positively confirms (gate_scribe_rescue).
                info(f"  [{self.label}] risky wake-word fire ({verdict.drop}) — holding for "
                     f"scribe rescue: '{text}'")
                self._scribe.submit(
                    audio, self._scribe_lang_hint,
                    lambda s_text, s_conf, s_lang, t=stt_ref, wq=clean_query(rest), a=audio,
                           fut=identify_future, stop=vad_stop_at:
                        self._on_scribe_rescue(s_text, s_conf, s_lang, t, wq, a, fut, stop),
                    label=self.label)
                return
            if verdict.quiet:
                fine(f"[{self.label}] dropping {verdict.drop}: {shown}")
            else:
                info(f"  [{self.label}] dropping wake-word fire — {verdict.drop}: '{text}'")
            return

        confidence = metrics["conf"] if metrics else None
        if not found:
            deep_idx = deep_wake_word_index(text)
            if deep_idx != -1:
                fine(f"[{self.label}] wake word present at char {deep_idx} but past the "
                     f"{WAKE_WORD_ANCHOR_CHARS}-char anchor — treating as overheard mention, "
                     f"not an address: {shown}")
        # Re-join silence-split fragments into one cumulative phrase (see CONCAT_MAX_GAP).
        # The server's voting expects the growing-phrase contract the phones send, not the
        # disjoint per-endpoint segments Whisper produces — so a wake-word-less fragment
        # that lands within CONCAT_MAX_GAP of the previous one extends it rather than
        # starting a new (half-a-sentence) command.
        gap = (vad_start_at - self._acc_last_stop_at
               if vad_start_at is not None and self._acc_last_stop_at is not None else None)
        is_continuation = (not found and not self.awaiting_wakeword and bool(self._acc_text)
                           and gap is not None and gap <= CONCAT_MAX_GAP)

        if found:
            conf_str = f"{confidence:.1%}" if confidence is not None else "n/a"
            info(f"  [{self.label}] Wake word detected in transcript '{text}' "
                 f"(confidence={conf_str}).")
            self.awaiting_wakeword = False
            self._reset_accumulator()
            self._acc_had_wake_word = True
            piece = rest
        elif is_continuation:
            piece = text
            fine(f"[{self.label}] continuing utterance (+{gap:.2f}s gap): "
                 f"'{self._acc_text}' + '{text}'")
        elif self.awaiting_wakeword:
            fine(f"[{self.label}] no wake word while awaiting one — dropping: {shown}")
            return
        else:
            self._reset_accumulator()  # in a listen window, but the gap is too long → new command
            piece = text

        self._acc_text = f"{self._acc_text} {piece}".strip() if is_continuation else piece
        self._acc_last_stop_at = vad_stop_at
        query = clean_query(self._acc_text)

        if not query:
            if found:
                # Arm the listen window locally rather than relying on the server's
                # "listen" broadcast bouncing back — that broadcast no longer fires
                # (we don't wake other outposts), so without this a bare wake word
                # would leave us awaiting_wakeword=False with no false-positive timer.
                self._begin_listen_window()
                self._client.notify_wake_word(self.label)
                trace(f"[{self.label}] wake word with no command after it — nothing to send.")
                # A bare wake word arms a listen window that used to go UNVERIFIED — the
                # 2026-08-05 FP: a hallucinated "Michelle." opened 15s of listening in a
                # chattering room (constant background speech holds the 2.5s revert timer)
                # until a background phrase cleared the risk floor and became a query.
                # Verify the fire with Scribe exactly like a command fire; a veto un-arms
                # the window (whisper_query='' → there's no server request to kill).
                if self._scribe is not None and SCRIBE_VERIFY_WAKE_WORD:
                    fine(f"[{self.label}] scribe verifying bare wake-word fire.")
                    self._scribe.submit(
                        audio, self._scribe_lang_hint,
                        lambda text, conf, language, t=stt_ref:
                            self._on_scribe_transcript(text, conf, language, t,
                                                       verify_wake_word=True, whisper_query=""),
                        label=self.label)
            else:
                trace(f"[{self.label}] transcript was only separators — nothing to send.")
            return

        # Fold the wake-word signal into the query so the server's reset and this query
        # can't arrive out of order — but only on the utterance's first send; a
        # continuation must not re-reset the conversation the wake word just opened.
        send_wake_word = self._acc_had_wake_word and not self._acc_sent_wake_word
        # At high Whisper confidence skip the Scribe call entirely (see should_skip_scribe)
        # and tell the server (scribe_skipped) not to hold the reply at the verdict gate for
        # a verdict that is never coming.
        skip_scribe = should_skip_scribe(confidence, query)
        self._client.send_query(query, lang, confidence, self.label, wake_word=send_wake_word,
                                audio=audio, stt_ms=stt_ms, stt_wait_ms=stt_wait_ms,
                                stt_trigger=stt_trigger,
                                identify_future=identify_future, engine=ENGINE_WHISPER,
                                scribe_skipped=skip_scribe)
        self._acc_sent_wake_word = self._acc_sent_wake_word or send_wake_word
        if found:
            self._begin_listen_window()      # entered a listen session…
        self._note_followup_utterance()      # …and keep the listen window open

        # Hand the same audio to Scribe as a second opinion — but ONLY now that Whisper has
        # accepted this utterance, never on raw background speech (that's the cost win). On a
        # wake-word send (found) Scribe also VERIFIES the wake word: if Scribe doesn't hear it,
        # the request we just started is a Whisper false positive and gets killed. On a
        # follow-up (no wake word) Scribe is purely an extra accuracy vote — there's no wake
        # word to disagree about. whisper_query is the exact phrase the server is now running,
        # so a veto kills that request and nothing else (see _on_scribe_transcript).
        if self._scribe is not None and SCRIBE_VERIFY_WAKE_WORD:
            if skip_scribe:
                info(f"  [{self.label}] skipping scribe — Whisper confidence {confidence:.1%} "
                     f"≥ {SCRIBE_SKIP_CONFIDENCE_PCT:.0f}% skip floor: '{query}'")
                return
            if found:
                fine(f"[{self.label}] scribe verifying wake word for '{query}'.")
            self._scribe.submit(
                audio, self._scribe_lang_hint,
                lambda text, conf, language, t=stt_ref, vww=found, wq=query, wc=confidence:
                    self._on_scribe_transcript(text, conf, language, t,
                                               verify_wake_word=vww, whisper_query=wq,
                                               whisper_conf=wc),
                label=self.label)

    def _veto_false_positive(self, whisper_query, reason):
        """Scribe, verifying a Whisper wake-word fire, refused to confirm it (no wake word, a
        non-command language, or no speech at all) → treat Whisper's detection as a false
        positive: kill the in-flight server request for its phrase and drop this mic's
        spuriously-armed listen window. The kill is phrase-targeted server-side, so if the
        server already moved on to a different phrase nothing is killed. Resets only THIS mic's
        listen state, not its mute: if Floxy is already speaking, the server's kill +
        _restore_outposts_if_muted owns the unmute, and unmuting here would let the mic capture
        Floxy's own voice."""
        action = (f"killing request for '{whisper_query}'" if whisper_query
                  else "un-arming the bare wake-word listen window")
        info(f"  [{self.label}] scribe DISAGREED — {reason}; treating Whisper's detection as a "
             f"false positive → {action}.")
        if whisper_query:
            self._client.send_false_positive(whisper_query, self.label)
        self.awaiting_wakeword = True
        self._cancel_listen_timers()
        self._reset_accumulator()

    def _settle_scribe_gate(self, whisper_query, why):
        """Scribe finished verifying a Whisper wake-word fire with neither a vote nor a veto.
        The server holds the reply for Scribe's verdict (WebRequestHandler
        MAX_WAIT_FOR_SCRIBE_VERDICT) and only a /stream vote or a /false_positive veto releases
        it — so a silent exit here would cost the reply the FULL cap (~1.5s) and get logged
        server-side as a network hiccup. Every near-miss rescue sits below the Scribe-skip
        confidence floor, so all of them would pay it. A bare fire (no query) has no reply
        holding, nothing to settle."""
        if not whisper_query:
            trace(f"[{self.label}] scribe: {why} on a bare wake-word fire — no gate to settle.")
            return
        trace(f"[{self.label}] scribe: {why} — settling the verdict gate for '{whisper_query}'.")
        self._client.send_scribe_settled(whisper_query, self.label)

    def _on_scribe_rescue(self, text, conf, language, speech_ended_at,
                          whisper_query, audio, identify_future, vad_stop_at):
        """Scribe's decode of an utterance whose Whisper wake-word fire the risk gate
        dropped as rescueable (see gate_whisper_transcript). Unlike a verify there is NO
        in-flight server request: a refused rescue changes nothing — the fire stays
        dropped, exactly as before the rescue existed. On a confirmed rescue this performs
        the whole fire, late, with Scribe's command as the query (Scribe is the engine
        that was actually sure). Runs on the Scribe worker thread."""
        if self.muted:
            trace(f"[{self.label}] scribe rescue: dropping — muted while Floxy speaks: '{text}'")
            return
        verdict = gate_scribe_rescue(text, whisper_query, self._lang_detector, scribe_conf=conf)
        if verdict.veto:
            info(f"  [{self.label}] scribe did NOT rescue the risk-dropped fire for "
                 f"'{whisper_query}' — {verdict.veto}; staying dropped.")
            return
        if not self.awaiting_wakeword:
            # Another utterance opened a session while Scribe was decoding; a late rescue
            # would reset the conversation that fire just started.
            info(f"  [{self.label}] scribe rescue arrived after another fire opened a "
                 f"session — dropping the late rescue: '{text}'")
            return
        query = clean_query(verdict.rest)
        if not query:
            trace(f"[{self.label}] scribe rescue: nothing left after cleanup — staying dropped.")
            return
        detected = ISO639_3_TO_1.get(language, language) if language else language
        lang = self._resolve_lang(self._arbitrate_language(text, detected)) or self.default_lang
        calibrated = conf / SCRIBE_CONF_CALIBRATION
        info(f"  [{self.label}] scribe RESCUED the risk-dropped wake-word fire: '{query}' "
             f"(scribe {conf:.1%}, whisper heard '{whisper_query}').")
        self.awaiting_wakeword = False
        self._reset_accumulator()
        self._acc_had_wake_word = True
        self._acc_text = verdict.rest
        self._acc_last_stop_at = vad_stop_at
        stt_ms = (round((time.time() - speech_ended_at) * 1000)
                  if speech_ended_at is not None else None)
        # scribe_skipped: no further Scribe verdict is coming (Scribe already spoke), so
        # the server must not hold the reply at the verdict gate.
        self._client.send_query(query, lang, calibrated, self.label, wake_word=True,
                                audio=audio, stt_ms=stt_ms, identify_future=identify_future,
                                engine=ENGINE_SCRIBE, scribe_skipped=True)
        self._acc_sent_wake_word = True
        self._begin_listen_window()
        self._note_followup_utterance()

    # ── Scribe transcript gate (runs on a Scribe worker thread, parallel to Whisper) ──
    def _on_scribe_transcript(self, text, conf, language, speech_ended_at=None,
                              verify_wake_word=False, whisper_query=None, whisper_conf=None):
        """Gate a parallel Scribe transcript: post it as an extra vote, or — when it is
        verifying a Whisper wake-word send — veto a false positive.

        A sibling of _on_transcript: it applies the same mute / language / wake-word gates, but
        NEVER touches the cross-fragment accumulator and NEVER carries the wake-word reset (only
        Whisper's send owns that — Scribe is an additional vote). Confidence is de-scaled onto
        Whisper's range so the server's voting compares the two fairly. Runs off the Scribe
        executor, concurrent with Whisper's _on_transcript; it only reads self.awaiting_wakeword,
        a benign race.

        verify_wake_word is True only for the Scribe job fired on a Whisper wake-word send. Scribe
        is the more accurate engine, so if it does NOT confirm the wake word — heard no speech at
        all (empty), a non-command language, or simply a different phrase with no wake word — we
        treat Whisper's detection as a false positive and kill the in-flight server request for
        `whisper_query`. `whisper_conf` is that fire's Whisper confidence (disagreement veto).
        Follow-up votes (verify_wake_word=False) never veto."""
        if self.muted:
            trace(f"[{self.label}] scribe: dropping — muted while Floxy speaks: '{text}'")
            return
        if verify_wake_word:
            # The verify decision is pure (gate_scribe_verify) so the offline harness runs the
            # identical one. All vetoes sit BEFORE the stale-turn drop below — a veto must
            # fire even after the turn has closed.
            verdict = gate_scribe_verify(text, whisper_query, self._lang_detector,
                                         whisper_conf=whisper_conf, scribe_conf=conf)
            if verdict.veto:
                self._veto_false_positive(whisper_query, verdict.veto)
                return
            if verdict.echo:
                trace(f"[{self.label}] scribe: dropping priming-phrase echo: '{text}'")
                self._settle_scribe_gate(whisper_query, "priming echo, no vote")
                return
            if verdict.near_miss:
                # The fire stands, but Scribe misheard the opening ("Mission complete" for
                # "Michelle good night"), so its command text is not a trustworthy second
                # opinion — Whisper's phrase runs alone rather than risk a wrong vote winning.
                info(f"  [{self.label}] scribe heard a near-miss wake word '{verdict.near_miss}' "
                     f"in '{text}' — fire stands, no vote.")
                self._settle_scribe_gate(whisper_query, "near-miss wake word, no vote")
                return
            found, rest = verdict.found, verdict.rest
        else:
            # A follow-up vote never vetoes: an empty or echoed transcript is just skipped, and
            # the anchor applies as usual (a deep wake-word mention stays overheard).
            if not text:
                trace(f"[{self.label}] scribe: empty transcript, nothing to send.")
                return
            if is_priming_echo(text):
                trace(f"[{self.label}] scribe: dropping priming-phrase echo: '{text}'")
                return
            found, rest = strip_wake_word(text)

        # Scribe reports ISO-639-3 ('eng'); normalize to the 2-letter subtag the routing uses.
        # Route on it, but fall back to the outpost default rather than dropping: Scribe has no
        # restricted-detection re-pick like Whisper, so an unmapped code is more likely a label
        # quirk than genuinely foreign speech.
        detected = ISO639_3_TO_1.get(language, language) if language else language
        lang = self._resolve_lang(self._arbitrate_language(text, detected)) or self.default_lang

        # Scribe never opens a session — it's only a second opinion on a Whisper-accepted
        # utterance, and Whisper (which always runs first) owns the wake-word reset. So if we're
        # back to awaiting a wake word the turn has already closed, and this vote is stale: a late
        # cloud-STT return whose Whisper sibling either lost/was debounced or whose reply has since
        # finished (2026-06-25 'Michelle, can I look at your eyes?' — a wake-word VERIFY job that
        # came back 0.5s after the reply ended and got injected as a fresh query). Drop it even
        # when it carries a wake word: a genuinely fresh wake word arrives via Whisper, which
        # reopens the session before its Scribe job returns. A veto still fires (checked earlier).
        if self.awaiting_wakeword:
            fine(f"[{self.label}] scribe: vote arrived after the turn closed — dropping: '{text}'")
            return

        query = clean_query(rest if found else text)
        if not query:
            trace(f"[{self.label}] scribe: nothing left after wake-word/cleanup — nothing to send.")
            if verify_wake_word:
                self._settle_scribe_gate(whisper_query, "nothing left after cleanup, no vote")
            return

        # De-scale Scribe's confidence onto Whisper's range (see SCRIBE_CONF_CALIBRATION). Drop
        # below the server's reject floor: it not only ignores sub-threshold queries but kills any
        # running request matching the phrase as a false positive — a low-conf Scribe echo of
        # Whisper's running phrase must never reach the server.
        calibrated = conf / SCRIBE_CONF_CALIBRATION
        if calibrated < SCRIBE_MIN_CONFIDENCE:
            fine(f"[{self.label}] scribe: dropping low-confidence vote (conf={conf:.1%} → "
                 f"calibrated {calibrated:.1%} < {SCRIBE_MIN_CONFIDENCE:.0%}): '{query}'")
            if verify_wake_word:
                self._settle_scribe_gate(whisper_query, "low-confidence vote dropped")
            return

        stt_ms = round((time.time() - speech_ended_at) * 1000) if speech_ended_at is not None else None
        info(f"  [{self.label}] scribe vote: '{query}' (lang={lang}, "
             f"calibrated confidence={calibrated:.1%}).")
        # wake_word=False always (Whisper owns the reset); audio=None skips a duplicate speaker
        # identification — Whisper's send already populated the server's identity history.
        # verified_phrase lets the server settle the wake-word verdict gate for the utterance
        # this vote actually belongs to (whisper_query is the exact phrase the server ran).
        # A verify job that got this far heard the wake word (anything else vetoed above), so
        # its vote also tells the server the fire is confirmed — see /false_positive.
        self._client.send_query(query, lang, calibrated, self.label, wake_word=False,
                                audio=None, stt_ms=stt_ms, engine=ENGINE_SCRIBE,
                                verified_phrase=whisper_query or "",
                                wake_word_confirmed=verify_wake_word)
