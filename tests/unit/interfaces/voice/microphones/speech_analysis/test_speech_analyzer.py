import pytest
import time
import threading
from interfaces.voice.microphones.speech_analysis.SpeechAnalyzer import SpeechAnalyzer, Phrase


@pytest.mark.unit
class TestSpeechAnalyzer:
    """Tests for SpeechAnalyzer voice query analysis and deduplication"""

    @pytest.fixture
    def test_analyzer(self):
        """Create analyzer in test suite mode (bypasses voting logic)"""
        return SpeechAnalyzer(running_as_testsuite=True)

    @pytest.fixture(autouse=True)
    def _reset_flood_flag(self):
        # The flood flag is a class variable with a 3-second live window; a test
        # that sets it and leaks it poisons is_best_transcription for every later
        # test in the same worker process (seen as flaky test_routes.py failures).
        yield
        SpeechAnalyzer.false_positive_flood_time = None

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer instance (not in test suite mode)"""
        return SpeechAnalyzer(running_as_testsuite=False)

    # ===== normalize_query() tests =====

    def test_normalize_query_removes_punctuation(self, analyzer):
        """Should remove all punctuation"""
        assert analyzer.normalize_query("Hello, world!") == "hello world"
        assert analyzer.normalize_query("What's up?") == "what is up"
        assert analyzer.normalize_query("Turn on the TV.") == "turn on the tv"

    def test_normalize_query_expands_contractions(self, analyzer):
        """Should expand all contractions"""
        # Test common contractions
        assert analyzer.normalize_query("what's") == "what is"
        assert analyzer.normalize_query("it's") == "it is"
        assert analyzer.normalize_query("I'm") == "i am"
        assert analyzer.normalize_query("you're") == "you are"
        assert analyzer.normalize_query("they're") == "they are"
        assert analyzer.normalize_query("won't") == "will not"
        assert analyzer.normalize_query("can't") == "cannot"
        assert analyzer.normalize_query("don't") == "do not"

    def test_normalize_query_normalizes_whitespace(self, analyzer):
        """Should normalize multiple spaces to single space"""
        assert analyzer.normalize_query("turn  on   the    lights") == "turn on the lights"
        assert analyzer.normalize_query("  spaces  everywhere  ") == "spaces everywhere"

    def test_normalize_query_keeps_numbers(self, analyzer):
        """Should keep numbers in the query"""
        assert analyzer.normalize_query("dim lights to 50 percent") == "dim lights to 50 percent"
        assert analyzer.normalize_query("set volume to 25") == "set volume to 25"

    def test_normalize_query_keeps_non_ascii_letters(self, analyzer):
        """Should keep Swedish åäö (and other non-ASCII letters)"""
        assert analyzer.normalize_query("Tänd lampan i hörnet!") == "tänd lampan i hörnet"
        assert analyzer.normalize_query("Sänk värmen på övervåningen") == "sänk värmen på övervåningen"

    def test_normalize_query_empty_string(self, analyzer):
        """Should handle empty string"""
        assert analyzer.normalize_query("") == ""
        assert analyzer.normalize_query("   ") == ""

    # ===== add_phrase() tests =====

    def test_add_phrase_creates_new_phrase(self, analyzer):
        """Should create new phrase if it doesn't exist"""
        result = analyzer.add_phrase("turn on lights", "test")
        assert result == "turn on lights"
        assert "turn on lights" in analyzer.phrases
        assert analyzer.phrases["turn on lights"].count == 1

    def test_add_phrase_increments_count(self, analyzer):
        """Should increment count for existing phrase"""
        analyzer.add_phrase("hello", "")
        analyzer.add_phrase("hello", "")
        analyzer.add_phrase("hello", "")
        assert analyzer.phrases["hello"].count == 3

    # ===== score_for_phrase() tests =====

    def test_score_for_phrase_more_votes_is_better(self, analyzer):
        """More votes should increase score"""
        analyzer.add_phrase("one vote", "")
        analyzer.add_phrase("three votes", "")
        analyzer.add_phrase("three votes", "")
        analyzer.add_phrase("three votes", "")

        one_score = analyzer.score_for_phrase(analyzer.phrases["one vote"])
        three_score = analyzer.score_for_phrase(analyzer.phrases["three votes"])
        assert three_score > one_score

    def test_score_for_phrase_high_confidence_boost(self, analyzer):
        """Higher confidence should boost score"""
        analyzer.add_phrase("normal", "", 0.0)
        analyzer.add_phrase("high confidence", "", 0.9)

        normal_score = analyzer.score_for_phrase(analyzer.phrases["normal"])
        high_conf_score = analyzer.score_for_phrase(analyzer.phrases["high confidence"])
        assert high_conf_score > normal_score

    # ===== get_best_match() tests =====

    def test_get_best_match_returns_highest_score(self, analyzer):
        """Should return phrase with highest score"""
        analyzer.add_phrase("low score", "")
        time.sleep(0.02)
        analyzer.add_phrase("high score", "")
        analyzer.add_phrase("high score", "")
        analyzer.add_phrase("high score", "")

        best = analyzer.get_best_match()
        assert best == "high score"

    def test_longer_cumulative_phrase_beats_shorter_prefix_with_more_votes(self, analyzer):
        """The split-fragment regression: the prefix "what is the" arrives first (two
        mics → 2 votes), then the re-joined cumulative "what is the kitchen temperature"
        (1 vote). The longer, more recent, more complete phrase must win despite fewer
        votes — otherwise the server answers half the command. Length + recency outweigh
        the extra early vote (see score_for_phrase)."""
        analyzer.add_phrase("what is the", "", 0.50)
        analyzer.add_phrase("what is the", "", 0.50)                       # 2nd mic → 2 votes
        time.sleep(0.02)
        analyzer.add_phrase("what is the kitchen temperature", "", 0.50)   # cumulative, 1 vote
        assert analyzer.get_best_match() == "what is the kitchen temperature"

    def test_get_best_match_empty_phrases(self, analyzer):
        """Should return None when no phrases"""
        assert analyzer.get_best_match() is None

    # ===== is_best_transcription() tests =====

    def test_is_best_transcription_rejects_duplicates(self, analyzer):
        """Should reject phrase that was just accepted"""
        for _ in range(3):
            analyzer.is_best_transcription("hello", False, 0.5)

        analyzer.is_best_transcription("hello", False, 0.5)  # Accept it
        result = analyzer.is_best_transcription("hello", False, 0.5)  # Try again
        assert result is False  # Should reject duplicate

    def test_is_best_transcription_dedupes_whitespace_variant(self, analyzer):
        """A second mic transcribing 'Goodnight' must not re-fire the running 'Good night' reply"""
        analyzer.is_best_transcription("good night", False, 0.51, "ReSpeaker")  # accepted, now running
        assert analyzer.last_accepted_phrase == "good night"

        result = analyzer.is_best_transcription("goodnight", False, 0.55, "USB CODEC")
        assert result is False  # collapses to a vote, no second reply
        assert analyzer.num_votes("goodnight") >= 1

    def test_is_best_transcription_competing_phrase_reclaims_running_slot(self, analyzer):
        """A later corroborating vote for an earlier-accepted phrase must be able to
        reclaim the running slot from a wrong phrase that narrowly won it.

        Reproduces a production mix-up ('Lost the outdoor temperature.'): mic #2 got the
        correct 'What's the outdoor temperature?' first, then mic #1's garbled 'Lost the outdoor
        temperature.' narrowly out-scored it and took over the request. The scribe/other-mic
        votes that then confirmed the correct phrase were being swallowed by the
        already-accepted guard, so the wrong phrase could never be dethroned."""
        # Correct phrase wins first and starts running.
        assert analyzer.is_best_transcription("what's the outdoor temperature", False, 0.523, "CODEC2", "whisper") is True
        assert analyzer.last_accepted_phrase == "what is the outdoor temperature"

        # Garbled phrase arrives newer + higher confidence and wins the slot. (In prod it
        # won on a razor-thin margin helped by an age penalty on the older correct phrase;
        # here a clear confidence edge flips it deterministically without a slow sleep.)
        time.sleep(0.02)
        assert analyzer.is_best_transcription("lost the outdoor temperature", False, 0.60, "CODEC1", "whisper") is True
        assert analyzer.last_accepted_phrase == "lost the outdoor temperature"

        # A corroborating vote for the correct phrase (2nd mic/engine) must now reclaim it.
        time.sleep(0.02)
        result = analyzer.is_best_transcription("what's the outdoor temperature", False, 0.55, "CODEC2scribe", "scribe")
        assert result is True
        assert analyzer.last_accepted_phrase == "what is the outdoor temperature"

    def test_is_best_transcription_records_source_mic(self, analyzer):
        """The originating mic should be stored on the candidate phrase for the Home tab"""
        analyzer.is_best_transcription("turn on lights", False, 0.5, "Studio Display Microphone")
        normalized = analyzer.normalize_query("turn on lights")
        assert analyzer.phrases[normalized].source == "Studio Display Microphone"

    def test_is_best_transcription_rejects_empty_query(self, analyzer):
        """Should reject empty normalized query"""
        result = analyzer.is_best_transcription("", False, 0.5)
        assert result is False

    def test_is_best_transcription_testsuite_mode_always_accepts(self, test_analyzer):
        """Test suite mode should always return True"""
        result = test_analyzer.is_best_transcription("anything", False, 0.0)
        assert result is True

    # ===== False Positive Flood Detection =====

    def test_false_positive_flood_blocks_queries(self, analyzer):
        """Should block queries for 3 seconds after flood notification"""
        # Trigger flood
        SpeechAnalyzer.notify_false_positive_flood()

        # Should reject queries during flood window
        result = analyzer.is_best_transcription("blocked query", False, 0.9)
        assert result is False

    # ===== Helper Methods =====

    def test_latest_phrase_returns_newest(self, analyzer):
        """Should return most recently added phrase"""
        analyzer.add_phrase("first", "")
        time.sleep(0.01)
        analyzer.add_phrase("second", "")
        time.sleep(0.01)
        analyzer.add_phrase("third", "")

        latest = analyzer.latest_phrase()
        assert latest.phrase == "third"

    def test_latest_phrase_empty_phrases(self, analyzer):
        """Should return None when no phrases"""
        assert analyzer.latest_phrase() is None

    def test_longest_phrase_returns_longest(self, analyzer):
        """Should return longest phrase"""
        analyzer.add_phrase("short", "")
        analyzer.add_phrase("this is the longest phrase here", "")
        analyzer.add_phrase("medium length", "")

        longest = analyzer.longest_phrase()
        assert longest.phrase == "this is the longest phrase here"

    def test_longest_phrase_empty_phrases(self, analyzer):
        """Should return None when no phrases"""
        assert analyzer.longest_phrase() is None

    def test_length_component_for_phrase(self, analyzer):
        """Should calculate length as ratio to longest phrase"""
        analyzer.add_phrase("short", "")
        analyzer.add_phrase("this is much longer", "")

        short_phrase = analyzer.phrases["short"]
        long_phrase = analyzer.phrases["this is much longer"]

        short_component = analyzer.length_component_for_phrase(short_phrase)
        long_component = analyzer.length_component_for_phrase(long_phrase)

        assert long_component == 1.0  # Longest phrase
        assert short_component < 1.0  # Shorter phrase
        assert short_component > 0.0

    def test_num_votes_empty_query(self, analyzer):
        """Should return 0 for empty query"""
        assert analyzer.num_votes("") == 0

    # ===== clean() =====

    def test_clean_resets_all_state(self, analyzer):
        """Should clear all phrases and state"""
        analyzer.add_phrase("test1", "", 0.9)
        analyzer.add_phrase("test2", "")
        analyzer.last_accepted_phrase = "test1"

        analyzer.clean()

        assert len(analyzer.phrases) == 0
        assert len(analyzer.accepted_phrases) == 0
        assert analyzer.last_accepted_phrase == ""

    def test_clean_preserves_display_snapshot(self, analyzer):
        """clean() should snapshot the utterance so display state survives the wipe"""
        analyzer.add_phrase("turn on the lights", "kitchen", 0.9)
        analyzer._add_accepted_phrase("turn on the lights", "kitchen")

        analyzer.clean()

        phrases, accepted, last_accepted, scores = analyzer.get_display_state()
        assert "turn on the lights" in phrases
        assert "turn on the lights" in accepted
        assert last_accepted == "turn on the lights"
        assert "turn on the lights" in scores

    def test_display_state_prefers_live_over_snapshot(self, analyzer):
        """A new utterance should replace the snapshot in display state"""
        analyzer.add_phrase("old phrase", "", 0.9)
        analyzer.clean()

        analyzer.add_phrase("new phrase", "", 0.8)
        phrases, _, _, scores = analyzer.get_display_state()
        assert "new phrase" in phrases
        assert "old phrase" not in phrases
        assert "new phrase" in scores

    def test_clean_bumps_utterance_generation(self, analyzer):
        """clean() marks the utterance boundary; per-utterance consumers (console STT
        timing) rely on the generation changing so later mics can't overwrite the
        first mic's value within the same utterance"""
        gen = analyzer.utterance_generation
        analyzer.clean()
        assert analyzer.utterance_generation == gen + 1
        analyzer.clean()
        assert analyzer.utterance_generation == gen + 2

    def test_display_state_empty_side_keeps_previous_snapshot(self, analyzer):
        """A live side that's still empty must not clobber the snapshot of the previous one"""
        analyzer.add_phrase("first utterance", "", 0.9)
        analyzer._add_accepted_phrase("first utterance", "")
        analyzer.clean()

        # New utterance has live phrases but nothing accepted yet
        analyzer.add_phrase("second utterance", "", 0.5)
        phrases, accepted, last_accepted, _ = analyzer.get_display_state()
        assert "second utterance" in phrases
        assert "first utterance" in accepted
        assert last_accepted == "first utterance"

    # ===== Thread Safety =====

    def test_thread_safe_add_phrase(self, analyzer):
        """Should handle concurrent phrase additions safely"""
        def add_phrases():
            for _ in range(100):
                analyzer.add_phrase("concurrent test", "")

        threads = [threading.Thread(target=add_phrases) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 500 total votes
        assert analyzer.phrases["concurrent test"].count == 500

    # ===== Phrase Class Tests =====

    def test_phrase_update(self):
        """Should update count and source"""
        phrase = Phrase("test", "source1")
        old_time = phrase.timestamp
        time.sleep(0.01)
        phrase.update("source2")

        assert phrase.count == 2
        assert phrase.source == "source2"
        assert phrase.timestamp > old_time

    def test_best_mic_is_highest_confidence_not_most_recent(self):
        """A low-confidence straggler mic must not displace the mic that actually
        heard the user clearly (the bug the HomeTab [best_mic +N] display fixed)"""
        phrase = Phrase("turn on the lights", "ReSpeaker", 0.92)
        phrase.update("Studio Display", 0.55)

        assert phrase.source == "Studio Display"   # most recent, kept for compat
        assert phrase.best_mic == "ReSpeaker"      # but the display shows the best one

    def test_best_mic_tracks_per_mic_peak_confidence(self):
        """A weaker repeat from the same mic must not lower its recorded peak"""
        phrase = Phrase("turn on the lights", "ReSpeaker", 0.9)
        phrase.update("Studio Display", 0.7)
        phrase.update("ReSpeaker", 0.3)

        assert phrase.source_confidence["ReSpeaker"] == 0.9
        assert phrase.best_mic == "ReSpeaker"

    def test_mic_count_counts_distinct_mics(self):
        """Repeats from the same mic must not inflate the +N count"""
        phrase = Phrase("turn on the lights", "ReSpeaker", 0.8)
        phrase.update("ReSpeaker", 0.8)
        assert phrase.mic_count == 1

        phrase.update("Studio Display", 0.6)
        phrase.update("MacBook Pro", 0.7)
        assert phrase.mic_count == 3

    def test_empty_source_not_counted_as_mic(self):
        """Sourceless votes (e.g. testsuite) must not pollute the per-mic stats"""
        phrase = Phrase("hello", "", 0.5)
        assert phrase.mic_count == 0
        assert phrase.best_mic == ""               # falls back to source

        phrase.update("", 0.9)
        assert phrase.mic_count == 0

        phrase.update("ReSpeaker", 0.4)
        assert phrase.mic_count == 1
        assert phrase.best_mic == "ReSpeaker"

    def test_best_engine_follows_best_mic(self):
        """best_engine reports the STT engine behind the winning (best_mic) vote"""
        phrase = Phrase("turn on the lights", "ReSpeaker", 0.6, "whisper")
        phrase.update("ReSpeaker", 0.92, "scribe")   # higher conf, same mic
        phrase.update("Studio Display", 0.55, "whisper")

        assert phrase.best_mic == "ReSpeaker"
        assert phrase.best_engine == "scribe"        # the engine of the peak vote

    def test_best_engine_tracks_winning_mic_engine(self):
        """When the winning mic comes from a different engine, best_engine follows it"""
        phrase = Phrase("play music", "ReSpeaker", 0.5, "whisper")
        phrase.update("Studio Display", 0.9, "scribe")

        assert phrase.best_mic == "Studio Display"
        assert phrase.best_engine == "scribe"

    def test_best_engine_empty_when_untagged(self):
        """Votes with no engine tag (e.g. phone outposts) report an empty engine"""
        phrase = Phrase("hello", "ReSpeaker", 0.8)
        assert phrase.best_engine == ""

    def test_add_phrase_records_engine(self):
        """add_phrase threads the engine into the per-source engine map"""
        analyzer = SpeechAnalyzer(running_as_testsuite=False)
        analyzer.add_phrase("turn on the lights", "kitchen", 0.9, "scribe")
        assert analyzer.phrases["turn on the lights"].best_engine == "scribe"

    # ===== Scribe settle tracking (wake-word verdict gate) =====

    def test_scribe_not_settled_for_fresh_utterance(self, analyzer):
        """A new utterance window starts un-settled, so a Whisper wake-word reply
        holding at the gate actually waits for Scribe's verdict"""
        assert not analyzer.scribe_has_settled(analyzer.utterance_generation)

    def test_mark_scribe_settled_settles_current_generation(self, analyzer):
        """A Scribe result whose Whisper phrase matches the current utterance window
        settles that window's generation and returns it"""
        analyzer.add_phrase("turn on the lights", "kitchen", 0.9, "whisper")
        gen = analyzer.mark_scribe_settled("turn on the lights")
        assert gen == analyzer.utterance_generation
        assert analyzer.scribe_has_settled(gen)

    def test_stale_scribe_result_does_not_settle(self, analyzer):
        """A Scribe result for a PREVIOUS utterance (its phrase was wiped by clean(),
        i.e. back-to-back wake words inside Scribe's cloud round-trip) must not settle
        the new utterance's gate — Scribe hasn't weighed in on that one yet"""
        analyzer.add_phrase("turn on the lights", "kitchen", 0.9, "whisper")
        analyzer.clean()
        analyzer.add_phrase("what time is it", "kitchen", 0.9, "whisper")
        assert analyzer.mark_scribe_settled("turn on the lights") is None
        assert not analyzer.scribe_has_settled(analyzer.utterance_generation)

    def test_unknown_phrase_does_not_settle(self, analyzer):
        """A phrase that never voted in the current window settles nothing"""
        assert analyzer.mark_scribe_settled("never heard this") is None
        assert not analyzer.scribe_has_settled(analyzer.utterance_generation)

    def test_clean_opens_new_unsettled_window(self, analyzer):
        """clean() starts a new utterance: the previous utterance's Scribe verdict
        must not release a reply belonging to the NEXT one — that's exactly the
        stale-verdict race the generation correlation exists to prevent"""
        analyzer.add_phrase("turn on the lights", "kitchen", 0.9, "whisper")
        analyzer.mark_scribe_settled("turn on the lights")
        analyzer.clean()
        assert not analyzer.scribe_has_settled(analyzer.utterance_generation)

    def test_scribe_settled_generation_is_monotonic(self, analyzer):
        """A verdict for a later utterance satisfies waits on earlier generations,
        so a reply from a turn the server has moved past never waits out the cap"""
        old_gen = analyzer.utterance_generation
        analyzer.clean()
        analyzer.add_phrase("what time is it", "kitchen", 0.9, "whisper")
        analyzer.mark_scribe_settled("what time is it")
        assert analyzer.scribe_has_settled(old_gen)

    # ===== Scribe wake-word confirmation (what outweighs another mic's veto) =====

    def test_a_confirmation_covers_only_the_phrase_it_was_heard_on(self, analyzer):
        """Mics whose Whisper transcripts disagree are each hallucinating over background
        speech: one Scribe hearing the wake word on its phrase says nothing about the other
        mic's fire, so the veto on that one must still count"""
        analyzer.add_phrase("read a bedtime story", "hall", 0.9, "whisper")
        analyzer.add_phrase("defrost the car", "kitchen", 0.9, "whisper")
        assert analyzer.mark_wake_word_confirmed("read a bedtime story", "hall")
        assert analyzer.wake_word_confirmed_by("read a bedtime story") == "hall"
        assert analyzer.wake_word_confirmed_by("defrost the car") is None

    def test_a_confirmation_is_matched_on_the_normalized_phrase(self, analyzer):
        """The Scribe job and the vetoing mic each carry their own punctuation and case of the
        same Whisper phrase; the match is on the words"""
        analyzer.add_phrase("defrost the car", "kitchen", 0.9, "whisper")
        assert analyzer.mark_wake_word_confirmed("Defrost the car.", "kitchen")
        assert analyzer.wake_word_confirmed_by("defrost the car!") == "kitchen"

    def test_a_confirmation_from_an_unlabelled_mic_still_counts(self, analyzer):
        """The caller tells a confirmation from none by `is not None`, so an outpost that sent
        no label must not look like no confirmation"""
        analyzer.add_phrase("defrost the car", "", 0.9, "whisper")
        assert analyzer.mark_wake_word_confirmed("defrost the car", "")
        assert analyzer.wake_word_confirmed_by("defrost the car") == ""

    def test_a_confirmation_for_a_phrase_outside_the_window_is_not_recorded(self, analyzer):
        analyzer.add_phrase("defrost the car", "kitchen", 0.9, "whisper")
        assert not analyzer.mark_wake_word_confirmed("open the garage", "hall")
        assert analyzer.wake_word_confirmed_by("open the garage") is None

    def test_clean_forgets_the_confirmations_of_the_previous_utterance(self, analyzer):
        """The same words said again in the next utterance are a new fire; last time's Scribe
        verdict says nothing about it"""
        analyzer.add_phrase("defrost the car", "kitchen", 0.9, "whisper")
        analyzer.mark_wake_word_confirmed("defrost the car", "kitchen")
        analyzer.clean()
        analyzer.add_phrase("defrost the car", "kitchen", 0.9, "whisper")
        assert analyzer.wake_word_confirmed_by("defrost the car") is None
