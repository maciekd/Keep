"""Local microphone outposts.

Server-internal outposts: one per connected microphone. They capture audio,
transcribe it with a shared Whisper (MLX) instance, and push the resulting text
to the server's /stream endpoint just like a remote Android/iOS outpost would.
The existing content-based voting in SpeechAnalyzer then treats them as peers of
the remote outposts.

The package itself imports nothing: `constants` and `scribe_status` are read by the reply-timing
gate on every voice server, also one without microphones of its own (common/Platform), and must
not bring the outposts' machinery along. `bootstrap.start_local_outposts` is the way in.
"""
