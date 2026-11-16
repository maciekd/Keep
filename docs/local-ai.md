# Keep on 100% local AI

Keep can do all of its thinking on a model that runs on your own hardware. Nothing you say, and nothing Keep reads
on your behalf, is then sent to an AI provider.

The default install uses the cloud (Anthropic), because that is the most capable setup. Local is a supported
alternative, and one line of `config.yaml` switches between the two.

## Supported setups

| Model | Server | `local_backend` | RAM |
|---|---|---|---|
| Qwen 3.8 27B, with vision | [oMLX](https://github.com/jundot/omlx) with Splash | `openai` | 48 GB |
| DeepSeek V4.1 Flash 1017, with vision | [DwarfStar](https://github.com/antirez/ds4) (`ds4-server`) | `ds4` | 128 GB |

These two are the ones we test against and actively support. Others also work:

- **Ollama** (`local_backend: ollama`) — runs `qwen3.8:27b-mlx` by default.
- **Any OpenAI-compatible server** (`local_backend: openai`) — llama.cpp's server, for example, with the model of
  your choice. Expect to do more tuning yourself.

The model server can run on the same Mac as Keep or on another machine on your network.

## Setting it up

The easy way: install Keep with the default cloud setup and ask it to help you move to local. It knows its own
configuration, can flip the switch itself and will walk you through starting the model server.

By hand:

1. **Start the model server.**
   - oMLX: follow its readme and load the Qwen model. It listens on `localhost:8000` by default.
   - DwarfStar: clone and build [ds4](https://github.com/antirez/ds4) beside the Keep folder, download the model and
     its vision encoder, then run `scripts/local_ds4_start.sh`. The comments at the top of that script explain the
     flags; the disk cache it turns on matters a lot for speed.
   - Ollama: `scripts/local_ollama_start.sh`.
2. **Point Keep at it.** In `config.yaml`:
   ```yaml
   local_backend: openai    # or ds4, or ollama
   ```
   If the server is on another machine or port, set `base_url` in the matching section (`openai_compatible:`, `ds4:`
   or `ollama:`). `config.example.yaml` documents every key.
3. **Restart Keep.**

Set `local_backend: none` and restart to go back to the cloud.

## What "100% local" covers

With `local_backend` set, every AI job in Keep runs on your model: the chat, the background checks, the nightly
memory curation and the photo descriptions.

A few things still reach the internet, and each is yours to turn off:

- **Telemetry** is on by default and, for the first 7 days, includes your queries and Keep's replies. Set
  `telemetry.enabled: false` if you want nothing to leave. See the Privacy section of the [readme](../readme.md#privacy).
- **Voice** speaks through ElevenLabs by default, a cloud service. Supertonic TTS is the engine that runs on your
  machine.
- **The interfaces and integrations you enable** talk to their own services: Telegram, your mail provider, your car,
  web search. That is their job, but a Telegram chat with Keep does pass through Telegram.
- **Self-improvement** (the nightly code-quality run and the developer commands) uses the Claude Code CLI, which is
  a cloud tool. It only runs where it has been set up.

## What you give up

- **Video.** No local backend takes video, so clips in messages and mail are left undescribed. Photos are fine.
- **Capability and speed.** The cloud models are stronger and faster than anything that fits in 48 or 128 GB. A
  local Keep handles everyday requests well; long, multi-step errands are where you will notice the difference.
- **Memory headroom.** The RAM figures above are for the model. Keep itself is light, but anything else heavy on
  the same machine will compete with it.
