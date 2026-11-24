# Keep on 100% local AI

Keep can do all its thinking on a model running on your own hardware, so all private information stays on your machine. 

On a Mac that can carry it, the installer asks where Keep should think and sets the local model up itself: choose **Local, automatic setup**. On any other machine, or for a model or server of your own, start with the default settings (Anthropic) and have Keep help you configure local AI. The first thing you'll do with Keep is change to local AI, so nothing sensitive will leave your machine.

One line of `config.yaml`, `local_backend`, is the switch between the two.

## What the installer sets up

Offered on a Mac with Apple Silicon, macOS 15 or newer, 32 GB of memory and 30 GB of free disk:

- [oMLX](https://github.com/jundot/omlx) through Homebrew (Homebrew itself where it is missing), in the version Keep was tested with and pinned there (`brew pin`). An oMLX you already have is kept as it is.
- The model, [Jundot/Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp), an 18 GB download into `~/.omlx/models/`.
- oMLX as a service that starts at login (`brew services`); `start.sh` and `stop.sh` leave it alone. Stop it with `brew services stop jundot/omlx/omlx`.
- `local_backend: openai` and the model's name in `config.yaml`, and no Anthropic key.

`KEEP_AI=local` in front of `bash` in the install command chooses it without the question.

**Memory.** oMLX refuses to load a model, and aborts a request, above a ceiling it works out from what the other apps hold. The model with a long prompt takes about 21 GB, so on a Mac with less than 36 GB the installer sets a fixed ceiling of 22 GB (`memory.memory_guard_tier: custom` in `~/.omlx/settings.json`): macOS then compresses the other apps instead of a turn failing. On an M1 Max with 32 GB that gives about 16 tokens a second, and 75 a second reading the prompt.

To go back to Claude: `local_backend: none`, and a key as `api_keys.anthropic`.

## Supported setups
| Model | Server                                                                                          | Local Backend | Min RAM |
|---|-------------------------------------------------------------------------------------------------|---------------|---------|
| Qwen 3.8 27B, with vision | [oMLX](https://github.com/jundot/omlx)                                                          | `openai`      | 32 GB   |
| DeepSeek V4.1 Flash 1017, with vision | [DwarfStar](https://github.com/antirez/ds4), [our fork](#dwarfstar-our-fork)                    | `ds4`         | 128 GB  |

These two are tested and supported.

Ollama (`ollama`) and any OpenAI-compatible server such as llama.cpp (`openai`) work too. The model server can run on Keep's server machine or elsewhere on your network.

## DwarfStar: our fork

Run DwarfStar from the `keep` branch of [maciekd/ds4](https://github.com/maciekd/ds4/tree/keep). This branch contains a number of performance optimizations that make normal requests go from ~16 s to  ~0.4 s on an M5 Ultra. `scripts/local_ds4_start.sh` needs this branch — upstream lacks its `--kv-cache-cold-anchor` flag. Clone it beside Keep, build, download the model and start it:

```bash
git clone -b keep https://github.com/maciekd/ds4 ../ds4 && cd ../ds4 && make
./download_model.sh ds41f-q2 && ./download_model.sh ds41f-vision
cd - && scripts/local_ds4_start.sh            # or: scripts/local_ds4_start.sh gguf/<another model>.gguf
```

## Setting it up

Easiest: ask Keep to move you to local — it flips the switch and walks you through the rest. 

By hand:

1. **Start the model server** — oMLX (`brew services start jundot/omlx/omlx`, `localhost:8000`, with the model in `~/.omlx/models/`), `scripts/local_ds4_start.sh` for DwarfStar ([our fork](#dwarfstar-our-fork)), or Ollama (`ollama serve`).
2. **Set `local_backend`** in `config.yaml` (`openai`, `ds4` or `ollama`), plus `base_url` in the matching section
   if the server is elsewhere, and for oMLX `openai_compatible.model` (the folder's name under `~/.omlx/models/`) (see [config.example.yaml](../config.example.yaml)).
3. **Restart Keep.** `local_backend: none` goes back to the cloud.

## Telemetry
Note that **Telemetry** is on by default and sends usage stats to our servers (see [Privacy](../readme.md#privacy)), set `telemetry.enabled: false` to disable.