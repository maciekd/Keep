# Keep on 100% local AI

Keep can do all its thinking on a model running on your own hardware, so all private information stays on your machine. 

On a Mac that can carry it, the installer asks where Keep should think and sets the local model up itself: choose **Local, automatic setup**. On any other machine, or for a model or server of your own, start on Claude and ask Keep to move you to local AI. Do that first, before you connect anything, and nothing sensitive leaves your machine.

One line of `config.yaml`, `local_backend`, is the switch between the two.

## What the installer sets up

On a Mac with Apple Silicon, 32 GB of memory and 30 GB of free disk, the installer:

- installs [oMLX](https://github.com/jundot/omlx) through Homebrew and starts it at login
- downloads the model
- points Keep at it, with no Anthropic key

If the setup fails, the install goes on with Claude instead.

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