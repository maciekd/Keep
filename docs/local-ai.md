# Keep on 100% local AI

Keep can do all its thinking on a model running on your own hardware, so all private information stays on your machine. 

The default install uses the cloud (Anthropic); but one line of `config.yaml` switches to local. It's recommended to start with Anthropic and have Keep help you configure local AI. The first thing you'll do with keep is change to local AI, so nothing sensitive will leave your machine. 

## Supported setups
| Model | Server | Local Backend | Min RAM |
|---|---|---------------|---------|
| Qwen 3.8 27B, with vision | [oMLX](https://github.com/jundot/omlx) with Splash | `openai`      | 48 GB   |
| DeepSeek V4.1 Flash 1017, with vision | [DwarfStar](https://github.com/antirez/ds4) | `ds4`         | 128 GB  |

These two are tested and supported. 

Ollama (`ollama`) and any OpenAI-compatible server such as llama.cpp (`openai`) work too. The model server can run on Keep's server machine or elsewhere on your network.

## Setting it up

Easiest: ask Keep to move you to local — it flips the switch and walks you through the rest. By hand:

1. **Start the model server** — oMLX per its readme (`localhost:8000`), `scripts/local_ds4_start.sh` for DwarfStar (see its comments), or `scripts/local_ollama_start.sh`.
2. **Set `local_backend`** in `config.yaml` (`openai`, `ds4` or `ollama`), plus `base_url` in the matching section
   if the server is elsewhere (see [config.example.yaml](https://github.com/maciekd/Keep/blob/master/config.example.yaml)).
3. **Restart Keep.** `local_backend: none` goes back to the cloud.
