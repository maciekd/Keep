from typing import Generator
from openai import OpenAI
from .AbstractLLM import AbstractLLM, ThinkingNotSupportedError, Thinking
from common.ConfigManager import config


class ChatGPTLLM(AbstractLLM):
    """Talks to OpenAI's chat API and serves as the base for every OpenAI-compatible backend.

    Subclasses (Grok, OpenRouter) only swap the API key name and base_url; message content is
    converted from the Anthropic block format and cache markers are stripped before sending.
    """

    def __init__(self, model: str = 'gpt-4.1', api_key_name: str = 'openai', base_url: str | None = None):
        self._api_key_name = api_key_name
        self.model = model
        self.client = None
        self.base_url = base_url

    @property
    def api_key(self) -> str | None:
        """Read at every use so a key stored at runtime reaches the next request (see GeminiLLM)."""
        return config.get_api_key(self._api_key_name)

    def api_key_name(self) -> str:
        return f"api_keys.{self._api_key_name}"

    def missing_api_key(self) -> str | None:
        return None if self.api_key else self.api_key_name()

    def name(self) -> str:
        return "GPT"

    def initialize(self):
        kwargs = {'api_key': self.api_key}
        if self.base_url:
            kwargs['base_url'] = self.base_url
        self.client = OpenAI(**kwargs)

    @staticmethod
    def _convert_anthropic_content(messages: list[dict]) -> list[dict]:
        """Convert Anthropic-format image content blocks to OpenAI format.

        When message content is a list (image blocks from Telegram), converts:
        - {"type": "image", "source": {"type": "base64", ...}} -> {"type": "image_url", ...}
        - {"type": "text", "text": T} -> unchanged (same format)

        Also strips Claude prompt-cache markers from the system message so they never
        reach an OpenAI-compatible backend (or any ChatGPTLLM subclass) as literal text.
        """
        converted = []
        for msg in messages:
            if isinstance(msg.get('content'), list):
                new_content = []
                for block in msg['content']:
                    if block.get('type') == 'image' and 'source' in block:
                        src = block['source']
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}
                        })
                    else:
                        new_content.append(block)
                converted.append({**msg, 'content': new_content})
            elif msg.get('role') == 'system':
                converted.append({**msg, 'content': ChatGPTLLM.strip_cache_markers(msg.get('content'))})
            else:
                converted.append(msg)
        return converted

    def create_streaming_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL,
                                    volatile_tail: int = 0, follow_up: bool = False) -> Generator[str, None, None]:
        if Thinking.of(thinking).full:
            raise ThinkingNotSupportedError(f"{self.name()} does not support native thinking mode")

        self._ensure_initialized()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_anthropic_content(messages),
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def create_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL, volatile_tail: int = 0,
                          follow_up: bool = False) -> str:
        if Thinking.of(thinking).full:
            raise ThinkingNotSupportedError(f"{self.name()} does not support native thinking mode")

        self._ensure_initialized()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_anthropic_content(messages),
            stream=False
        )

        return response.choices[0].message.content

    def create_completion_with_images(self, messages: list[dict], images: list, background: bool = False) -> str:
        self._ensure_initialized()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_anthropic_content(self._image_messages(messages, images)),
            stream=False
        )
        return response.choices[0].message.content

    @classmethod
    def _image_messages(cls, messages: list[dict], images: list) -> list[dict]:
        """`messages` with the images attached to the first user message as data-URL parts."""
        processed_messages = []
        for msg in messages:
            if msg['role'] == 'user' and images:
                content = [{"type": "text", "text": msg['content']}]
                for image in images:
                    content.append({"type": "image_url", "image_url": {"url": cls._process_image(image)}})
                processed_messages.append({"role": "user", "content": content})
                images = []
            else:
                processed_messages.append(msg)
        return processed_messages

    @staticmethod
    def _process_image(image) -> str:
        import base64

        if isinstance(image, (bytes, bytearray)):
            base64_image = base64.b64encode(image).decode('utf-8')
            if image.startswith(b'\x89PNG'):
                return f"data:image/png;base64,{base64_image}"
            else:
                return f"data:image/jpeg;base64,{base64_image}"

        if isinstance(image, str) and '/' not in image and '\\' not in image:
            try:
                base64.b64decode(image)
                return f"data:image/jpeg;base64,{image}"
            except Exception:
                pass

        if isinstance(image, str):
            with open(image, 'rb') as f:
                image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                if image.lower().endswith('.png'):
                    return f"data:image/png;base64,{base64_image}"
                else:
                    return f"data:image/jpeg;base64,{base64_image}"

        raise ValueError(f"Unsupported image format: {type(image)}")
