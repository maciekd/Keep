"""The photo-context briefing: the few hundred words of the memory bank a photo analyst needs, extracted once per bank version and sent with every photo instead of the whole bank."""
import hashlib
import json
import threading
import time
from pathlib import Path

from assistant.llm.AbstractLLM import AbstractLLM, LLMUnavailable
from common.Paths import CACHE_DIR
from common.logging import info, error, trace

NO_MEMORY = "(no memory available)"

EXTRACT_PROMPT = """You are preparing the briefing a photo analyst reads before describing a family's photo and video library. From the memory bank below, extract only what helps recognise who, what and where is in a photo or video:
- the household: every person by name, with relationship, approximate age or birth year, and anything known about their looks
- pets, by name and kind
- the home and the places the family frequents (summer house, grandparents' house, workplace, school, usual holiday spots), with town or country
- cars, boats, bikes and other recognisable belongings
- family facts that date or explain photos: birthdays, anniversaries, moves, renovations, recurring events

Leave out everything else: preferences, rules, contacts outside the family, tasks, how the assistant should behave, anything secret. Write short markdown bullet lists under headings, in the language of the bank, no more than about 400 words. Return only the briefing."""


class PhotoContext:
    """What the indexing prompt carries about the family: an extract of the bank, not the bank.

    The bank is six thousand tokens of preferences, rules and contacts of which a photo analyst
    needs a few hundred — the names, faces, places and pets. One LLM call per bank version makes
    the extract; it is kept in `cache/photo_context.json` under the bank's digest, so a restart
    costs nothing and the nightly curation costs one call. The extract is identical from photo to
    photo, which is what the provider's context cache keys on."""

    CACHE_PATH = CACHE_DIR / "photo_context.json"
    _lock = threading.Lock()
    _memo: tuple[str, str] | None = None    # (bank digest, extract) of the last answer

    @classmethod
    def briefing(cls, bank: str, llm: AbstractLLM) -> str:
        """The briefing for `bank`, from memory, the cache file or one extraction call. An empty
        bank needs no call. When the call fails the whole bank stands in for this photo and the
        extraction is tried again on the next one; the provider refusing every request
        (LLMUnavailable) is the caller's to handle, as for the photo itself."""
        if not bank.strip():
            return NO_MEMORY
        digest = hashlib.sha256(bank.encode()).hexdigest()
        with cls._lock:
            if cls._memo is not None and cls._memo[0] == digest:
                return cls._memo[1]
            extract = cls._read_cached(digest)
            if extract is None:
                extract = cls._extract(bank, digest, llm)
                if extract is None:
                    return bank
            cls._memo = (digest, extract)
            return extract

    @classmethod
    def _read_cached(cls, digest: str) -> str | None:
        path = Path(cls.CACHE_PATH)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            trace(f"Photos: the photo-context cache {path} is unreadable ({e}), extracting anew")
            return None
        if data.get("bank_sha256") != digest or not isinstance(data.get("extract"), str):
            trace("Photos: the memory bank changed since the photo context was extracted")
            return None
        return data["extract"]

    @classmethod
    def _extract(cls, bank: str, digest: str, llm: AbstractLLM) -> str | None:
        info(f"Photos: extracting the photo context from the memory bank ({len(bank)} chars)")
        t0 = time.time()
        try:
            extract = (llm.think(system_message=EXTRACT_PROMPT, user_message=bank) or "").strip()
        except LLMUnavailable:
            raise
        except Exception as e:
            error(f"Photos: could not extract the photo context ({e}) — this photo gets the whole bank")
            return None
        if not extract:
            error("Photos: the photo-context extraction came back empty — this photo gets the whole bank")
            return None
        info(f"Photos: photo context extracted in {time.time() - t0:.1f}s: {len(extract)} chars "
             f"from a bank of {len(bank)}")
        try:
            path = Path(cls.CACHE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"bank_sha256": digest, "extract": extract, "made_at": time.time()}, indent=1))
        except OSError as e:
            trace(f"Photos: could not write the photo-context cache: {e}")
        return extract
