import threading
import time

from common.logging import info, error, trace, fine
from common.FloxyDatabase import FloxyDatabase
from ..llm import create_llm, parse_llm_config
from assistant.commands.Commands import Commands
from assistant.commands.formats.CommandFormats import CommandFormats

SYSTEM_PROMPT = """
You are a cache safety classifier for a voice assistant. You will be given a user query and the assistant's response. 
Your job is to determine if this response is SAFE or UNSAFE to serve from cache in the future for the same query.

Examples of UNSAFE responses (should NOT be cached):
- References time of day, current weather, dates, day of week
- Contains real-time data (stock prices, sports scores, news, references to current weather)
- Time-dependent phrases ("Good morning", "Good evening")
- References to previous conversation context ("as I mentioned", "like we discussed")
- If user references previous context (query: "Ok lets do it" response: "turn_off("Kitchen lights"), we do not know what the user said previously.
- Commands related to playing music can be dependent on context (such as when room the user is it), so for example "Pause the music" cannot be cached safely since we're not sure what room the user wants music paused in.
- Commands related to showing photos - these can get outdated over time

Examples of SAFE responses (should be cached):
- query: "Turn off the aircon in the bedroom.", response: "execute(action_name: "AC Off [Bedroom]") The AC in the bedroom is now off."
- query: "Turn off the kitchen lights.", response: "execute(action_name: "TurnOffKitchenLights") Got it, lights are off."
- query: "Turn off the TV in five minutes" - even though this references time it is safe in cache since the response to this query will always be the same, independent of time of day
- query: "User (VoiceToVoice in Bedroom): turn off the lights" - this is safe because it's clear that the user is in the Bedroom from context

"""

# Sonnet 5.5, Opus 5.5 and Fable refuse a request that asks for their reasoning
# (`reasoning_extraction`) and return nothing, so such a backend is asked for the verdict alone.
EXPLAINED_VERDICT = """Start your reply with a short explanation of your thinking.
Then ALWAYS end your reply with SAFE or UNSAFE on a single line."""
BARE_VERDICT = "ALWAYS end your reply with SAFE or UNSAFE on a single line."


def system_prompt(refuses_reasoning: bool) -> str:
    return SYSTEM_PROMPT + (BARE_VERDICT if refuses_reasoning else EXPLAINED_VERDICT)


class CacheAuditor:
    """Classifies cached LLM responses as safe or unsafe to replay and evicts the unsafe ones.

    Without the key its LLM tier needs (Claude's by default) the auditor is disabled: the server
    still starts, cache entries just stay unaudited until the key is configured."""
    def __init__(self):
        self._running = False
        self._thread = None
        llm_name, model = parse_llm_config('cache_auditor')
        self._llm = create_llm(llm_name, model)
        self._system_prompt = system_prompt(self._llm.refuses_reasoning_requests())
        self.disabled_reason = self._llm.missing_api_key()
        if self.disabled_reason:
            info(f"Cache auditor disabled: no {self.disabled_reason}")
        else:
            self._llm.initialize()
        # The launcher builds the auditor before the DialogManager picks the format from the
        # smart tier, so the parser is rebuilt when that choice lands.
        self._format = CommandFormats.active()(Commands.command_names())
        CommandFormats.on_change(self._on_command_format_changed)

    def _on_command_format_changed(self, fmt) -> None:
        self._format = fmt(Commands.command_names())
        trace(f"CacheAuditor reads cached replies in the {fmt.name} command format now")

    def start(self):
        if self.disabled_reason:
            trace(f"CacheAuditor not started: no {self.disabled_reason}")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="CacheAuditor", daemon=True)
        self._thread.start()
        info("Starting CacheAuditor...")

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            try:
                self._audit_pass()
            except Exception as e:
                error(f"CacheAuditor pass failed: {e}")
            for _ in range(60 * 15):
                if not self._running:
                    return
                time.sleep(1)

    def _has_commands(self, response: str) -> bool:
        """Check if a response contains any commands."""
        return len(self._format.extract_all(response)) > 0

    def _audit_pass(self):
        db = FloxyDatabase.get_instance()
        entries = db.get_unaudited_cache_entries(limit=1000)
        if not entries:
            trace("CacheAuditor: no unaudited entries! Nice.")
            return

        trace(f"CacheAuditor: auditing {len(entries)} entries...")
        flagged = 0
        for entry_id, query, response in entries:
            if not self._running:
                return
            try:
                if not self._has_commands(response):
                    db.mark_not_cachable(entry_id)
                    flagged += 1
                    continue

                user_message = f"Query: {query}\nResponse: {response}"
                llm_response = self._llm.think(self._system_prompt, user_message).strip()
                verdict = llm_response.split("\n")[-1].strip().upper()
                trace(f"CacheAuditor {verdict}: '{query}' - '{response.replace("\n", "")}'")
                res = llm_response.replace("UNSAFE", "").replace("SAFE", "").strip()
                fine(f"Reasoning: {res}")
                if verdict == "UNSAFE":
                    db.mark_not_cachable(entry_id)
                    flagged += 1
                else:
                    db.mark_audited(entry_id)
            except Exception as e:
                error(f"CacheAuditor: failed to audit entry {entry_id}: {e}")

        info(f"CacheAuditor: pass complete. Audited {len(entries)} entries, flagged {flagged} as unsuitable for cache.")
