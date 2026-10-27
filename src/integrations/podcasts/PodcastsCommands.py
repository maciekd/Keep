from assistant.commands.CommandSet import CommandSet
from integrations.podcasts.PodcastsIntegration import PodcastsIntegration


class PodcastsCommands(CommandSet):
    """The podcast reports: list, read, search, draft a LinkedIn post, regenerate."""
    MULTI_PARAM_COMMANDS = frozenset({"load_podcast_artifact", "podcast_draft", "podcast_regenerate"})

    @staticmethod
    def _podcasts() -> PodcastsIntegration:
        return PodcastsIntegration.get_instance()

    def cmd_podcast_episodes(self, parameter_string: str):
        text = (parameter_string or "").strip()
        if text and not text.isdigit():
            return "podcast_episodes failed: limit must be a number"
        self.background_command(lambda limit: self._podcasts().list_episodes(limit), int(text) if text else 5)

    def cmd_load_podcast_artifact(self, *args: str):
        if len(args) < 2:
            return "load_podcast_artifact failed: expected 'slug, kind' parameters"
        self.background_command(lambda p: self._podcasts().load_artifact(*p), (args[0].strip(), args[1].strip()))

    def cmd_podcast_draft(self, *args: str):
        if len(args) < 2 or not args[1].strip().isdigit():
            return "podcast_draft failed: expected 'slug, idea_no' parameters, idea_no a number"
        self.background_command(lambda p: self._podcasts().pipeline.draft(*p), (args[0].strip(), int(args[1].strip())))

    def cmd_podcast_search(self, parameter_string: str):
        if not (parameter_string or "").strip():
            return "podcast_search failed: expected a query"
        self.background_command(self._podcasts().search, parameter_string.strip())

    def cmd_podcast_regenerate(self, *args: str):
        if len(args) < 2:
            return "podcast_regenerate failed: expected 'slug, kind' parameters"
        self.background_command(lambda p: self._podcasts().pipeline.start_regenerate(*p), (args[0].strip(), args[1].strip()))

    def describe_podcast_episodes(self, p):          return "Listing the podcast reports"
    def describe_load_podcast_artifact(self, p):     return "Reading a podcast report"
    def describe_podcast_draft(self, p):             return "Drafting a LinkedIn post from a podcast"
    def describe_podcast_search(self, p):            return f"Searching the podcasts for {p.strip()}" if p.strip() else "Searching the podcasts"
    def describe_podcast_regenerate(self, p):        return "Regenerating a podcast report"
