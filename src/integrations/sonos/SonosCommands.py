from assistant.commands.CommandSet import CommandSet
from common.logging import trace
from integrations.sonos.SonosIntegration import SonosIntegration


class SonosCommands(CommandSet):
    """The Sonos-only command left in the folder: a Sonos favorite or playlist by name. The music
    commands every speaker system answers (play, pause, play_music …) and the Spotify sign-in are
    core's, run against the MusicSystem contract."""
    MULTI_PARAM_COMMANDS = frozenset({"sonos_enqueue_and_play"})
    SELF_EVIDENT_COMMANDS = frozenset({"sonos_enqueue_and_play"})

    @property
    def sonos(self) -> SonosIntegration:
        return SonosIntegration.get_instance()

    #  todo: this is not in the prompt, do we still want it?
    def cmd_sonos_enqueue_and_play(self, *args: str):
        if len(args) < 2:
            return "sonos_enqueue_and_play failed: expected 'zone, playlist' parameters"
        room = args[0].strip()
        playlist = args[1].strip()
        if problem := self.sonos.zone_problem(room):
            return f"sonos_enqueue_and_play failed: {problem}"

        if playlist in self.sonos.get_favorites(room):
            trace("Sonos favorite, playing.")
            self.sonos.play_favorite(room, playlist)
        else:
            trace("Sonos playlist, playing.")
            self.sonos.play_playlist(room, playlist, shuffle=True)

    def describe_sonos_enqueue_and_play(self, p):
        parts = [x.strip() for x in p.split(",", 1)]
        if len(parts) == 2:
            return f"Playing {parts[1]} in {parts[0]}"
        return "Playing music"
