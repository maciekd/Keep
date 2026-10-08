from assistant.commands.CommandSet import CommandSet
from assistant.SystemNote import system_note
from common.logging import info, trace
from integrations.music.MusicIntegration import MusicManager

INTEGRATION = "Sonos & Spotify"
NOTE_SOURCE = "Spotify"


class MusicCommands(CommandSet):
    """The music commands: playback and volume in a Sonos zone, Spotify search, playlists and
    favorites, and the sign-in of the user's Spotify account (spotify_login / spotify_login_code)."""
    MULTI_PARAM_COMMANDS = frozenset({"play_music", "sonos_enqueue_and_play"})

    # Built by the first music command when Sonos & Spotify is disabled — never at startup,
    # where it would start discovering speakers on a network nobody asked it to look at.
    @staticmethod
    def _get_music_manager() -> MusicManager:
        return MusicManager.get_instance()

    @property
    def music_manager(self) -> MusicManager:
        return self._get_music_manager()

    def cmd_search_music(self, parameter_string: str):
        self.background_command(self.music_manager.search_spotify, parameter_string)

    def cmd_play_music(self, *args: str):
        if len(args) < 2:
            return "play_music failed: expected 'zone, id' parameters"
        room = args[0].strip()
        id = args[1].strip()
        self.music_manager.play_in_zone(id, room)

    def cmd_play(self, parameter_string: str):
        self.music_manager.play(parameter_string)

    def cmd_pause(self, parameter_string: str):
        self.music_manager.pause(parameter_string)

    def cmd_next(self, parameter_string: str):
        self.music_manager.next_track(parameter_string)

    def cmd_volume_up(self, parameter_string: str):
        self.music_manager.adjust_volume(parameter_string, 4)

    def cmd_volume_down(self, parameter_string: str):
        self.music_manager.adjust_volume(parameter_string, -4)

    #  todo: this is not in the prompt, do we still want it?
    def cmd_sonos_enqueue_and_play(self, *args: str):
        if len(args) < 2:
            return "sonos_enqueue_and_play failed: expected 'zone, playlist' parameters"
        room = args[0].strip()
        playlist = args[1].strip()

        favorites = self.music_manager.get_favorites(room)
        if playlist in favorites:
            trace("Sonos favorite, playing.")
            self.music_manager.play_favorite(room, playlist)
        else:
            trace("Sonos playlist, playing.")
            self.music_manager.play_playlist(room, playlist, shuffle=True)

    # --- the user's Spotify sign-in: the URL out, the address the browser lands on back in ---

    def cmd_spotify_login(self, parameter_string: str):
        """Starts the flow: the sign-in URL and what to do with it come back as the command's
        result, which the LLM relays; when no chat asked (voice), a system note puts the link
        in the chats too, since a link cannot be heard."""
        self.background_command(self._run_spotify_login, (self.commands.asked_from_chat(),), lane=self.commands.CONFIG_LANE)

    def cmd_spotify_login_code(self, parameter_string: str):
        """Finishes the flow with the whole address the browser landed on; the outcome is the
        command's result, and a note to the chats only when no chat asked."""
        pasted = parameter_string.strip().strip("\"'`").strip()
        if not pasted:
            return 'spotify_login_code failed: paste the address the browser landed on, e.g. spotify_login_code("http://127.0.0.1:8888/callback?code=...&state=...")'
        self.background_command(self._run_spotify_login_code, (pasted, self.commands.asked_from_chat()), lane=self.commands.CONFIG_LANE)

    def _integration(self, command: str):
        """(integration, "") when Sonos & Spotify can sign a user in, else (None, the reason)."""
        monitor = self.commands.integration_monitor()
        integration = monitor.find(INTEGRATION)
        if integration is None or not hasattr(integration, "login_begin"):
            state = getattr(integration, "state", None)
            label = self.commands.state_label(getattr(state, "value", state), "not available")
            info(f"{command}: {INTEGRATION} is {label}, no sign-in possible")
            if label == "disabled":
                return None, f'{INTEGRATION} is disabled — enable_integration("{INTEGRATION}") first.'
            return None, f"{INTEGRATION} is {label} — set api_keys.spotify.client_id and client_secret and enable it first."
        return integration, ""

    def _run_spotify_login(self, from_chat: bool = False) -> str:
        integration, problem = self._integration("spotify_login")
        if integration is None:
            return problem
        try:
            sign_in = integration.login_begin()
        except Exception as e:   # no client id or secret
            info(f"spotify_login: {e}")
            return f"spotify_login failed: {e}"
        text = (f"Spotify sign-in: {sign_in.instructions} into the chat as "
                f'spotify_login_code("<the pasted address>") (quote it). The link is valid for about ten minutes.')
        if from_chat:
            trace("spotify_login: a chat asked, the reply carries the link — no note pushed")
        else:
            system_note(NOTE_SOURCE, text)
        return text

    def _run_spotify_login_code(self, pasted: str, from_chat: bool = False) -> str:
        integration, problem = self._integration("spotify_login_code")
        if integration is None:
            return problem
        try:
            failure = integration.login_finish(pasted)
        except Exception as e:
            failure = f"{type(e).__name__}: {e}"
        if failure:
            info(f"spotify_login_code: not signed in — {failure}")
            return f"Spotify sign-in did not complete: {failure}"
        user = integration.spotify_user_name()
        text = "Spotify: signed in" + (f" as {user}" if user else "") + ". "
        problem = integration.spotify_library_error()
        if problem:
            info(f"spotify_login_code: signed in, but the playlist library did not load — {problem}")
            text += (f"The playlist library could not be loaded yet ({problem}); it is tried again every "
                     f"{integration.REFRESH_INTERVAL_SECONDS // 60} minutes, and the S- ids appear in "
                     "SONOS PLAYLISTS AND ZONES once it has.")
        else:
            listed, total = len(integration.get_spotify_playlists()), integration.spotify_playlists_total()
            text += (f"{listed} of the account's {total} playlists appear in SONOS PLAYLISTS AND ZONES (ids S-0, "
                     "S-1, …) within a minute — read them from the section before naming one — and can be "
                     "started with play_music.")
        if from_chat:
            trace("spotify_login_code: a chat asked, the reply says it — no note pushed")
        else:
            system_note(NOTE_SOURCE, text)
        return text

    def describe_spotify_login(self, p):          return "Signing the Spotify account in"
    def describe_spotify_login_code(self, p):     return "Finishing the Spotify sign-in"

    def describe_search_music(self, p):           return f"Searching music for {p.strip()}"
    def describe_play(self, p):                   return f"Playing in {p.strip()}" if p.strip() else "Playing"
    def describe_pause(self, p):                  return f"Pausing in {p.strip()}" if p.strip() else "Pausing"
    def describe_next(self, p):                   return f"Skipping to next track in {p.strip()}" if p.strip() else "Skipping to next track"
    def describe_volume_up(self, p):              return f"Turning up the volume in {p.strip()}" if p.strip() else "Turning up the volume"
    def describe_volume_down(self, p):            return f"Turning down the volume in {p.strip()}" if p.strip() else "Turning down the volume"

    def describe_play_music(self, p):
        room = p.split(",", 1)[0].strip()
        return f"Playing music in {room}" if room else "Playing music"

    def describe_sonos_enqueue_and_play(self, p):
        parts = [x.strip() for x in p.split(",", 1)]
        if len(parts) == 2:
            return f"Playing {parts[1]} in {parts[0]}"
        return "Playing music"
