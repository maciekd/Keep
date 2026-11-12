"""A SpotifyAccount for tests: built without its __init__, so no config is read and no token file is touched."""
import threading
from unittest.mock import Mock

from assistant.integration.SingletonMixin import SingletonMixin
from assistant.integration.music.SpotifyAccount import SpotifyAccount, SpotifyPlaylist


def make_account() -> SpotifyAccount:
    """A SpotifyAccount bypassing __init__: no config, a mocked user token, two cached playlists."""
    account = SpotifyAccount.__new__(SpotifyAccount)
    account.config = Mock()
    account.config.get.return_value = None
    account._auth = Mock()
    account._auth.signed_in.return_value = True
    account._auth.needs_sign_in_again.return_value = False
    account._auth.lacks_scopes.return_value = False
    account._app_client = None
    account._user_client = None
    account._user_client_token = ""
    account._lock = threading.Lock()
    account._playlists = [SpotifyPlaylist("Discover Weekly", "spotify:playlist:dw1", "Spotify"),
                          SpotifyPlaylist("Road Trip", "spotify:playlist:rt2", "")]
    account._playlists_total = 2
    account._library_error = ""
    account._library_fetched_at = 0.0
    account._not_before = 0.0
    account._sign_in_again_noted = False
    account._search_ids = {}
    account._next_search_id = 0
    return account


def install(account: SpotifyAccount) -> SpotifyAccount:
    """Makes `account` the process's SpotifyAccount (what `SpotifyAccount.get_instance()` hands out)."""
    SingletonMixin._instances[SpotifyAccount] = account
    return account
