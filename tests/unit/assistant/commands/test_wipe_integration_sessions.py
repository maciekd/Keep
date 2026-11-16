"""`//wipe` gathers the sessions the integration folders keep in the tree by discovery
(`Integration.wipe_targets`) and ends each on its account first (`Integration.log_out`)."""
from pathlib import Path
from unittest.mock import patch

from assistant.commands import DebugCommands as dc_mod
from assistant.integration.Integration import Integration


def _folder(tmp_path):
    session = tmp_path / "lamp.session"

    class LampIntegration(Integration):
        """A lamp. It lights. Nothing more."""
        name = "Lamp"
        config_keys = []
        log_outs = []

        def check_health(self):
            return True

        @classmethod
        def wipe_targets(cls):
            return [(session, "Lamp login"), (session.with_name("lamp.session-journal"), "Lamp journal")]

        @classmethod
        def log_out(cls):
            cls.log_outs.append(True)
            return "logged out of the lamp"

    class KettleIntegration(Integration):
        """A kettle. It boils. Nothing more."""
        name = "Kettle"
        config_keys = []

        def check_health(self):
            return True

    return session, LampIntegration, KettleIntegration


def test_integration_targets_are_every_folders_wipe_targets(tmp_path):
    session, lamp, kettle = _folder(tmp_path)
    with patch("assistant.integration.bootstrap.integration_classes", return_value=[lamp, kettle]):
        targets = dc_mod.integration_targets()
    assert [p for p, _ in targets] == [session, session.with_name("lamp.session-journal")]


def test_log_out_runs_for_the_folders_whose_session_is_present_and_notes_its_first_target(tmp_path):
    session, lamp, kettle = _folder(tmp_path)
    with patch("assistant.integration.bootstrap.integration_classes", return_value=[lamp, kettle]):
        assert dc_mod.DebugCommands._log_out([(Path("/elsewhere/floxy.db"), "database")]) == {}
        assert lamp.log_outs == []
        notes = dc_mod.DebugCommands._log_out([(session, "Lamp login"),
                                               (session.with_name("lamp.session-journal"), "Lamp journal")])
    assert notes == {session: "logged out of the lamp"} and lamp.log_outs == [True]


def test_a_log_out_that_raises_is_a_note_not_a_crash(tmp_path):
    session, lamp, _ = _folder(tmp_path)

    def boom():
        raise OSError("no network")

    with patch("assistant.integration.bootstrap.integration_classes", return_value=[lamp]), \
         patch.object(lamp, "log_out", side_effect=boom):
        notes = dc_mod.DebugCommands._log_out([(session, "Lamp login")])
    assert "no network" in notes[session]


def test_the_spotify_sign_in_is_a_core_wipe_target_whatever_speaker_system_is_installed():
    """Contract with `//wipe`: the refresh token spotipy keeps under cache/oauth is a signed-in
    state every speaker system shares, so it is the core's own target under the token path the
    sign-in writes — not a folder's (a HEOS-only house has it too), and not the `.cache` file of
    spotipy's default handler."""
    from assistant.integration.music.SpotifyAuth import TOKEN_PATH
    from integrations.sonos.SonosIntegration import SonosIntegration
    core = [path for path, _ in dc_mod.WIPE_TARGETS]
    assert TOKEN_PATH in core and not any(p.name == ".cache" for p in core)
    assert SonosIntegration.wipe_targets() == []
