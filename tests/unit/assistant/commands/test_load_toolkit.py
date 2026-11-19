"""load_toolkit: the references come back as a background command's reply, a toolkit the
LLM still sees is answered with a pointer, and the command runs even in test-suite mode."""
from unittest.mock import Mock, patch

import pytest

from assistant.commands import Toolkits
from assistant.commands.Commands import Commands
from assistant.commands.CommandProcessor import CommandProcessor
from tests.unit.waiting import FAILURE_BOUND


@pytest.fixture
def commands():
    # A fresh CommandProcessor's shape: a bare Mock's truthy command_reply would make
    # _publish_bg_results fold it into the reply as a pending error.
    cp = Mock(command_reply=None, command_reply_contains_error=False)
    return Commands(cp)


def _prompt_manager(loaded: dict):
    pm = Mock()
    pm.render_template = lambda text: text.replace("{PHOTO_COUNT}", "1234")
    pm.message_history.loaded_toolkits.return_value = loaded
    pm.carried_data = Mock(side_effect=lambda toolkit, record=True: Toolkits.render_sections(
        toolkit, {"home": '## LIGHTS\n"Lamp [Kök]": off'}))
    return pm


@pytest.mark.unit
def test_home_brings_the_house_data_and_again_on_a_pointer_back(commands):
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: _prompt_manager({}))):
        fresh = commands._run_load_toolkit("home")
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: _prompt_manager({"home": 1000.0}))):
        again = commands._run_load_toolkit("home")
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: _prompt_manager({}))):
        web = commands._run_load_toolkit("web")

    assert fresh.startswith("# Toolkit: home\n") and fresh.rstrip().endswith('"Lamp [Kök]": off')
    assert "\n## HOME AUTOMATION DATA\n" in fresh
    assert again.startswith("The home toolkit was loaded at") and again.rstrip().endswith('"Lamp [Kök]": off')
    assert "HOME AUTOMATION DATA" not in web


@pytest.mark.unit
def test_the_autonomous_check_gets_the_data_without_setting_the_baseline(commands):
    """The wake-up's rounds never reach the history, so what it is sent must not pass for
    what the history's readers were sent (the delta notes are diffed against that)."""
    from assistant.ChannelProfile import AUTONOMOUS, TELEGRAM
    pm = _prompt_manager({"home": 1000.0})
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: pm)):
        commands._current_profile = AUTONOMOUS
        result = commands._run_load_toolkit("home")
        pm.carried_data.assert_called_once_with(Toolkits.by_name("home"), record=False)
        commands._current_profile = TELEGRAM
        commands._run_load_toolkit("home")
        assert pm.carried_data.call_args.kwargs == {"record": True}
    assert result.rstrip().endswith('"Lamp [Kök]": off')


@pytest.mark.unit
def test_cmd_goes_through_the_background_path(commands):
    """The whole real path: run() splats the parsed params into cmd_load_toolkit, and
    background_command splats the list into _run_load_toolkit — a name must not arrive as
    its characters (the bug the first console request hit: unknown ['w', 'e', 'b'])."""
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: _prompt_manager({}))):
        assert commands.run("load_toolkit", ['"web"', "x"]) == ""
        assert commands.wait_for_background(timeout=FAILURE_BOUND)
    commands.command_processor.set_command_reply.assert_called_once()
    reply = commands.command_processor.set_command_reply.call_args.args[0]
    assert "# Toolkit: web\n" in reply and "# Toolkit: x\n" in reply
    assert "No toolkit named" not in reply
    assert "load_toolkit" in Commands.MULTI_PARAM_COMMANDS
    assert commands.is_pure("load_toolkit") and not commands.is_pure("open_url")


@pytest.mark.unit
def test_renders_the_reference_with_placeholders_filled(commands):
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: _prompt_manager({}))):
        result = commands._run_load_toolkit("photos")
    assert result.startswith("# Toolkit: photos\n")
    assert "1234 indexed photos" in result
    assert "{PHOTO_COUNT}" not in result


@pytest.mark.unit
def test_several_names_in_one_call(commands):
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: _prompt_manager({}))):
        result = commands._run_load_toolkit('"email"', "x")
    assert "# Toolkit: email\n" in result
    assert "# Toolkit: x\n" in result
    assert result.index("# Toolkit: email") < result.index("# Toolkit: x")


@pytest.mark.unit
def test_toolkit_still_in_the_window_gets_a_pointer(commands):
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: _prompt_manager({"email": 0.0}))):
        result = commands._run_load_toolkit("email")
    assert "was loaded at" in result
    assert "latest_email" not in result


@pytest.mark.unit
def test_unknown_and_empty_names(commands):
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: _prompt_manager({}))):
        result = commands._run_load_toolkit("mail")
        empty = commands._run_load_toolkit(" ")
    assert result.startswith("No toolkit named 'mail'")
    assert "email" in result
    assert empty.startswith("load_toolkit failed")
    assert ", ".join(Toolkits.names()) in empty


@pytest.mark.unit
def test_without_a_prompt_manager_the_raw_file_is_served(commands):
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: None)):
        result = commands._run_load_toolkit("cars")
    assert "car_open_trunk" in result


@pytest.mark.unit
def test_test_mode_runs_the_pure_command_only():
    """The prompt suite must receive the reference, so load_toolkit runs in test mode while
    every other command is still only noted."""
    processor = CommandProcessor(test_suite_mode=True)
    ran = []
    processor.commands.run = lambda name, params: ran.append(name) or ""
    dm = Mock(is_killed=False)
    dm.web_request_handler.wait_so_that_response_timing_is_natural = Mock(return_value=True)
    try:
        processor.invoke_command("load_toolkit", ["email"], dm)
        processor.invoke_command("turn_on", ["Kitchen lamp"], dm)
        processor.command_queue.join()
    finally:
        processor.kill()
    assert ran == ["load_toolkit"]
