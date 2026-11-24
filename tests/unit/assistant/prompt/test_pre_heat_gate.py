"""The pre-heat gate of the PromptManager: no pre-heat goes out beside a turn or ahead of the
turn a boot opens the conversation with, the one asked for meanwhile is sent once the gate
opens, and a section note made while a turn runs stands behind the rows that turn wrote."""
import pytest

from tests.unit.assistant.prompt.test_gated_sections import indexed, manager, workspace  # noqa: F401 — fixtures

KETTLE = "Kettle: 40°C, heating"


@pytest.fixture
def heated(indexed):  # noqa: F811
    """A manager whose prompt a request has sent, and the pre-heats that went out since."""
    indexed.sent = []
    indexed.on_prompt_assembled(indexed.sent.append)
    return indexed


def _kettle_starts(manager):  # noqa: F811
    """A section change: a delta note, and with it a pre-heat of the longer prefix."""
    manager.monitor.folders["kettle"] = True
    manager.monitor.notify()


def _texts(messages) -> str:
    return "\n".join(str(m["content"]) for m in messages)


@pytest.mark.unit
def test_with_no_turn_in_flight_a_section_change_is_pre_heated_at_once(heated):
    _kettle_starts(heated)
    assert len(heated.sent) == 1 and KETTLE in _texts(heated.sent[0])


@pytest.mark.unit
def test_a_change_during_a_turn_is_pre_heated_once_when_the_turn_ends(heated):
    heated.turn_started()
    _kettle_starts(heated)
    heated._assemble_prompt()
    assert heated.sent == [], "nothing prefills beside the turn"
    heated.turn_ended()
    assert len(heated.sent) == 1, "two changes during the turn are one pre-heat after it"
    assert KETTLE in _texts(heated.sent[0]), "the prompt as it stands when the gate opens"


@pytest.mark.unit
def test_a_turn_during_which_nothing_changed_ends_without_a_pre_heat(heated):
    heated.turn_started()
    heated.turn_ended()
    assert heated.sent == []


@pytest.mark.unit
def test_the_owed_pre_heat_waits_for_the_last_of_two_turns(heated):
    heated.turn_started()
    heated.turn_started()
    _kettle_starts(heated)
    heated.turn_ended()
    assert heated.sent == [], "the second turn is still prefilling"
    heated.turn_ended()
    assert len(heated.sent) == 1


@pytest.mark.unit
def test_a_closing_turn_leaves_the_pre_heat_to_its_history_update(heated):
    heated.turn_started()
    _kettle_starts(heated)
    heated.turn_ended(closing=True)
    assert heated.sent == [], "a pre-heat here would be of the prompt without the exchange"
    heated.update_history("Marcus", "is the kettle on?", "It is heating.")
    assert len(heated.sent) == 1
    sent = _texts(heated.sent[0])
    assert "It is heating." in sent and KETTLE in sent


@pytest.mark.unit
def test_a_closing_turn_beside_another_one_owes_its_update_to_that_turns_end(heated):
    heated.turn_started()
    heated.turn_started()
    heated.turn_ended(closing=True)
    heated.update_history("Marcus", "and the sauna?", "The sauna is off.")
    assert heated.sent == []
    heated.turn_ended()
    assert len(heated.sent) == 1 and "The sauna is off." in _texts(heated.sent[0])


@pytest.mark.unit
def test_a_boot_hold_lasts_through_the_opening_turn(heated):
    heated.hold_pre_heats("the first-run brief opens the conversation")
    heated._assemble_prompt()
    assert heated.sent == []
    heated.turn_started()
    heated.release_pre_heats("too late: the turn has the hold")
    assert heated.sent == [], "the turn took the hold over, so nothing is released beside it"
    heated.turn_ended()
    assert len(heated.sent) == 1


@pytest.mark.unit
def test_a_boot_hold_whose_turn_never_comes_is_released_with_the_pre_heat_it_held_back(heated):
    heated.hold_pre_heats("the conversation the restart interrupted is resumed")
    _kettle_starts(heated)
    assert heated.sent == []
    heated.release_pre_heats("the turn is not opened")
    assert len(heated.sent) == 1 and KETTLE in _texts(heated.sent[0])
    heated.release_pre_heats("again")
    assert len(heated.sent) == 1, "a second release has nothing to send"


@pytest.mark.unit
def test_a_release_with_nothing_held_back_sends_nothing(heated):
    heated.hold_pre_heats("the first-run brief opens the conversation")
    heated.release_pre_heats("the turn is not opened")
    heated.release_pre_heats("no hold at all")
    assert heated.sent == []


@pytest.mark.unit
def test_after_the_gate_opened_the_pre_heats_go_out_directly_again(heated):
    heated.turn_started()
    heated.turn_ended()
    _kettle_starts(heated)
    assert len(heated.sent) == 1


def _order(manager, *needles) -> list[int]:  # noqa: F811
    """Where each text stands in the next request (the texts are each test's own: the workers'
    history outlives a test)."""
    contents = [str(m["content"]) for m in manager.get_messages()]
    return [next(i for i, content in enumerate(contents) if needle in content) for needle in needles]


@pytest.mark.unit
def test_a_note_made_during_a_turn_stands_behind_the_rows_the_turn_writes(heated):
    """The turn's rounds went out without the note: in front of them it would change the
    bytes of a reply the server has cached."""
    _kettle_starts(heated)
    heated.add_command_result("light_on: the hall light is on")
    heated.update_history("Marcus", "hall light please", "The hall light is on.")
    result, reply, note = _order(heated, "the hall light is on", "The hall light is on.", KETTLE)
    assert result < note and reply < note


@pytest.mark.unit
def test_a_note_a_request_has_sent_keeps_its_place_in_front_of_the_next_turn(heated):
    _kettle_starts(heated)
    heated.get_messages()
    heated.update_history("Marcus", "porch light please", "The porch light is on.")
    note, reply = _order(heated, KETTLE, "The porch light is on.")
    assert note < reply, "moved, the request that sent it would no longer be a prefix of the next"
