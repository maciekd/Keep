"""common/imessage/Wire: which calls are on the wire and what a request's arguments must look like."""
from dataclasses import fields
from typing import Optional

import pytest

from common.imessage import Wire
from common.imessage.MessagesHost import MessagesHost


@pytest.mark.unit
class TestTheCallsComeFromTheInterface:
    def test_every_abstract_call_but_the_binary_one_is_a_json_call(self):
        """A call added to MessagesHost is on the wire without a line in the service or the client."""
        assert set(Wire.calls()) == set(MessagesHost.__abstractmethods__) - {"attachment"}

    def test_a_call_knows_which_arguments_may_be_left_out(self):
        send = Wire.calls()["send"]
        assert list(send.parameters) == ["recipient", "body", "owner"]
        assert send.defaults == {"owner": ""}
        assert Wire.calls()["poll"].defaults == {}


@pytest.mark.unit
class TestArguments:
    def test_a_missing_argument_takes_the_default_of_the_interface(self):
        assert Wire.arguments(Wire.calls()["health"], {}) == ({"refresh": False}, "")

    def test_a_missing_argument_without_a_default_is_refused_by_name(self):
        found, problem = Wire.arguments(Wire.calls()["poll"], {"after_rowid": 4, "horizon_ts": None})
        assert found == {} and "limit" in problem

    def test_an_argument_the_interface_lacks_is_dropped(self):
        """What a caller one version ahead may send."""
        found, problem = Wire.arguments(Wire.calls()["resolve"], {"recipient": "Alice", "owner": "Marcus", "tone": "warm"})
        assert (found, problem) == ({"recipient": "Alice", "owner": "Marcus"}, "")

    @pytest.mark.parametrize("data", [None, [], "poll", 7], ids=["nothing", "a list", "a string", "a number"])
    def test_anything_but_an_object_is_refused(self, data):
        found, problem = Wire.arguments(Wire.calls()["scheduled"], data)
        assert found == {} and problem

    @pytest.mark.parametrize("call, data", [
        ("poll", {"after_rowid": "4", "horizon_ts": None, "limit": 10}),
        ("poll", {"after_rowid": True, "horizon_ts": None, "limit": 10}),
        ("poll", {"after_rowid": 4.5, "horizon_ts": None, "limit": 10}),
        ("poll", {"after_rowid": 4, "horizon_ts": "yesterday", "limit": 10}),
        ("messages", {"rowids": [1, "2"]}),
        ("messages", {"rowids": 3}),
        ("messages", {"rowids": list(range(Wire.MAX_LIST_ITEMS + 1))}),
        ("send", {"recipient": ["Alice"], "body": "hi"}),
        ("send", {"recipient": "Alice", "body": None}),
        ("health", {"refresh": 1}),
    ], ids=["a string for a ROWID", "a bool for a ROWID", "a fraction for a ROWID", "a word for a time",
            "a string among the rows", "a number for a list", "more rows than a list may hold",
            "a list for a name", "nothing for a text", "a number for a flag"])
    def test_a_value_of_another_type_is_refused(self, call, data):
        found, problem = Wire.arguments(Wire.calls()[call], data)
        assert found == {} and problem

    def test_a_whole_number_passes_for_a_time_and_none_for_no_horizon(self):
        call = Wire.calls()["cursor_at"]
        assert Wire.arguments(call, {"horizon_ts": 1794000000})[1] == ""
        assert Wire.arguments(call, {"horizon_ts": None}) == ({"horizon_ts": None}, "")

    def test_fits_reads_an_optional_list(self):
        assert Wire.fits(Optional[list[int]], None) and Wire.fits(Optional[list[int]], [1, 2])
        assert not Wire.fits(Optional[list[int]], [1, None])

    def test_every_result_of_a_call_names_what_its_answer_cannot_be_without(self):
        """A call added to the interface with a result type of its own needs its entry: without
        one the answer to it could not be told from an empty object."""
        results = {call.result for call in Wire.calls().values()}
        assert results == set(Wire.ANSWER_FIELDS)
        for result, names in Wire.ANSWER_FIELDS.items():
            assert names and set(names) <= {f.name for f in fields(result)}

    def test_an_empty_object_is_no_answer_to_any_call(self):
        for call in Wire.calls().values():
            assert not Wire.answers(call.result, {}), call.name

    def test_what_a_host_answers_is_an_answer_and_stays_one_with_a_field_more_or_a_later_one_less(self):
        from common.imessage.MessagesHost import Count, MessageBatch, SendResult
        assert Wire.answers(Count, Wire.to_wire(Count(error="database is locked")))
        assert Wire.answers(SendResult, Wire.to_wire(SendResult(service="sent:imessage")))
        assert Wire.answers(MessageBatch, {"messages": [], "cursor": 7, "tapbacks": 2}), "a newer end's field"
        assert Wire.answers(SendResult, {"service": "sent", "error": ""}), "an older end knows fewer fields"
        assert not Wire.answers(SendResult, {"error": ""}) and not Wire.answers(Count, {"error": ""})

    def test_fits_reads_a_result_as_an_object_whose_known_fields_fit(self):
        from common.imessage.MessagesHost import MessageBatch
        assert Wire.fits(MessageBatch, {"messages": [{"rowid": 7, "reply_to": "p:0/abc"}], "tapbacks": 2})
        assert not Wire.fits(MessageBatch, {"messages": [{"rowid": None}]})
        assert not Wire.fits(MessageBatch, [{"rowid": 7}])
