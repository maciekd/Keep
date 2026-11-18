"""Which logged errors may be a defect in the code, and which the code foresaw."""
import pytest
import requests

from assistant.ErrorKind import classify
from assistant.integration.Integration import HealthWarning


def _raised_by(module: str, name: str = "ServiceError", **attributes) -> Exception:
    """An exception of a class *module* defines, as a client library would raise it."""
    exception = type(name, (Exception,), {"__module__": module})("the garage service said no")
    for key, value in attributes.items():
        setattr(exception, key, value)
    return exception


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


def _is_defect(*chain, handled=True, outside=False) -> bool:
    return classify(list(chain), handled, outside)[0]


@pytest.mark.unit
class TestDefects:
    def test_an_exception_nothing_caught_is_a_defect_even_when_it_is_the_network_s(self):
        assert _is_defect(ConnectionResetError("reset"), handled=False)

    @pytest.mark.parametrize("exception", [TypeError("x"), AttributeError("x"), KeyError("zones"),
                                           IndexError("x"), ZeroDivisionError("x")])
    def test_a_programming_error_is_a_defect_whatever_the_call_site_says(self, exception):
        assert _is_defect(exception, outside=True)

    def test_a_generic_exception_nobody_named_is_a_defect(self):
        assert _is_defect(ValueError("Missing key inputs argument"))
        assert _is_defect(RuntimeError("the door reports two states"))

    def test_a_library_s_exception_outside_the_listed_ones_is_a_defect(self):
        assert _is_defect(_raised_by("numpy.exceptions", "AxisError"))

    @pytest.mark.parametrize("status", [400, 422])
    def test_a_service_s_answer_that_the_request_was_malformed_is_a_defect(self, status):
        assert _is_defect(_raised_by("anthropic", "BadRequestError", status_code=status))
        assert _is_defect(_raised_by("httpx", "HTTPStatusError", response=_Response(status)), outside=True)

    def test_a_programming_error_raised_while_handling_an_outage_is_a_defect(self):
        assert _is_defect(TypeError("x"), ConnectionResetError("reset"))


@pytest.mark.unit
class TestForeseen:
    def test_a_message_with_no_exception_behind_it_is_foreseen(self):
        assert not _is_defect()

    @pytest.mark.parametrize("exception", [ConnectionResetError("reset"), TimeoutError("slow"),
                                           PermissionError("denied"), requests.ReadTimeout("slow")])
    def test_what_the_system_or_the_network_refused_is_foreseen(self, exception):
        assert not _is_defect(exception)

    def test_a_client_library_s_own_exception_is_foreseen(self):
        assert not _is_defect(_raised_by("caldav.lib.error", "PropfindError"))
        assert not _is_defect(_raised_by("telegram.error", "NetworkError"))

    @pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
    def test_a_status_about_access_rate_or_the_service_itself_is_foreseen(self, status):
        assert not _is_defect(_raised_by("openai", "APIStatusError", status_code=status))

    def test_a_failure_the_tree_gave_a_class_of_its_own_is_foreseen(self):
        assert not _is_defect(HealthWarning("one account of two is signed out"))
        assert not _is_defect(_raised_by("integrations.garage.GarageClient", "DoorAway"))

    def test_a_generic_exception_is_foreseen_when_it_came_from_an_outage(self):
        assert not _is_defect(RuntimeError("the door did not answer"), ConnectionResetError("reset"))

    def test_a_generic_exception_is_foreseen_where_the_call_site_reports_the_outside(self):
        assert not _is_defect(RuntimeError("Work: token expired"), outside=True)


@pytest.mark.unit
def test_the_verdict_comes_with_its_reason():
    defect, why = classify([_raised_by("caldav.lib.error", "PropfindError")])
    assert not defect and "PropfindError" in why and "caldav" in why
