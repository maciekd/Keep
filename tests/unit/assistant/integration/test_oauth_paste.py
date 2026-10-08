"""OAuthPaste: the pasted answer of a loopback-by-paste sign-in parsed and checked against the flow."""
from unittest.mock import patch

import pytest

from assistant.integration.OAuthPaste import SignIn, parse_pasted, pasted_code

REDIRECT = "http://localhost:18722/"
LINK = "https://accounts.example/authorize?client_id=cid&state=st-1"


@pytest.mark.unit
class TestParsePasted:
    def test_the_whole_address_or_its_query_with_quotes_around(self):
        assert parse_pasted(f'"{REDIRECT}?state=s1&code=4%2Fabc"') == ("4/abc", "s1", "")
        assert parse_pasted("state=s1&code=4%2Fabc") == ("4/abc", "s1", "")
        assert parse_pasted("?code=AQx&state=st") == ("AQx", "st", "")
        assert parse_pasted(f"{REDIRECT}?error=access_denied") == ("", "", "access_denied")

    def test_anything_without_the_parameters_is_no_code(self):
        for pasted in ("  4/0Abare  ", "'AQbare'", REDIRECT, "the page says it cannot be reached"):
            assert parse_pasted(pasted) == ("", "", ""), pasted
        # The sign-in link itself carries the state but no code: still nothing to exchange.
        assert parse_pasted(LINK)[0] == ""


@pytest.mark.unit
class TestPastedCode:
    def test_the_code_comes_out_only_with_the_flows_state(self):
        assert pasted_code(f"{REDIRECT}?state=st-1&code=4%2Fabc", "st-1", "Google", 'google_login("P")') == ("4/abc", "")

    def test_each_way_the_paste_can_be_wrong_is_named_and_the_login_command_given(self):
        login = 'google_login("P")'
        no_flow = pasted_code(f"{REDIRECT}?state=st-1&code=x", None, "Google", login)
        refused = pasted_code(f"{REDIRECT}?error=access_denied&state=st-1", "st-1", "Google", login)
        foreign = pasted_code(f"{REDIRECT}?state=st-2&code=x", "st-1", "Google", login)
        bare = pasted_code("4/0Abare", "st-1", "Google", login)
        no_code = pasted_code(f"{REDIRECT}?state=st-1&scope=email", "st-1", "Google", login)

        assert all(code == "" for code, _ in (no_flow, refused, foreign, bare, no_code))
        assert login in no_flow[1] and login in foreign[1]
        assert "access_denied" in refused[1]
        assert "whole address" in bare[1] and "whole address" in no_code[1]
        assert len({no_flow[1], refused[1], foreign[1], bare[1]}) == 4

    def test_a_refusal_is_reported_before_the_flow_is_checked(self):
        code, problem = pasted_code(f"{REDIRECT}?error=access_denied", None, "Spotify", "spotify_login()")
        assert code == "" and "access_denied" in problem


@pytest.mark.unit
class TestSignIn:
    def test_the_instructions_carry_the_link_the_landing_address_and_the_account_when_known(self):
        named = SignIn(LINK, "st-1", REDIRECT, "marcus@example.com")
        anonymous = SignIn(LINK, "st-1", REDIRECT)
        assert LINK in named.instructions and REDIRECT in named.instructions
        assert "marcus@example.com" in named.instructions
        assert " as " not in anonymous.instructions and REDIRECT in anonymous.instructions
