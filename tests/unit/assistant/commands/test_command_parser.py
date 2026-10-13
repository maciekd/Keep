import pytest
from assistant.commands.CommandParser import CommandParser


@pytest.mark.unit
class TestCommandParser:
    """Tests for CommandParser logic"""

    @pytest.fixture
    def parser(self):
        """Create a parser with common commands"""
        commands = ["turn_on", "turn_off", "dim", "play", "remember", "execute", "false_positive", "create_event", "show_history", "list_integrations",
                    "self_improvement", "self_analysis", "config_set"]
        return CommandParser(commands)

    def test_is_possibly_command_empty_string(self, parser):
        """Empty string is considered possibly a command"""
        assert parser.is_possibly_command("") is True

    def test_is_possibly_command_partial_match(self, parser):
        """Partial command names should be recognized"""
        assert parser.is_possibly_command("tur") is True
        assert parser.is_possibly_command("turn_") is True
        assert parser.is_possibly_command("turn_on") is True
        assert parser.is_possibly_command("turn_on(") is True

    def test_is_possibly_command_complete_match(self, parser):
        """Complete commands should be recognized"""
        assert parser.is_possibly_command("play(") is True
        assert parser.is_possibly_command("dim(kitchen") is True

    def test_is_possibly_command_not_a_command(self, parser):
        """Non-commands should be rejected"""
        assert parser.is_possibly_command("hello world") is False
        assert parser.is_possibly_command("this is text") is False
        assert parser.is_possibly_command("xyz123") is False

    def test_find_complete_command_with_closing_paren(self, parser):
        """Should find complete command with closing parenthesis"""
        assert parser.find_complete_command("turn_on(lights)") is True
        assert parser.find_complete_command("play(music) and more") is True

    def test_find_complete_command_without_closing_paren(self, parser):
        """Should not find complete command without closing parenthesis"""
        assert parser.find_complete_command("turn_on(lights") is False
        assert parser.find_complete_command("play(") is False

    def test_extract_parameters_simple(self, parser):
        """Should extract simple parameters"""
        assert parser.extract_parameters("turn_on(kitchen lights)") == "kitchen lights"
        assert parser.extract_parameters("dim(50)") == "50"

    def test_extract_parameters_empty(self, parser):
        """Should handle empty parameters"""
        assert parser.extract_parameters("execute()") == ""
        assert parser.extract_parameters("no_params") == ""

    def test_clean_parameters_removes_quotes(self, parser):
        """Should remove various quote types"""
        assert parser.clean_parameters('"kitchen lights"') == "kitchen lights"
        assert parser.clean_parameters("'bedroom'") == "bedroom"
        assert parser.clean_parameters('`hallway`') == "hallway"

    def test_clean_parameters_removes_param_names(self, parser):
        """Should remove parameter name prefixes"""
        assert parser.clean_parameters("light_name: kitchen") == "kitchen"
        assert parser.clean_parameters("dim_percentage:50") == "50"
        assert parser.clean_parameters("room_name=bedroom") == "bedroom"

    def test_clean_parameters_replaces_brackets(self, parser):
        """Should replace brackets with parentheses"""
        assert parser.clean_parameters("[item1, item2]") == "(item1, item2)"

    def test_clean_parameters_removes_type_annotations(self, parser):
        """Should remove type annotations"""
        assert parser.clean_parameters("str=kitchen") == "kitchen"
        assert parser.clean_parameters("int=50") == "50"
        assert parser.clean_parameters("str:bedroom") == "bedroom"

    def test_clean_parameters_complex(self, parser):
        """Should handle complex parameter cleaning"""
        input_params = 'light_name: "kitchen lamp", dim_percentage: int=50'
        expected = "kitchen lamp, 50"
        assert parser.clean_parameters(input_params) == expected

    def test_clean_parameters_preserves_url_schemes(self, parser):
        """URL schemes (https:, http:, ftp:) must survive parameter cleaning"""
        assert parser.clean_parameters("https://www.reuters.com") == "https://www.reuters.com"
        assert parser.clean_parameters("http://example.com/page") == "http://example.com/page"
        assert parser.clean_parameters("ftp://files.example.com") == "ftp://files.example.com"
        assert parser.clean_parameters("url: https://www.reuters.com/world") == "https://www.reuters.com/world"

    def test_extract_command_and_params_simple(self, parser):
        """Should extract command name and parameters"""
        result, remainder = parser.extract_command_and_params("turn_on(kitchen lights) hello")

        assert result[0] == "turn_on"
        assert result[1] == ["kitchen lights"]
        assert remainder == " hello"

    def test_extract_command_and_params_incomplete(self, parser):
        """Should return None for incomplete commands"""
        result, remainder = parser.extract_command_and_params("turn_on(lights")

        assert result is None
        assert remainder is None

    def test_extract_command_and_params_empty_params(self, parser):
        """Should handle commands with empty parameters"""
        result, remainder = parser.extract_command_and_params("execute()")

        assert result[0] == "execute"
        assert result[1] == []
        assert remainder == ""

    def test_case_insensitive_command_names(self, parser):
        """Command names should be case-insensitive"""
        result, _ = parser.extract_command_and_params("TURN_ON(lights)")
        assert result[0] == "turn_on"

        result, _ = parser.extract_command_and_params("TuRn_OfF(lights)")
        assert result[0] == "turn_off"

    def test_multiple_commands_in_sequence(self, parser):
        """Should handle extracting first command when multiple are present"""
        content = "turn_on(kitchen) turn_off(bedroom)"
        result, remainder = parser.extract_command_and_params(content)

        assert result[0] == "turn_on"
        assert result[1] == ["kitchen"]
        assert "turn_off(bedroom)" in remainder

    def test_command_with_special_characters_in_params(self, parser):
        """Should handle special characters in parameters"""
        result, _ = parser.extract_command_and_params("play(Rock & Roll)")
        assert result[1] == ["Rock & Roll"]

        result, _ = parser.extract_command_and_params("remember(Don't forget!)")
        joined = ", ".join(result[1])
        assert "Don" in joined and "forget" in joined

    def test_nested_parentheses_in_params(self, parser):
        """Should handle nested parentheses in parameters"""
        result, remainder = parser.extract_command_and_params(
            "remember(AC units (Gästhuset, Sovrum), but not kitchen (fixed at 22°C).)"
        )
        assert result[0] == "remember"
        joined = ", ".join(result[1])
        assert "Gästhuset" in joined
        assert "kitchen" in joined
        assert "22°C" in joined
        assert remainder == ""

    def test_find_complete_command_with_nested_parens(self, parser):
        """Should not report complete until all parentheses are balanced"""
        assert parser.find_complete_command("remember(units (A, B)") is False
        assert parser.find_complete_command("remember(units (A, B))") is True

    def test_multiple_commands_with_nested_parens(self, parser):
        """Should extract first command correctly when it has nested parens"""
        content = "remember(note (important)) turn_on(kitchen)"
        result, remainder = parser.extract_command_and_params(content)
        assert result[0] == "remember"
        assert "(important)" in ", ".join(result[1])
        assert "turn_on(kitchen)" in remainder

    # ===== <code> tag handling =====

    def test_is_possibly_command_with_code_tag(self, parser):
        """Should recognize commands wrapped in <code> tags"""
        assert parser.is_possibly_command("<code>turn_on(") is True
        assert parser.is_possibly_command("<code>dim(kitchen") is True

    def test_is_possibly_command_code_tag_only(self, parser):
        """<code> alone strips to empty, which is possibly a command"""
        assert parser.is_possibly_command("<code>") is True

    def test_extract_command_wrapped_in_code_tags(self, parser):
        """Should extract command from <code>cmd(params)</code>"""
        result, remainder = parser.extract_command_and_params("<code>turn_on(kitchen)</code> Done!")
        assert result[0] == "turn_on"
        assert result[1] == ["kitchen"]
        assert remainder == " Done!"

    def test_extract_command_with_opening_code_tag_only(self, parser):
        """Should extract command from <code>cmd(params)"""
        result, remainder = parser.extract_command_and_params("<code>dim(lamp, 75)")
        assert result[0] == "dim"
        assert result[1] == ["lamp", "75"]

    def test_find_complete_command_with_code_tags(self, parser):
        """Should find complete command through <code> tags"""
        assert parser.find_complete_command("<code>turn_on(kitchen)</code>") is True
        assert parser.find_complete_command("<code>turn_on(kitchen") is False

    # ===== find_command_start() =====

    def test_find_command_start_at_position_zero(self, parser):
        assert parser.find_command_start("turn_on(kitchen)") == 0

    def test_find_command_start_after_prefix_text(self, parser):
        assert parser.find_command_start("Sure! turn_on(kitchen)") == 6

    def test_find_command_start_not_found(self, parser):
        assert parser.find_command_start("no commands here") == -1

    def test_find_command_start_second_command_earlier(self, parser):
        pos = parser.find_command_start("Hello turn_off(b) and turn_on(a)")
        assert pos == 6

    def test_find_command_start_case_insensitive(self, parser):
        assert parser.find_command_start("TURN_ON(kitchen)") == 0
        assert parser.find_command_start("Sure! Turn_Off(bedroom)") == 6

    def test_find_command_start_with_code_tags_returns_original_position(self, parser):
        """Position must refer to the original string, not a stripped version (regression for 73c66685)"""
        assert parser.find_command_start("<code>turn_on(kitchen)</code>") == 6
        assert parser.find_command_start("<code>dim(lamp, 50)</code>") == 6

    # ===== extract_all_commands() =====

    def test_extract_all_commands_single(self, parser):
        commands = parser.extract_all_commands("turn_on(kitchen)")
        assert len(commands) == 1
        assert commands[0] == ("turn_on", ["kitchen"])

    def test_extract_all_commands_multiple(self, parser):
        commands = parser.extract_all_commands("turn_on(kitchen) turn_off(bedroom)")
        assert len(commands) == 2
        assert ("turn_on", ["kitchen"]) in commands
        assert ("turn_off", ["bedroom"]) in commands

    def test_extract_all_commands_with_prefix_text(self, parser):
        commands = parser.extract_all_commands("Sure! Here's the result: turn_on(kitchen)")
        assert len(commands) == 1
        assert commands[0] == ("turn_on", ["kitchen"])

    def test_extract_all_commands_with_backticks(self, parser):
        commands = parser.extract_all_commands("`turn_on(kitchen)`")
        assert len(commands) == 1
        assert commands[0] == ("turn_on", ["kitchen"])

    def test_extract_all_commands_no_commands(self, parser):
        assert parser.extract_all_commands("Just plain text here") == []

    # ===== Quote-aware paren matching =====

    def test_find_complete_command_ignores_parens_in_quotes(self, parser):
        """Parentheses inside double-quoted strings should not count"""
        assert parser.find_complete_command('remember(note: "item (important)")') is True
        assert parser.find_complete_command('remember(note: "item (important)"') is False

    def test_find_complete_command_numbered_list_in_quotes(self, parser):
        """Numbered list items like 1) 2) inside quotes should not close the command"""
        content = 'execute(report: "Please: 1) check logs 2) verify"'
        assert parser.find_complete_command(content) is False
        content_closed = 'execute(report: "Please: 1) check logs 2) verify")'
        assert parser.find_complete_command(content_closed) is True

    def test_find_matching_paren_skips_quoted_parens(self, parser):
        """Nested parens inside quotes should be ignored by depth tracking"""
        result, remainder = parser.extract_command_and_params(
            'execute(report: "caffeinate() was called (twice)") done'
        )
        assert result is not None
        assert result[0] == "execute"
        joined = ", ".join(result[1])
        assert "caffeinate" in joined
        assert "twice" in joined
        assert remainder == " done"

    def test_command_with_numbered_list_in_quoted_param(self, parser):
        """The exact bug: numbered list 1) inside quoted param closed the command early"""
        content = 'execute(bug: "Please investigate: 1) Was caffeinate() called? 2) Did it work?")'
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert result[0] == "execute"
        assert "caffeinate" in ", ".join(result[1])
        assert remainder == ""

    def test_extract_all_commands_ignores_commands_in_quoted_params(self, parser):
        """Command names inside quoted params should not be extracted as separate commands"""
        content = 'execute(report: "Was execute() called?") turn_on(kitchen)'
        commands = parser.extract_all_commands(content)
        assert len(commands) == 2
        assert commands[0][0] == "execute"
        assert commands[1][0] == "turn_on"

    # ===== clean_parameters preserves colons in values =====

    def test_clean_parameters_preserves_colon_in_value(self, parser):
        """Colons inside parameter values should not be stripped"""
        result = parser.clean_parameters('note: "USER REPORT: something happened"')
        assert "REPORT:" in result

    def test_clean_parameters_strips_multiple_param_names_after_commas(self, parser):
        """Parameter names after commas should still be stripped"""
        result = parser.clean_parameters("room: kitchen, brightness: 50")
        assert result == "kitchen, 50"

    def test_clean_parameters_preserves_mid_text_colon(self, parser):
        """word: patterns in the middle of text (not at param positions) should survive"""
        result = parser.clean_parameters("The user says: hello world")
        assert result == "The user says: hello world"

    # ===== Numbered list marker handling (unquoted) =====

    def test_numbered_list_after_newline_not_premature_close(self, parser):
        """Numbered list items like 1) 2) after newlines should not close the command"""
        content = "execute(Please analyze:\n1) Check the logs\n2) Verify the output\n3) Report findings)"
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert result[0] == "execute"
        joined = ", ".join(result[1])
        assert "Check the logs" in joined
        assert "Report findings" in joined
        assert remainder == ""

    def test_find_complete_command_with_unquoted_numbered_list(self, parser):
        """find_complete_command should not be fooled by numbered list markers"""
        incomplete = "execute(Please:\n1) Check logs\n2) Verify"
        assert parser.find_complete_command(incomplete) is False
        complete = "execute(Please:\n1) Check logs\n2) Verify)"
        assert parser.find_complete_command(complete) is True

    def test_inline_list_marker_after_colon_with_content_following(self, parser):
        """Inline '1)' after multi-word label colon with text following should be a list marker"""
        content = "execute(Here are the steps: 1) do this and that)"
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert "do this and that" in ", ".join(result[1])

    def test_param_value_after_colon_not_list_marker(self, parser):
        """A numeric param value like 'param: 75)' at end should NOT be a list marker"""
        result, _ = parser.extract_command_and_params('dim(light_name: "lamp", dim_percentage: 75)')
        assert result is not None
        assert result[0] == "dim"
        assert result[1] == ["lamp", "75"]

    def test_nested_parens_with_numbered_list(self, parser):
        """Real-world bug: nested parens + numbered list caused premature close"""
        content = (
            "execute(Analyze the image (a desk setup photo) and answer:\n"
            "1) Was the image clear?\n"
            "2) Did it show the monitor?)"
        )
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert result[0] == "execute"
        joined = ", ".join(result[1])
        assert "desk setup photo" in joined
        assert "Was the image clear" in joined
        assert remainder == ""

    def test_numeric_param_not_treated_as_list_marker(self, parser):
        """Numeric params like dim(50) should not be treated as list markers"""
        result, _ = parser.extract_command_and_params("dim(50)")
        assert result is not None
        assert result[0] == "dim"
        assert result[1] == ["50"]

    def test_room_number_not_treated_as_list_marker(self, parser):
        """Room names with numbers like 'Bedroom 2' should not be treated as list markers"""
        result, _ = parser.extract_command_and_params("turn_on(Bedroom 2)")
        assert result is not None
        assert result[0] == "turn_on"
        assert result[1] == ["Bedroom 2"]

    def test_comma_number_not_treated_as_list_marker(self, parser):
        """Numbers after comma like 'lamp, 50' should not be treated as list markers"""
        result, _ = parser.extract_command_and_params("dim(lamp, 50)")
        assert result is not None
        assert result[0] == "dim"
        assert result[1] == ["lamp", "50"]

    def test_paren_number_not_treated_as_list_marker(self, parser):
        """Numbers right after opening paren like set(5) should not be list markers"""
        result, _ = parser.extract_command_and_params("execute(5)")
        assert result is not None
        assert result[1] == ["5"]

    def test_decimal_argument_not_treated_as_list_marker(self, parser):
        """A bare decimal like 57.5501633) is a number — the digits after the dot are not a
        list marker (regression: the LLM's coordinates failed to parse until it quoted them)"""
        text = 'config_set("location.home.latitude", 57.5501633)'
        for streaming in (True, False):
            result, remainder = parser.extract_command_and_params(text, streaming=streaming)
            assert result == ("config_set", ["location.home.latitude", "57.5501633"]), streaming
            assert remainder == ""
        result, remainder = parser.extract_command_and_params(
            'config_set("x.y", 11.9462948)\nSparar koordinaterna.', streaming=True)
        assert result == ("config_set", ["x.y", "11.9462948"])
        assert "Sparar" in remainder

    def test_sentence_then_numbered_list_still_a_list_marker(self, parser):
        """The decimal rule must not undo the sentence-end rule: 'done. 2) next' stays a list"""
        assert parser.find_complete_command("execute(First step done. 2) next step)", streaming=True) is True
        assert parser.find_complete_command("execute(First step done. 2)", streaming=True) is False

    # ===== Streaming mode: ambiguous end-of-buffer =====

    def test_streaming_does_not_match_ambiguous_digit_paren_at_end(self, parser):
        """During streaming, multi-word label colon + digit paren should NOT be treated as complete"""
        assert parser.find_complete_command("execute(Here are the steps: 1)", streaming=True) is False

    def test_non_streaming_matches_digit_paren_at_end(self, parser):
        """After stream ends, multi-word label colon + digit paren IS complete (1 is a param value)"""
        assert parser.find_complete_command("execute(Here are the steps: 1)", streaming=False) is True

    def test_streaming_matches_when_content_follows_digit_paren(self, parser):
        """During streaming, multi-word label '1) text' has content after 1) so it's a list marker"""
        assert parser.find_complete_command("execute(Here are the steps: 1) do this)", streaming=True) is True

    def test_streaming_newline_list_marker_unaffected(self, parser):
        """Newline-based list markers don't need the streaming flag"""
        assert parser.find_complete_command("execute(Steps:\n1)", streaming=True) is False
        assert parser.find_complete_command("execute(Steps:\n1)", streaming=False) is False

    # ===== Named parameter colon vs list marker (regression: ccc835e / d59f466) =====

    def test_named_param_colon_not_list_marker_streaming(self, parser):
        """schedule_wakeup(in_minutes: 52) must be complete even during streaming"""
        assert parser.find_complete_command("schedule_wakeup(in_minutes: 52)", streaming=True) is True

    def test_named_param_colon_with_trailing_newline_text(self, parser):
        """Command with named param followed by response text on new lines"""
        content = "schedule_wakeup(in_minutes: 52)\n\nGot it! I'll preheat the Model X."
        assert parser.find_complete_command(content, streaming=False) is True

    def test_named_param_colon_extract(self, parser):
        """Named param colon should extract command correctly"""
        content = "schedule_wakeup(in_minutes: 52)\n\nGot it!"
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert result[0] == "schedule_wakeup"
        assert result[1] == ["52"]
        assert "Got it!" in remainder

    def test_named_param_after_comma_not_list_marker(self, parser):
        """Colon param after comma should not be treated as list marker"""
        assert parser.find_complete_command("dim(room: kitchen, brightness: 75)", streaming=True) is True

    def test_single_identifier_colon_is_param_syntax(self, parser):
        """A single identifier right after '(' followed by ':' is param syntax"""
        assert parser.find_complete_command("execute(Steps: 1)", streaming=True) is True
        assert parser.find_complete_command("execute(Steps: 1)", streaming=False) is True

    def test_multi_word_label_colon_still_list_marker(self, parser):
        """Multi-word text before colon is a label, not param — list marker applies"""
        content = "execute(Here are the steps: 1) do this 2) do that)"
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert "do this" in ", ".join(result[1])

    def test_question_mark_trigger_unaffected_by_colon_fix(self, parser):
        """The '?' trigger for list markers is not affected by the colon param fix"""
        content = "execute(whats this? 1) whats the name)"
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert "whats the name" in ", ".join(result[1])

    # ===== Time value in HH:MM format (regression: _is_list_marker_paren false positive) =====

    def test_time_param_not_treated_as_list_marker(self, parser):
        """Time value like 12:00 in a named param must not trigger list marker detection"""
        content = "create_event(title: Meeting, end_time: 12:00) I've created the event."
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert result[0] == "create_event"
        assert "Meeting" in ", ".join(result[1])
        assert remainder == " I've created the event."

    def test_time_param_first_after_open_paren(self, parser):
        """Time value in a named param right after '(' should not be a list marker"""
        content = "create_event(start_time: 08:00) Done."
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert result[0] == "create_event"
        assert remainder == " Done."

    def test_time_param_streaming(self, parser):
        """Time value param should not be treated as list marker during streaming"""
        assert parser.find_complete_command(
            "create_event(title: Meeting, end_time: 12:00)", streaming=True
        ) is True

    def test_multi_word_label_with_time_still_list_marker(self, parser):
        """A multi-word label followed by time-like pattern should still be a list marker"""
        content = "execute(Meeting at 12:00) some text after"
        result, remainder = parser.extract_command_and_params(content)
        assert result is None

    # ===== Positional date-time values (regression: show_history(2026-09-07 00:00, 2026-09-07 23:59)) =====

    def test_positional_datetime_range_not_list_marker(self, parser):
        """'23:59)' at the end of a positional date-time range is the command terminator"""
        content = "show_history(2026-09-07 00:00, 2026-09-07 23:59)\nlist_integrations()\n\nChecking Monday."
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert result[0] == "show_history"
        assert result[1] == ["2026-09-07 00:00", "2026-09-07 23:59"]
        assert remainder.startswith("\nlist_integrations()")

    def test_positional_datetime_range_streaming(self, parser):
        assert parser.find_complete_command(
            "show_history(2026-09-07 00:00, 2026-09-07 23:59)", streaming=True) is True

    # ===== Time value with equals-sign parameter assignment (regression: end_time=10:00) =====

    def test_time_param_with_equals_not_list_marker(self, parser):
        """Time value like 10:00 in a param assigned with = must not trigger list marker"""
        content = 'create_event(title="Meeting", date=2026-04-06, start_time=08:00, end_time=10:00) Done!'
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None
        assert result[0] == "create_event"
        assert remainder == " Done!"

    def test_time_param_with_equals_streaming(self, parser):
        """Time value param with = should not be treated as list marker during streaming"""
        assert parser.find_complete_command(
            'create_event(title="Meeting", end_time=10:00)', streaming=True
        ) is True

    # ===== Remainder preserves leading space (regression: words merging in Telegram) =====

    def test_remainder_preserves_leading_space(self, parser):
        """Leading space in remainder must be preserved to avoid merged words like 'trya'"""
        result, remainder = parser.extract_command_and_params("turn_on(kitchen) try a different approach")
        assert result[0] == "turn_on"
        assert remainder == " try a different approach"

    def test_remainder_no_leading_space_when_none_present(self, parser):
        """When there's no space after closing paren, remainder has no leading space"""
        result, remainder = parser.extract_command_and_params("turn_on(kitchen)\nnext line")
        assert result[0] == "turn_on"
        assert remainder == "\nnext line"

    # ===== self_improvement with deeply nested parentheses and numbered lists =====

    def test_self_improvement_with_nested_parens_and_numbered_list(self, parser):
        """Real command with many nested parentheses and numbered list markers"""
        content = (
            'self_improvement(Fix the Telegram HTML formatting fallback in '
            'TelegramBotIntegration.py _send_or_edit method (around lines 376-420). '
            'Currently when sending/editing a message with parse_mode="HTML" fails, '
            'the except Exception block silently falls back to sending the SAME text '
            'WITHOUT parse_mode — causing users to see raw HTML tags like literal '
            '<b>, <code>, <pre> in the message. The fix should: '
            '1) LOG the error when the HTML send/edit fails (log.warning or log.error '
            'with the exception details), '
            '2) RETRY ONCE with parse_mode="HTML" before giving up on HTML, '
            '3) Only THEN fall back — but strip all HTML tags from the text before '
            'sending without parse_mode, so the user sees clean plain text instead of '
            'raw tags. Apply this pattern to BOTH the new-message path '
            '(around line 376-383) and the edit-message path (around line 400-404), '
            'and also the _fallback_send_new method if it has the same issue. Use the '
            'existing _sanitize_html or a simple regex like re.sub(r\'<[^>]*>\', \'\', text) '
            'to strip tags in the fallback. Do NOT change the autonomous message path in '
            'AutonomousActionManager.py — only fix TelegramBotIntegration.py.)'
        )
        result, remainder = parser.extract_command_and_params(content)
        assert result is not None, f"Failed to parse command. Content length: {len(content)}"
        assert result[0] == "self_improvement"
        # One verbatim argument: the dispatch hands a single-parameter command its first
        # piece only, so a request split on its commas would lose everything after the first
        assert len(result[1]) == 1
        request = result[1][0]
        assert request.startswith("Fix the Telegram HTML formatting")
        assert request.endswith("only fix TelegramBotIntegration.py.")
        assert "(around lines 376-420)" in request
        assert "re.sub(r'<[^>]*>', '', text)" in request
        assert remainder == ""

    def test_free_text_request_with_commas_and_apostrophe_is_one_argument(self, parser):
        """A self_analysis question is prose: a comma does not start a second argument, an
        apostrophe does not open a string for the paren matching, and nothing is cleaned."""
        question = ("At 19:21 on Oct 7, Marcus's timer (three minutes, kitchen) never rang; "
                    "why, and was it the gate?")
        result, remainder = parser.extract_command_and_params(f"self_analysis({question}) then more")
        assert result == ("self_analysis", [question])
        assert remainder == " then more"

    @pytest.mark.parametrize("command", ["improve_integration", "report_issue"])
    def test_the_other_prose_commands_take_their_request_whole_too(self, command):
        """improve_integration splits its leading integration name off itself; the parser must
        hand it the whole payload, commas and apostrophes included, like self_improvement."""
        request = "Tesla, the trunk didn't close at 10:46 (Oct 8), see the log; it's the second time"
        result, remainder = CommandParser([command]).extract_command_and_params(f"{command}({request})")
        assert result == (command, [request])
        assert remainder == ""

    def test_self_improvement_find_complete_command(self, parser):
        """find_complete_command should recognize the deeply nested self_improvement"""
        content = (
            'self_improvement(Fix the Telegram HTML formatting fallback in '
            'TelegramBotIntegration.py _send_or_edit method (around lines 376-420). '
            'Currently when sending/editing a message with parse_mode="HTML" fails, '
            'the except Exception block silently falls back to sending the SAME text '
            'WITHOUT parse_mode — causing users to see raw HTML tags like literal '
            '<b>, <code>, <pre> in the message. The fix should: '
            '1) LOG the error when the HTML send/edit fails (log.warning or log.error '
            'with the exception details), '
            '2) RETRY ONCE with parse_mode="HTML" before giving up on HTML, '
            '3) Only THEN fall back — but strip all HTML tags from the text before '
            'sending without parse_mode, so the user sees clean plain text instead of '
            'raw tags. Apply this pattern to BOTH the new-message path '
            '(around line 376-383) and the edit-message path (around line 400-404), '
            'and also the _fallback_send_new method if it has the same issue. Use the '
            'existing _sanitize_html or a simple regex like re.sub(r\'<[^>]*>\', \'\', text) '
            'to strip tags in the fallback. Do NOT change the autonomous message path in '
            'AutonomousActionManager.py — only fix TelegramBotIntegration.py.)'
        )
        assert parser.find_complete_command(content) is True
        assert parser.find_complete_command(content, streaming=True) is True


@pytest.mark.unit
class TestBrowseEvalRawParams:
    """browse_eval forwards its argument verbatim — clean_parameters would mangle JS."""

    @pytest.fixture
    def parser(self):
        return CommandParser(["browse_eval", "browse_open", "turn_on", "dim"])

    def test_browse_eval_preserves_quotes(self, parser):
        js = "return document.querySelector('a[href*=\"bankid\"]').href"
        result, remainder = parser.extract_command_and_params(f"browse_eval({js})")
        assert result[0] == "browse_eval"
        assert result[1] == [js]
        assert remainder == ""

    def test_browse_eval_preserves_brackets_and_colons(self, parser):
        js = "return [...document.querySelectorAll('input')].map(i => ({name: i.name, value: i.value}))"
        result, _ = parser.extract_command_and_params(f"browse_eval({js})")
        assert result[1] == [js]

    def test_browse_eval_in_extract_all_commands(self, parser):
        js = "return {a: 1, b: '2'}"
        commands = parser.extract_all_commands(f"some text browse_eval({js}) trailing")
        assert len(commands) == 1
        assert commands[0] == ("browse_eval", [js])

    def test_browse_eval_regex_with_double_quote_in_char_class(self, parser):
        # Regex literals can contain a lone " inside a [...] character class.
        # Tracking it as a string delimiter would swallow the closing parens.
        js = 'return document.body.innerHTML.match(/autostarttoken[=:"\\s]*[a-f0-9-]{20,}/gi)'
        result, remainder = parser.extract_command_and_params(f"browse_eval({js})")
        assert result is not None
        assert result[0] == "browse_eval"
        assert result[1] == [js]
        assert remainder == ""
        assert parser.find_complete_command(f"browse_eval({js})") is True
        assert parser.find_complete_command(f"browse_eval({js})", streaming=True) is True

    def test_browse_eval_regex_with_capture_group(self, parser):
        # Regex with parens (capture group) — the parens are inside the regex
        # and must not affect outer paren depth.
        js = "return location.href.match(/token=([a-z0-9-]+)/i)"
        result, _ = parser.extract_command_and_params(f"browse_eval({js})")
        assert result is not None
        assert result[1] == [js]

    def test_browse_eval_division_not_treated_as_regex(self, parser):
        # `/` after an identifier is division, not regex — must keep tracking quotes.
        js = "return document.title.length / 2"
        result, _ = parser.extract_command_and_params(f"browse_eval({js})")
        assert result is not None
        assert result[1] == [js]


@pytest.mark.unit
class TestKeepBracketsCommands:
    """The config commands take setting keys whose [N] is a list index — the parser must
    not turn it into (N) like it does for Indigo device names."""

    @pytest.fixture
    def parser(self):
        return CommandParser(["config_set", "config_help", "config_request", "turn_on",
                              "browse_wait", "browse_network"])

    def test_config_set_keeps_single_quoted_index(self, parser):
        result, remainder = parser.extract_command_and_params(
            "config_set('icloud.devices[0].id', '00000000-1111-2222-3333-444444444444')")
        assert result == ("config_set", ["icloud.devices[0].id", "00000000-1111-2222-3333-444444444444"])
        assert remainder == ""

    def test_config_set_keeps_double_quoted_index(self, parser):
        result, _ = parser.extract_command_and_params('config_set("icloud.devices[1].display_name", "Anna\'s iPhone")')
        assert result[1] == ["icloud.devices[1].display_name", "Anna's iPhone"]

    def test_config_set_keeps_unquoted_index(self, parser):
        result, _ = parser.extract_command_and_params("config_set(icloud.devices[2].id, ABC)")
        assert result[1] == ["icloud.devices[2].id", "ABC"]

    def test_config_help_and_request_keep_index(self, parser):
        assert parser.extract_command_and_params('config_help("icloud.devices[0]")')[0][1] == ["icloud.devices[0]"]
        assert parser.extract_command_and_params("config_request(icloud.devices[0].id)")[0][1] == ["icloud.devices[0].id"]

    def test_batch_of_config_sets_all_keep_index(self, parser):
        text = ("config_set('icloud.devices[0].id', 'A')\n"
                "config_set('icloud.devices[0].display_name', 'Kitchen iPad')")
        commands = parser.extract_all_commands(text)
        assert commands == [("config_set", ["icloud.devices[0].id", "A"]),
                            ("config_set", ["icloud.devices[0].display_name", "Kitchen iPad"])]

    def test_browse_regex_arguments_keep_their_character_classes(self, parser):
        # The toolkit documents both arguments as regexes: a normalized "[Cc]omplete"
        # would become "(Cc)omplete" and silently never match.
        result, _ = parser.extract_command_and_params('browse_wait("[Cc]omplete|klar", 5)')
        assert result == ("browse_wait", ["[Cc]omplete|klar", "5"])
        result, _ = parser.extract_command_and_params('browse_network("/orders/[0-9]+")')
        assert result == ("browse_network", ["/orders/[0-9]+"])
        result, _ = parser.extract_command_and_params("browse_network(api/poll/[a-f0-9-]{36})")
        assert result == ("browse_network", ["api/poll/[a-f0-9-]{36}"])

    def test_other_commands_still_normalize_brackets(self, parser):
        result, _ = parser.extract_command_and_params("turn_on(Takspots [Gästhus])")
        assert result[1] == ["Takspots (Gästhus)"]
        result, _ = parser.extract_command_and_params('turn_on("Takspots [Gästhus]")')
        assert result[1] == ["Takspots (Gästhus)"]


@pytest.mark.unit
class TestSplitParameters:
    """Quote-aware tokenizer that powers parameter parsing.

    The split_parameters() helper handles three quoting forms (", ', `) when they
    appear at a parameter-start position (immediately after '(' or ','). Mid-token
    quotes are treated as literal characters so natural-language values like
    `I'm tired` survive without false-positive single-quote tracking.
    """

    @pytest.fixture
    def parser(self):
        return CommandParser([])

    def test_single_single_quoted(self, parser):
        assert parser.split_parameters("'bedroom'") == ["bedroom"]

    def test_single_backtick_quoted(self, parser):
        assert parser.split_parameters("`hallway`") == ["hallway"]

    def test_two_quoted_params_preserves_inner_comma(self, parser):
        # The headline use case: an arg whose value contains a comma must not
        # split the parameter list at that internal comma.
        assert parser.split_parameters('"John, Jr.", "Hi there"') == ["John, Jr.", "Hi there"]

    def test_escaped_double_quote_inside_double_quoted(self, parser):
        assert parser.split_parameters(r'"He said \"hi\"", "ok"') == ['He said "hi"', "ok"]

    def test_escaped_single_quote_inside_single_quoted(self, parser):
        assert parser.split_parameters(r"'it\'s late', 'now'") == ["it's late", "now"]

    def test_escaped_backslash(self, parser):
        assert parser.split_parameters(r'"path\\to\\file"') == [r"path\to\file"]

    def test_unknown_escape_passes_through(self, parser):
        # Backslash before a non-escapable char is left as-is — \n stays \n,
        # not an actual newline.
        assert parser.split_parameters(r'"line1\nline2"') == [r"line1\nline2"]

    def test_mixed_quoted_and_unquoted(self, parser):
        assert parser.split_parameters('unquoted, "quoted, with comma"') == ["unquoted", "quoted, with comma"]

    def test_apostrophe_in_unquoted_token_is_literal(self, parser):
        # Single quote not at param-start must NOT trigger quoted-mode, so
        # remember(I'm tired) parses as a single token.
        assert parser.split_parameters("I'm tired") == ["I'm tired"]

    def test_unquoted_url_passes_through(self, parser):
        # Backward compat: when the LLM forgets to quote, URL stays intact.
        assert parser.split_parameters("https://example.com/path") == ["https://example.com/path"]

    def test_url_after_param_name_prefix(self, parser):
        # `name: value` prefix is stripped per token by clean_parameter.
        assert parser.split_parameters("url: https://example.com/x") == ["https://example.com/x"]

    def test_param_name_quoted_value_strips_both(self, parser):
        # Legacy LLM pattern `name: "value"` — strip the prefix AND the quotes.
        assert parser.split_parameters('name: "value"') == ["value"]

    def test_empty_input(self, parser):
        assert parser.split_parameters("") == []
        assert parser.split_parameters("   ") == []

    def test_trailing_comma_yields_empty_token(self, parser):
        assert parser.split_parameters("a,") == ["a", ""]

    def test_double_comma_yields_empty_middle_token(self, parser):
        assert parser.split_parameters("a,,b") == ["a", "", "b"]

    def test_whitespace_around_quoted_values(self, parser):
        assert parser.split_parameters('  "a"  ,  "b"  ') == ["a", "b"]

    def test_quoted_value_brackets_normalized_to_parens(self, parser):
        assert parser.split_parameters('"Takspots [Gästhus]"') == [
            "Takspots (Gästhus)"
        ]

    def test_quoted_value_with_inner_parens_preserved(self, parser):
        # The original motivating example: open_url("https://x/page(2)").
        # extract_command_and_params end-to-end exercise.
        result, remainder = parser.extract_command_and_params(
            'open_url("https://example.com/page(2)") next'
        )
        assert result is not None
        assert result[1] == ["https://example.com/page(2)"]
        assert remainder == " next"

    def test_send_imessage_with_comma_in_first_arg(self, parser):
        # End-to-end: quoted recipient with embedded comma stays intact.
        result, _ = parser.extract_command_and_params(
            'send_imessage("John, Jr.", "Hi there")'
        )
        assert result is not None
        assert result[1] == ["John, Jr.", "Hi there"]

    def test_under_quoted_send_imessage_still_splits(self, parser):
        # Backward compat: when the LLM doesn't quote, the parser splits on
        # every comma — the dispatcher in Commands.run is responsible for
        # re-joining trailing tokens for body-style multi-param handlers.
        result, _ = parser.extract_command_and_params(
            "send_imessage(John, Hello, friend, how are you?)"
        )
        assert result is not None
        assert result[1] == ["John", "Hello", "friend", "how are you?"]


@pytest.mark.unit
class TestTripleQuotedArguments:
    """A long multi-line argument (a procedure's text) travels in a triple-quoted block:
    everything up to the closing triple quote is the value, whatever it contains."""

    @pytest.fixture
    def parser(self):
        return CommandParser(["save_procedure", "watch", "remember"])

    def test_parens_quotes_commas_and_list_markers_inside_do_not_end_the_command(self, parser):
        body = ('1) open shop.example (the cart)\n'
                '2) click "Pay", don\'t press Back)\n'
                "`fetch('/pay?id=' + id).then(r => r.text())`")
        content = f'save_procedure("shop-checkout", "paying at shop.example", """{body}""") Saved it.'
        (name, params), remainder = parser.extract_command_and_params(content)
        assert name == "save_procedure"
        assert params == ["shop-checkout", "paying at shop.example", body]
        assert remainder == " Saved it."

    def test_the_newlines_wrapping_the_block_are_not_part_of_the_value(self, parser):
        content = 'save_procedure("a", "b", """\nline one\nline two\n""")'
        (_, params), _ = parser.extract_command_and_params(content)
        assert params[2] == "line one\nline two"

    @pytest.mark.parametrize("streaming", [True, False])
    def test_an_unclosed_block_is_an_incomplete_command(self, parser, streaming):
        content = 'save_procedure("a", "b", """step one) and then'
        assert parser.extract_command_and_params(content, streaming=streaming) == (None, None)

    @pytest.mark.parametrize("streaming", [True, False])
    def test_streaming_completeness_check_waits_for_the_closing_triple_quote(self, parser, streaming):
        # The streaming path returns a bool; an unclosed block must not come back as a truthy sentinel.
        content = 'save_procedure("a", "b", """step one) and then'
        assert parser.find_complete_command(content, streaming=streaming) is False
        assert parser.find_complete_command(content + ' done""")', streaming=streaming) is True

    def test_a_block_starting_on_its_own_line_is_still_one_argument(self, parser):
        """Regression: only spaces were skipped before an argument, so a block after a line
        break was read as plain text and cut at its commas."""
        content = 'save_procedure("shop-checkout",\n    "paying",\n    """Paid, shipped (1)\nstep 2""",\n    "2026-11-08")'
        (_, params), remainder = parser.extract_command_and_params(content)
        assert params == ["shop-checkout", "paying", "Paid, shipped (1)\nstep 2", "2026-11-08"]
        assert remainder == ""

    def test_a_block_can_be_followed_by_another_argument(self, parser):
        (_, params), _ = parser.extract_command_and_params('watch("""Parcel (big), fragile""", "ordered", 2026-03-12)')
        assert params == ["Parcel (big), fragile", "ordered", "2026-03-12"]

    def test_an_ordinary_empty_string_argument_still_works(self, parser):
        (name, params), remainder = parser.extract_command_and_params('watch("Call the plumber", "", "")')
        assert (name, params, remainder) == ("watch", ["Call the plumber", "", ""], "")

    def test_split_parameters_reads_a_block_verbatim(self, parser):
        assert parser.split_parameters('"a", """x, "y" (z)"""') == ["a", 'x, "y" (z)']


@pytest.mark.unit
class TestRunDispatcher:
    """Commands.run dispatches list[str] → handler signatures.

    Multi-param handlers (in MULTI_PARAM_COMMANDS) get the args splatted as
    positional arguments. Single-param handlers receive a single joined string
    via params[0]. Under-quoted multi-param input gets trailing tokens re-joined
    by the handler itself.
    """

    @pytest.fixture
    def commands(self):
        from unittest.mock import Mock
        from assistant.commands.Commands import Commands
        return Commands(Mock())

    def test_single_param_dispatch_passes_first_token(self, commands):
        from unittest.mock import Mock
        commands.cmd_turn_on = Mock(return_value="")
        commands.run("turn_on", ["kitchen"])
        commands.cmd_turn_on.assert_called_once_with("kitchen")

    def test_single_param_dispatch_empty_list(self, commands):
        from unittest.mock import Mock
        commands.cmd_turn_on = Mock(return_value="")
        commands.run("turn_on", [])
        commands.cmd_turn_on.assert_called_once_with("")

    def test_multi_param_dispatch_splats(self, commands):
        from unittest.mock import Mock
        commands.cmd_dim = Mock(return_value="")
        commands.run("dim", ["lamp", "50"])
        commands.cmd_dim.assert_called_once_with("lamp", "50")

    def test_send_imessage_under_quoted_joins_body(self, commands):
        # LLM under-quoted: 4 tokens. cmd_send_imessage joins args[1:] as body.
        from unittest.mock import patch, Mock
        with patch("integrations.imessage.IMessageIntegration.IMessageIntegration") as mock_im:
            mock_inst = Mock()
            mock_inst.send.return_value = ""
            mock_im.get_instance.return_value = mock_inst
            commands.run("send_imessage", ["John", "Hello", "friend", "how are you?"])
            mock_inst.send.assert_called_once()
            recipient, body = mock_inst.send.call_args[0]
            assert recipient == "John"
            # Trailing tokens re-joined with ", " preserve the original phrasing.
            assert "Hello" in body and "friend" in body and "how are you?" in body

    def test_send_imessage_quoted_preserves_comma_in_first_arg(self, commands):
        from unittest.mock import patch, Mock
        with patch("integrations.imessage.IMessageIntegration.IMessageIntegration") as mock_im:
            mock_inst = Mock()
            mock_inst.send.return_value = ""
            mock_im.get_instance.return_value = mock_inst
            commands.run("send_imessage", ["John, Jr.", "Hi there"])
            recipient, body = mock_inst.send.call_args[0]
            assert recipient == "John, Jr."
            assert "Hi there" in body

    def test_dim_zero_rewrites_to_turn_off_with_single_arg(self, commands):
        from unittest.mock import Mock
        commands.cmd_turn_off = Mock(return_value="")
        commands.run("dim", ["kitchen", "0"])
        commands.cmd_turn_off.assert_called_once_with("kitchen")

    def test_unknown_command_returns_error_string(self, commands):
        result = commands.run("does_not_exist", ["whatever"])
        assert "Unknown command" in result
