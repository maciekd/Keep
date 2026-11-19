import re


class CommandParser:
    """Recognises and parses `command(params)` invocations in the LLM's streamed reply text.

    Where a call stands decides whether it is one: a call starts its line — behind nothing but
    whitespace and the backticks or `<code>` tag a model wraps it in — or follows another
    call on the same line. A `name(…)` anywhere else is the model talking about the command
    and stays reply text. On the production server, June to November 2026, 16,714 of 17,003
    calls started their line and 12 followed another call; not one of the 243 written inside
    a sentence was meant to run, and about 45 of those ran (a `restart_server()` out of
    "Want me to run restart_server() after you've edited the file?"). A list bullet in front
    is a sentence too: all 29 calls behind one were mentions.
    """
    COMMAND_PARAMETERS_START = "("
    COMMAND_PARAMETERS_END = ")"
    # What may stand between the start of a line, or the call before, and a call.
    _LEAD_RE = re.compile(r"(?:\s|`|</?code>)*")
    _CODE_OPEN = "<code>"

    # Commands whose argument is opaque source text (e.g. JavaScript) that must
    # not be normalized — clean_parameters would strip quotes/brackets/backticks
    # and mangle name: prefixes that are valid syntax in the payload.
    RAW_PARAM_COMMANDS = {"browse_eval"}

    # Commands whose one argument is free prose for another agent. The whole payload is
    # the argument, verbatim: splitting it on commas handed self_improvement only the
    # words before the request's first comma (prod, 2026-10-08: "Maciek asked at 10:46 on
    # Oct 8" was all Claude Code got). Unlike RAW_PARAM_COMMANDS the text is not JS, so
    # an apostrophe in it must not open a string for the paren matching.
    # improve_integration and improve_local_integration split their leading integration name
    # off the payload themselves.
    FREE_TEXT_COMMANDS = {"self_improvement", "self_analysis", "improve_integration", "improve_local_integration",
                          "report_issue"}

    # Commands whose arguments carry square brackets as syntax — a setting key's
    # list index (icloud.devices[0].id) or a regex character class ([Cc]omplete,
    # /orders/[0-9]+) — and must not be normalized to parens like the home
    # automation names every other command carries.
    # A mail folder is named as the server names it: Gmail's own are "[Gmail]/Sent Mail",
    # "[Gmail]/All Mail" … (prod 2026-11-17: the Sent folder was asked for as "(Gmail)/Sent Mail").
    KEEP_BRACKETS_COMMANDS = {"config_set", "config_help", "config_request", "config_delete", "browse_wait", "browse_network",
                              "latest_email", "move_email"}

    def __init__(self, possible_commands: list[str]):
        self.possible_commands = possible_commands
        self._names_of: tuple[str, ...] | None = None
        self._names_re: re.Pattern | None = None

    @staticmethod
    def _strip_code_tags(content: str) -> str:
        """Strip <code> and </code> tags that the LLM may wrap commands in."""
        return content.replace("<code>", "").replace("</code>", "")

    def is_possibly_command(self, content: str, mid_line: bool = False) -> bool:
        """True while `content` is, or may still grow into, a call. With `mid_line` it
        continues a line that already holds a sentence, where no call starts before the
        next line break."""
        lead = content[:len(content) - len(content.lstrip())]
        content = content.strip()
        if len(content) <= 0:
            return True
        if mid_line and "\n" not in lead:
            return False
        content_lower = content.lower()
        if self._CODE_OPEN.startswith(content_lower):
            return True
        stripped_lower = self._strip_code_tags(content_lower)
        for c in self.possible_commands:
            c += self.COMMAND_PARAMETERS_START
            if c.startswith(content_lower) or content_lower.startswith(c):
                return True
            if stripped_lower != content_lower:
                if c.startswith(stripped_lower) or stripped_lower.startswith(c):
                    return True
        return False

    def unterminated_command(self, content: str) -> str | None:
        """The name of the command `content` opens but never closes — a whole known name
        followed by '(' — or None. A bare name or a name prefix is a mention (the prompt
        tells the LLM to leave the parentheses out when it only talks about a command),
        never a call, so it does not count."""
        stripped_lower = self._strip_code_tags(content).strip().lower()
        matches = [c for c in self.possible_commands
                   if stripped_lower.startswith(c + self.COMMAND_PARAMETERS_START)]
        return max(matches, key=len) if matches else None

    @staticmethod
    def _is_list_marker_paren(content: str, pos: int, streaming: bool = False) -> bool:
        """Check if ')' at pos is part of a numbered list marker like '1)', '2)'.
        Returns True when digit(s) before ')' are preceded by a list-introducing
        context: newline, colon/semicolon, or sentence-ending punctuation.
        For inline markers (not after newline), also requires content after the ')'
        to distinguish '1) item text' from 'param: 75)' at end of command.
        When streaming=True and there's no content after ')' yet (ambiguous),
        assumes it's a list marker to avoid premature command closing.
        A colon preceded by a single identifier right after '(' or ',' is treated
        as parameter assignment (e.g. 'name: value') rather than list context."""
        if pos == 0:
            return False
        digit_start = pos - 1
        while digit_start >= 0 and content[digit_start].isdigit():
            digit_start -= 1
        if digit_start == pos - 1:
            return False
        check_pos = digit_start
        while check_pos >= 0 and content[check_pos] in (' ', '\t'):
            check_pos -= 1
        if check_pos < 0:
            return True
        ch = content[check_pos]
        if ch in ('\n', '\r'):
            return True
        if ch in (':', ';', '.', '?', '!'):
            # "57.5501633)" is a decimal, not a sentence end followed by a list number.
            if ch == '.' and check_pos > 0 and content[check_pos - 1].isdigit():
                return False
            if ch == ':':
                p = check_pos - 1
                while p >= 0 and content[p] in (' ', '\t'):
                    p -= 1
                id_end = p
                while p >= 0 and (content[p].isalnum() or content[p] == '_'):
                    p -= 1
                if id_end > p:
                    identifier = content[p + 1:id_end + 1]
                    # When identifier before ':' is purely numeric (e.g. "12" in
                    # "end_time: 12:00"), look further back for a named parameter
                    # context — "param: digits:digits)" is a time value, not list context.
                    if identifier.isdigit():
                        if CommandParser._is_bare_datetime_value(content, id_end):
                            return False
                        q = p
                        while q >= 0 and content[q] in (' ', '\t'):
                            q -= 1
                        if q >= 0 and content[q] in (':', '='):
                            q -= 1
                            while q >= 0 and content[q] in (' ', '\t'):
                                q -= 1
                            name_end = q
                            while q >= 0 and (content[q].isalnum() or content[q] == '_'):
                                q -= 1
                            if name_end > q:
                                while q >= 0 and content[q] in (' ', '\t'):
                                    q -= 1
                                if q < 0 or content[q] in ('(', ','):
                                    return False
                    while p >= 0 and content[p] in (' ', '\t'):
                        p -= 1
                    if p < 0 or content[p] in ('(', ','):
                        return False
            stripped = content[pos + 1:].lstrip()
            if streaming and len(stripped) == 0:
                return True
            return len(stripped) > 0 and stripped[0] != ')'
        return False

    @staticmethod
    def _is_bare_datetime_value(content: str, end: int) -> bool:
        """True when the parameter segment ending at `end` (the last char before the
        time's ':') holds no letters — a positional date/time value like
        '2026-09-07 23:59' rather than prose with a time in it ('Meeting at 12:00').
        The segment runs back to the enclosing '(' or the previous ','."""
        q = end
        while q >= 0 and content[q] not in ('(', ','):
            if content[q].isalpha():
                return False
            q -= 1
        return q >= 0

    @staticmethod
    def _is_list_continuation(content: str, pos: int) -> bool:
        """Check if ')' at pos continues an already-identified numbered list.
        Returns True when digits precede ')' and non-empty content follows,
        indicating '2) next item' rather than a command terminator."""
        if pos == 0:
            return False
        digit_start = pos - 1
        while digit_start >= 0 and content[digit_start].isdigit():
            digit_start -= 1
        if digit_start == pos - 1:
            return False
        stripped = content[pos + 1:].lstrip()
        return len(stripped) > 0 and stripped[0] != ')'

    def _command_uses_raw_params(self, content: str) -> bool:
        """True when content begins with a RAW_PARAM_COMMANDS command (e.g. browse_eval).
        Single-quoted strings in those commands' JS args must be paren-skipped too."""
        paren_idx = content.find(self.COMMAND_PARAMETERS_START)
        if paren_idx <= 0:
            return False
        name = self._strip_code_tags(content[:paren_idx]).strip().lower()
        return name in self.RAW_PARAM_COMMANDS

    # JS keywords that allow a regex literal to follow them (rather than division).
    _REGEX_PRECEDING_KEYWORDS = {
        "return", "typeof", "in", "of", "new", "delete", "throw", "void",
        "else", "do", "instanceof", "yield", "await", "case",
    }

    @classmethod
    def _is_regex_start(cls, content: str, pos: int) -> bool:
        """Heuristic: is the '/' at pos the start of a JS regex literal (vs division)?
        Regex follows expression-start contexts: openers, operators, commas, or certain
        keywords (return, typeof, ...). Division follows identifiers, numbers, or ')', ']'."""
        p = pos - 1
        while p >= 0 and content[p] in ' \t\n\r':
            p -= 1
        if p < 0:
            return True
        prev = content[p]
        if prev.isalnum() or prev == '_' or prev == ')' or prev == ']':
            end = p
            while p >= 0 and (content[p].isalnum() or content[p] == '_'):
                p -= 1
            ident = content[p + 1:end + 1]
            return ident in cls._REGEX_PRECEDING_KEYWORDS
        return True

    @staticmethod
    def _skip_regex_literal(content: str, pos: int) -> int:
        """Given pos pointing at the opening '/' of a regex literal, return the index
        right after the closing '/' (and any trailing flag chars). Handles escaped
        chars and character classes [...]. If unterminated, returns len(content)."""
        n = len(content)
        i = pos + 1
        in_class = False
        while i < n:
            ch = content[i]
            if ch == '\\' and i + 1 < n:
                i += 2
                continue
            if ch == '[' and not in_class:
                in_class = True
            elif ch == ']' and in_class:
                in_class = False
            elif ch == '/' and not in_class:
                i += 1
                while i < n and content[i].isalpha():
                    i += 1
                return i
            elif ch == '\n':
                return i
            i += 1
        return n

    def _find_matching_paren(self, content: str, streaming: bool = False) -> int:
        """Find index of the closing ')' that matches the first '('.
        Parentheses inside double-quoted strings are ignored. For commands whose
        argument is opaque source text (RAW_PARAM_COMMANDS, e.g. browse_eval),
        single-quoted strings AND JS regex literals are also paren-skipped — JS args
        routinely contain unmatched quotes/parens inside `'...'` strings or `/.../` regexes."""
        start = content.find(self.COMMAND_PARAMETERS_START)
        if start < 0:
            return -1
        track_single = self._command_uses_raw_params(content)
        depth = 0
        in_double_quote = False
        in_single_quote = False
        list_markers_at_depth1 = 0
        n = len(content)
        i = start
        while i < n:
            ch = content[i]
            if track_single and not in_double_quote and not in_single_quote and ch == '/' and self._is_regex_start(content, i):
                i = self._skip_regex_literal(content, i)
                continue
            if not in_double_quote and not in_single_quote and content.startswith(self._BLOCK_QUOTE, i):
                # A triple-quoted block is opaque up to its closing triple quote; unclosed,
                # the command is not complete yet.
                end = content.find(self._BLOCK_QUOTE, i + 3)
                if end < 0:
                    return -1
                i = end + 3
                continue
            if ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif ch == "'" and track_single and not in_double_quote:
                in_single_quote = not in_single_quote
            elif not in_double_quote and not in_single_quote:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    if depth == 1:
                        if self._is_list_marker_paren(content, i, streaming):
                            list_markers_at_depth1 += 1
                            i += 1
                            continue
                        if list_markers_at_depth1 > 0 and self._is_list_continuation(content, i):
                            list_markers_at_depth1 += 1
                            i += 1
                            continue
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return -1

    def extract_command_and_params(self, content: str, streaming: bool = False) -> tuple[tuple[str, list[str]] | None, str | None]:
        """
        Extract command name and parameters from content.
        Returns ((command_name, params_list), remainder) or (None, None) if not a complete command.
        params_list is a list of cleaned argument strings (one entry per comma-separated parameter).
        For RAW_PARAM_COMMANDS and FREE_TEXT_COMMANDS the list always contains a single element
        with the verbatim payload.
        """
        call = self._first_call(content, streaming)
        if call is None:
            return None, None
        command_name, raw_params, remainder = call
        params, _ = self._scan(raw_params, self._is_verbatim(command_name),
                               command_name in self.KEEP_BRACKETS_COMMANDS)
        return (command_name, params), remainder

    def text_after_quote(self, content: str, blocks_only: bool = False) -> str | None:
        """What the first call of the content carries between the closing quote of a value and
        the next comma, or None when every quoted value ends where its quote closes. Such text
        is code written around the value (prod 2026-11-17: a mail body sent as
        a triple-quoted block, `.replace(…) if False else` and a second block, went out garbled) or a string whose own
        quotes were not escaped: either way the value the call would run with is not the one
        that was meant. With `blocks_only` only a triple-quoted block is looked at — after a
        plain string the text is glued to the value, as it always was."""
        call = self._first_call(content)
        if call is None or self._is_verbatim(call[0]):
            return None
        _, strays = self._scan(call[1])
        return next((text for text, after_block in strays if after_block or not blocks_only), None)

    def _is_verbatim(self, command_name: str) -> bool:
        return command_name in self.RAW_PARAM_COMMANDS or command_name in self.FREE_TEXT_COMMANDS

    def _first_call(self, content: str, streaming: bool = False) -> tuple[str, str, str] | None:
        """The first complete call of the content as (command name, the raw text between its
        parentheses, the text after it), or None when there is none."""
        # Strip <code> tags so commands wrapped in them are parsed correctly
        content = self._strip_code_tags(content)

        index = self._find_matching_paren(content, streaming)
        if index <= 0:
            return None

        command_with_params = content[:index + 1]
        command_name = command_with_params.split('(', 1)[0].lower().strip()
        return command_name, self.extract_parameters(command_with_params), content[index + 1:]

    def extract_parameters(self, command: str) -> str:
        """Extract raw parameters from command string."""
        start = command.find(self.COMMAND_PARAMETERS_START)
        end = command.rfind(self.COMMAND_PARAMETERS_END)
        if 0 <= start < end:
            return command[start + 1:end]
        return ""

    # Characters recognized as quote delimiters when they appear at the start of
    # a parameter (immediately after '(' or ',', possibly preceded by whitespace).
    # Mid-token occurrences are treated as literal characters to avoid breaking
    # natural-language parameters like remember(I'm tired).
    _QUOTE_CHARS = ('"', "'", "`")
    # Python's triple quote, for a long multi-line argument (a procedure's text): everything
    # up to the closing triple quote is the value, verbatim — quotes, commas and parentheses
    # inside need no escaping.
    _BLOCK_QUOTE = '"""'

    @classmethod
    def split_parameters(cls, raw: str, raw_param_command: bool = False,
                         keep_brackets: bool = False) -> list[str]:
        """Split raw parameter text into a list of cleaned argument strings.

        Quote-aware: when a parameter starts with ", ', or `, the matching closing
        quote delimits the argument (commas, parentheses, etc. inside are part of
        the value). Inside a quoted argument, escape sequences \\", \\', \\`, and
        \\\\ are resolved to their literal characters; other backslash escapes
        pass through as-is.

        Unquoted arguments fall through to clean_parameter() so that legacy LLM
        output (param_name: value, type=value, etc.) keeps working.

        For RAW_PARAM_COMMANDS the entire payload is returned as a single-element
        list with no cleaning applied. With keep_brackets (KEEP_BRACKETS_COMMANDS)
        square brackets stay as written instead of becoming parentheses.
        """
        return cls._scan(raw, raw_param_command, keep_brackets)[0]

    @classmethod
    def _scan(cls, raw: str, raw_param_command: bool = False,
              keep_brackets: bool = False) -> tuple[list[str], list[tuple[str, bool]]]:
        """The arguments of `split_parameters`, and beside them every text found between a
        closing quote and the next comma, each as (the text, whether a triple-quoted block
        closed before it)."""
        if raw_param_command:
            return [raw], []

        if raw.strip() == "":
            return [], []

        tokens: list[str] = []
        strays: list[tuple[str, bool]] = []
        n = len(raw)
        i = 0
        while i < n:
            # Skip leading whitespace at the start of each parameter slot — a line break too,
            # so a triple-quoted block that starts on its own line is still seen as one
            while i < n and raw[i] in ' \t\r\n':
                i += 1
            if i >= n:
                tokens.append("")
                break

            ch = raw[i]
            if raw.startswith(cls._BLOCK_QUOTE, i):
                end = raw.find(cls._BLOCK_QUOTE, i + 3)
                end = n if end < 0 else end
                tokens.append(raw[i + 3:end].strip("\n"))
                i = after = min(end + 3, n)
                while i < n and raw[i] != ',':
                    i += 1
                if raw[after:i].strip():
                    strays.append((raw[after:i].strip(), True))
            elif ch in cls._QUOTE_CHARS:
                quote = ch
                i += 1
                buf: list[str] = []
                while i < n:
                    c = raw[i]
                    if c == '\\' and i + 1 < n and raw[i + 1] in ('"', "'", '`', '\\'):
                        buf.append(raw[i + 1])
                        i += 2
                        continue
                    if c == quote:
                        i += 1
                        break
                    buf.append(c)
                    i += 1
                # Skip whitespace right after the closing quote.
                while i < n and raw[i] in ' \t\r\n':
                    i += 1
                # Tolerate stray non-whitespace text between the closing quote
                # and the next comma — append it to the value so nothing is
                # silently lost (e.g. "abc"def parses as "abcdef").
                after = i
                while i < n and raw[i] != ',':
                    buf.append(raw[i])
                    i += 1
                if after < i:
                    strays.append((raw[after:i].strip(), False))
                # Normalize square brackets to parens so room-name suffixes like
                # "Ceiling Spots [Garage]" match the parser's canonical form
                # whether or not the LLM wrapped the value in quotes.
                value = ''.join(buf)
                if not keep_brackets:
                    value = value.replace("[", "(").replace("]", ")")
                tokens.append(value)
            else:
                buf = []
                while i < n and raw[i] != ',':
                    buf.append(raw[i])
                    i += 1
                tokens.append(cls.clean_parameter(''.join(buf), keep_brackets=keep_brackets))

            if i < n and raw[i] == ',':
                i += 1
                # If the comma was the final char, an empty trailing token is implied.
                if i >= n:
                    tokens.append("")
        return tokens, strays

    @classmethod
    def clean_parameter(cls, token: str, keep_brackets: bool = False) -> str:
        """Clean a single unquoted parameter token.

        Strips a leading 'name:' / 'name=' / 'str=' / 'int=' prefix (the LLM
        sometimes emits parameter names), strips matching outer quotes around
        the resulting value (so `name: "value"` becomes `value`), replaces
        square brackets with parens (unless keep_brackets), and trims whitespace. URL schemes
        (https:, http:, ftp:) and music ID prefixes (F-1, P-0) are protected
        from the prefix stripper.
        """
        token = token.strip()
        # Strip leading param-name prefix: "param: value" -> "value", "param=value" -> "value"
        token = re.sub(
            r'^(?![FP]-\d)(?!https?:|ftp:)[a-z_][a-z0-9_]*\s*[:=]\s*',
            '',
            token,
            count=1,
            flags=re.IGNORECASE,
        )
        # Strip type annotations like str="value" or int=123
        token = re.sub(r'\b(str|int)[=:]', '', token, flags=re.IGNORECASE)
        token = token.strip()
        # If what remains is wrapped in matching outer quotes, peel them off
        # and resolve common escape sequences inside.
        if len(token) >= 2 and token[0] in cls._QUOTE_CHARS and token[-1] == token[0]:
            inner = token[1:-1]
            inner = (inner.replace('\\\\', '\x00')
                          .replace('\\"', '"')
                          .replace("\\'", "'")
                          .replace('\\`', '`')
                          .replace('\x00', '\\'))
            token = inner
        if not keep_brackets:
            token = token.replace("[", "(").replace("]", ")")
        return token.strip()

    @classmethod
    def clean_parameters(cls, params: str) -> str:
        """Back-compat shim: tokenize quote-aware, clean each token, re-join with ', '.

        Most callers should use split_parameters() directly to preserve parameter
        boundaries (especially for multi-arg commands with embedded commas in
        quoted values). This shim exists for tests and any external callers that
        still want the legacy single-string view.
        """
        return ", ".join(cls.split_parameters(params))

    def find_complete_command(self, content: str, streaming: bool = False) -> bool:
        """
        Check if content contains a complete command with balanced parentheses.
        Parentheses inside double-quoted strings are ignored. For RAW_PARAM_COMMANDS
        (e.g. browse_eval), parens inside single-quoted strings AND JS regex literals
        are also ignored.
        """
        content = self._strip_code_tags(content)
        track_single = self._command_uses_raw_params(content)
        open_count = 0
        close_count = 0
        in_double_quote = False
        in_single_quote = False
        list_markers_at_depth1 = 0
        n = len(content)
        i = 0
        while i < n:
            ch = content[i]
            if track_single and not in_double_quote and not in_single_quote and ch == '/' and self._is_regex_start(content, i):
                i = self._skip_regex_literal(content, i)
                continue
            if not in_double_quote and not in_single_quote and content.startswith(self._BLOCK_QUOTE, i):
                # A triple-quoted block is opaque up to its closing triple quote; unclosed,
                # the command is not complete yet.
                end = content.find(self._BLOCK_QUOTE, i + 3)
                if end < 0:
                    return False
                i = end + 3
                continue
            if ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif ch == "'" and track_single and not in_double_quote:
                in_single_quote = not in_single_quote
            elif not in_double_quote and not in_single_quote:
                if ch == '(':
                    open_count += 1
                elif ch == ')':
                    if open_count - close_count == 1:
                        if self._is_list_marker_paren(content, i, streaming):
                            list_markers_at_depth1 += 1
                            i += 1
                            continue
                        if list_markers_at_depth1 > 0 and self._is_list_continuation(content, i):
                            list_markers_at_depth1 += 1
                            i += 1
                            continue
                    close_count += 1
            i += 1
        return open_count > 0 and close_count >= open_count

    def _names(self) -> re.Pattern:
        """Every `name(` of the known commands, the longest name first where two start alike."""
        names = tuple(self.possible_commands)
        if names != self._names_of:
            by_length = sorted(names, key=len, reverse=True)
            self._names_re = re.compile("(?:" + "|".join(re.escape(name.lower()) for name in by_length) + r")\(")
            self._names_of = names
        return self._names_re

    # A reply as received still holds its thinking blocks, and the text behind one begins
    # the reply as the stream showed it: the block's end counts as a line's end.
    _LINE_ENDS = ("\n", "</thinking>", "</think>")

    @classmethod
    def _starts_line(cls, content: str, pos: int, mid_line: bool) -> bool:
        """Whether a call may start at `pos`: nothing but its wrapping stands before it on its
        line. The content's first line is the rest of a sentence when `mid_line`."""
        line = max(at + len(end) if (at := content.rfind(end, 0, pos)) >= 0 else 0 for end in cls._LINE_ENDS)
        return (line > 0 or not mid_line) and cls._LEAD_RE.fullmatch(content, line, pos) is not None

    @classmethod
    def mid_line_after(cls, text: str, mid_line: bool) -> bool:
        """Whether reply text that was `mid_line` before `text` is inside a sentence after it."""
        line = text.rfind("\n") + 1
        return (mid_line and line == 0) or cls._LEAD_RE.fullmatch(text, line) is None

    def find_command_start(self, content: str, mid_line: bool = False) -> int:
        """The position of the earliest `command_name(` that starts its line, or -1. A text
        that begins where a call ended begins on a clear line, so a walk that goes on behind
        each call it reads finds the calls that follow one another on a line."""
        if not self.possible_commands:
            return -1
        for match in self._names().finditer(content.lower()):
            if self._starts_line(content, match.start(), mid_line):
                return match.start()
        return -1

    def call_end(self, content: str) -> int:
        """Where the text after the first call of `content` begins, in the text as written,
        or -1 without a complete call. The parser drops every code tag before it looks, so
        the length of its rest counts the call's characters and the tags are stepped over."""
        call, rest = self.extract_command_and_params(content)
        if call is None:
            return -1
        call_chars = len(self._strip_code_tags(content)) - len(rest)
        position = 0
        while call_chars > 0 and position < len(content):
            tag = next((t for t in ("<code>", "</code>") if content.startswith(t, position)), None)
            if tag:
                position += len(tag)
            else:
                position += 1
                call_chars -= 1
        return position

    def calls_in_prose(self, content: str) -> list[tuple[str, list[str]]]:
        """The complete calls of a reply that stand where no call runs — inside a sentence or
        behind a list bullet — as (name, params), in order. One that is part of other code
        (`window.stop()`, `def stop(self)`) or runs over its line's end is no call anybody
        wrote, and what a call that did run holds between its parentheses is its value."""
        found: list[tuple[str, list[str]]] = []
        if not self.possible_commands:
            return found
        names, lower = self._names(), content.lower()
        at, mid_line = 0, False
        while match := names.search(lower, at):
            pos = match.start()
            end = self.call_end(content[pos:])
            if self._starts_line(content[at:], pos - at, mid_line):
                at, mid_line = (pos + end, False) if end > 0 else (pos + 1, True)
                continue
            written = end > 0 and "\n" not in content[pos:pos + end]
            before = content[max(pos - 2, 0):pos]
            glued = before[-1:].isalnum() or before[-1:] == "." or (before[-1:] == "_" and before[:-1].isalnum())
            if written and not glued:
                found.append(self.extract_command_and_params(content[pos:])[0])
            at, mid_line = (pos + end if written else match.end()), True
        return found

    # A line that is a call and nothing else: a lowercase name, its parentheses (a
    # triple-quoted value may run over lines), the line's end.
    _CALL_LINE_RE = re.compile(r'^[ \t]*([a-z_][a-z0-9_]*)\((?:"""[\s\S]*?"""|[^\n])*\)[ \t]*$', re.M)

    def unknown_calls(self, content: str) -> list[tuple[str, str]]:
        """The lines of a complete reply that are a call of something that is no command, as
        (name, the line as written), in order. Opus 5.5 loses the head of a call now and then:
        `_act("35")` for `browse_act("35")`, `_load_toolkit("web")` — seven replies on the
        production server between 2026-10-20 and 2026-11-18, none before. A name with a dot
        (`console.log(…)`) or a capital is code or prose, not a call somebody meant, and so
        is a plain word (`cos(x)`, `print(total)`): only a name shaped like a command's — an
        underscore in it, or the last words of a real one (`eval` of `browse_eval`) — counts."""
        names = {name.lower() for name in self.possible_commands}
        return [(match.group(1), match.group(0).strip()) for match in self._CALL_LINE_RE.finditer(content)
                if match.group(1) not in names and self._command_shaped(match.group(1), names)]

    @staticmethod
    def _command_shaped(name: str, names: set[str]) -> bool:
        return "_" in name or any(command.endswith("_" + name) for command in names)

    def find_partial_command_tail(self, content: str, mid_line: bool = False) -> int:
        """Find position where the tail of content is a prefix of a known command name, or of
        the `<code>` tag a call may be wrapped in. Returns the start position of the partial
        match, or -1 if none found.
        Used during streaming to avoid flushing a partial command name embedded at the buffer's end.
        Only matches where a call may start (`_starts_line`): a word inside a sentence is
        never held, whatever command name it begins like."""
        content_lower = content.lower()
        best_pos = -1
        for cmd in (*self.possible_commands, self._CODE_OPEN):
            for length in range(min(len(cmd), len(content_lower)), 0, -1):
                if content_lower.endswith(cmd[:length]):
                    pos = len(content_lower) - length
                    if not self._starts_line(content, pos, mid_line):
                        break
                    if pos > 0 and (best_pos == -1 or pos < best_pos):
                        best_pos = pos
                    break
        return best_pos

    def extract_all_commands(self, content: str) -> list[tuple[str, list[str]]]:
        """Extract every command from a complete string, returning list of (command_name, params)."""
        commands = []
        remaining = self._strip_code_tags(content.replace("`", ""))
        while remaining:
            pos = self.find_command_start(remaining)
            if pos < 0:
                break
            result, extra = self.extract_command_and_params(remaining[pos:])
            if result is not None:
                commands.append(result)
                remaining = extra or ""
            else:
                remaining = remaining[pos + 1:]
        return commands
