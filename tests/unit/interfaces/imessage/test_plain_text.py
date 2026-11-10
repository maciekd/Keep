"""PlainText: a reply in the chat layer's HTML as the plain bubbles iMessage shows."""
import pytest

from interfaces.imessage import PlainText


@pytest.mark.unit
class TestRender:
    def test_the_dialects_tags_go_and_the_words_stay(self):
        assert PlainText.render("<b>Dinner</b> is at <i>seven</i>, <code>table 4</code>") == "Dinner is at seven, table 4"

    def test_entities_are_the_characters_they_stand_for(self):
        assert PlainText.render("Tom &amp; Jerry: 3 &lt; 5 &gt; 1") == "Tom & Jerry: 3 < 5 > 1"

    def test_a_link_keeps_its_address(self):
        assert PlainText.render('See <a href="https://menu.example/today">the menu</a>.') == \
            "See the menu (https://menu.example/today)."

    def test_a_link_shown_as_its_own_address_is_not_said_twice(self):
        assert PlainText.render('<a href="https://menu.example">https://menu.example</a>') == "https://menu.example"

    def test_a_code_block_keeps_its_lines(self):
        assert PlainText.render("Run:\n<pre>cd ~/Keep\n./start.sh</pre>") == "Run:\ncd ~/Keep\n./start.sh"

    def test_markdown_a_model_slipped_into_is_taken_out_as_well(self):
        text = "# Plan\n**First** the `oven`, then *wait* ~~an hour~~ ***ten minutes***.\n```python\nprint(1)\n```"
        assert PlainText.render(text) == "Plan\nFirst the oven, then wait an hour ten minutes.\nprint(1)"

    def test_a_markdown_link_keeps_its_address_like_any_link(self):
        assert PlainText.render("See [the menu](https://menu.example/today) or [https://x.example](https://x.example)") == \
            "See the menu (https://menu.example/today) or https://x.example"

    @pytest.mark.parametrize("text", [
        "the file is __init__.py in my_pkg__v2__final",
        "set __name__ and __main__, call foo__bar__baz()",
        "2**3**4 is big, and a ** b ** c too",
        "if a<b and c>d then x<y",
        "f(*args, **kwargs) and glob *.py or *.txt",
        "5*x + 3*y = 2 * z",
        "snake_case and _private_ names",
        "I <3 you, and 2 < 3 > 1",
        "a > b || c, then <script> and <think> are words here",
        "Dinner is at seven. See you there!",
        "- milk\n- eggs\n1. first\n2. second",
        "It costs 5 # per item (#1 seller)",
    ])
    def test_text_that_only_looks_like_markup_comes_through_untouched(self, text):
        assert PlainText.render(text) == text

    @pytest.mark.parametrize("code, shown", [
        ("<code>x__y__z</code>", "x__y__z"),
        ("<code>2**3**4</code>", "2**3**4"),
        ("`a_b**c**d`", "a_b**c**d"),
        ("<pre># not a heading\n**kwargs = {}\nif a &lt; b: pass</pre>", "# not a heading\n**kwargs = {}\nif a < b: pass"),
        ("```python\n# setup\nx = __name__\n```", "# setup\nx = __name__"),
        ('<pre><code class="language-sh">ls *.py</code></pre>', "ls *.py"),
    ], ids=["dunders in a span", "powers in a span", "a backtick span", "a block", "a fenced block", "a block with a language"])
    def test_code_is_never_read_as_markup(self, code, shown):
        assert PlainText.render(code) == shown
        assert PlainText.render(f"Run\n{code}\nnow").split() == f"Run {shown} now".split()

    def test_a_list_keeps_its_bullets_and_a_lone_asterisk_its_place(self):
        assert PlainText.render("• milk\n• eggs\n2 * 3 = 6") == "• milk\n• eggs\n2 * 3 = 6"

    def test_a_less_than_sign_that_is_no_tag_is_not_eaten(self):
        assert PlainText.render("I <3 you, and 2 < 3 > 1") == "I <3 you, and 2 < 3 > 1"

    def test_blocks_become_lines_and_list_items_lines_with_a_dash(self):
        assert PlainText.render("<p>One</p><ul><li>two</li><li>three</li></ul><br>four") == "One\n\n- two\n- three\nfour"
        assert PlainText.render("<h1>Menu</h1><p>Soup</p><p>Fish</p>") == "Menu\n\nSoup\n\nFish"
        assert PlainText.render("* milk\n* eggs") == "- milk\n- eggs"

    def test_html_a_model_reaches_for_outside_the_dialect_goes_too(self):
        text = '<table><tr><td>Mon</td><td>12°</td></tr><tr><td>Tue</td><td>9°</td></tr></table><img src="x.png"><ins>new</ins>'
        assert PlainText.render(text) == "Mon | 12°\nTue | 9°\nnew"

    def test_a_tag_with_attributes_is_still_a_tag(self):
        assert PlainText.render('<span class="tg-spoiler">the code</span> <a href=https://keep.example>here</a>') == \
            "the code here (https://keep.example)"

    @pytest.mark.parametrize("address", ["https://x.example/?q=1&region=SE&timestamp=17", "https://x.example/?a=1&copy=2&not=3",
                                         "https://x.example/?lt=1&gt=2&amp=3"],
                             ids=["region and timestamp", "copy and not", "the names of the escapes themselves"])
    def test_an_ampersand_in_an_address_is_not_the_start_of_a_character(self, address):
        """`&reg`, `&times`, `&copy` are characters to an HTML parser even without their
        semicolon; in a reply they are the parameters of a link."""
        assert PlainText.render(address) == address
        assert PlainText.render(f"<code>{address}</code>") == address
        assert PlainText.render(f'<a href="{address}">the page</a>') == f"the page ({address})"

    def test_a_link_that_is_never_closed_keeps_its_address(self):
        assert PlainText.render('See <a href="https://menu.example">the <b>menu</b> today\nand come') == \
            "See the menu today (https://menu.example)\nand come"

    def test_an_unclosed_link_does_not_swallow_the_next_one(self):
        rendered = PlainText.render('<ul><li><a href="https://a.example">one<li><a href="https://b.example">two</a></ul>')
        assert rendered.split("\n") == ["- one (https://a.example)", "- two (https://b.example)"]

    def test_an_attribute_html_writes_without_a_value_is_still_an_attribute(self):
        assert PlainText.render("<table><tr><td nowrap>Mon</td><td>12°</td></tr></table>") == "Mon | 12°"

    def test_code_keeps_its_indentation_and_its_blank_lines(self):
        code = "  def f():\n      return 1\n\n\n\n  x = f()   "
        assert PlainText.render(f"<pre>{code}</pre>") == code
        assert PlainText.render(f"Like this:\n<pre>{code}</pre>\nDone.") == f"Like this:\n{code}\nDone."
        assert PlainText.split(PlainText.render(f"<pre>{code}</pre>")) == [code.rstrip()], "the bubble lost what the render kept"

    def test_nothing_one_can_see_is_nothing(self):
        assert PlainText.render("&nbsp;&#8203;") == ""

    def test_runs_of_blank_lines_become_one(self):
        assert PlainText.render("One\n\n\n\n\nTwo  \n") == "One\n\nTwo"

    def test_nothing_but_markup_is_nothing(self):
        assert PlainText.render("<b> </b>\n") == "" and PlainText.render(None) == ""


@pytest.mark.unit
class TestSplit:
    def test_a_short_text_is_one_bubble(self):
        assert PlainText.split("On my way", 40) == ["On my way"]

    def test_a_long_text_is_cut_between_paragraphs(self):
        first, second = "a" * 30, "b" * 30
        assert PlainText.split(f"{first}\n\n{second}", 40) == [first, second]

    def test_then_between_lines_then_sentences_then_words(self):
        assert PlainText.split("first line here\nsecond line here ok", 25) == ["first line here", "second line here ok"]
        assert PlainText.split("It is done. The oven is off now.", 25) == ["It is done.", "The oven is off now."]
        assert PlainText.split("one two three four five six", 15) == ["one two three", "four five six"]

    def test_a_word_longer_than_a_bubble_is_cut_where_it_must_be(self):
        assert PlainText.split("x" * 50, 20) == ["x" * 20, "x" * 20, "x" * 10]

    def test_no_bubble_is_longer_than_the_limit_and_no_word_is_lost(self):
        text = " ".join(f"word{n}" for n in range(400)) + "\n\n" + "A sentence. " * 80
        bubbles = PlainText.split(text, 200)
        assert all(0 < len(bubble) <= 200 for bubble in bubbles)
        assert " ".join(bubbles).split() == text.split()

    @pytest.mark.parametrize("emoji", ["👨\u200d👩\u200d👧\u200d👦", "🇸🇪", "👍🏽", "❤\ufe0f", "e\u0301"],
                             ids=["a family", "a flag", "a skin tone", "a variation selector", "an accent"])
    def test_text_without_a_break_is_not_cut_inside_one_character(self, emoji):
        for lead in range(len(emoji) + 1):
            bubbles = PlainText.split("x" * (40 - lead) + emoji + "y" * 30, 40)
            assert sum(emoji in bubble for bubble in bubbles) == 1, f"cut inside it, {lead} characters before the limit"
            assert "".join(bubbles) == "x" * (40 - lead) + emoji + "y" * 30

    def test_a_bubble_that_begins_with_indented_code_keeps_the_indentation(self):
        text = "word " * 8 + "\n\n    indented = True\n    more = 1"
        assert PlainText.split(text, 45) == [("word " * 8).strip(), "    indented = True\n    more = 1"]

    def test_a_break_near_the_start_does_not_make_a_bubble_of_two_words(self):
        bubbles = PlainText.split("Hi there.\n\n" + "word " * 40, 100)
        assert len(bubbles[0]) > 25
