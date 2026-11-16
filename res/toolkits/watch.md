The watch list (see "The watch list" in the prompt): `load_watch(name: str, ...)` loads an item in full — do that before you
answer a question about it or update it. The commands that change the list:

`save_watch(name: str, note: str, due: str = "", every: str = "")` — start watching something, or update the item with that
name. `note` is its current state with every detail you will need; it REPLACES the old note, so write the whole state,
not only what is new. `due` is the date it is due or expected, as YYYY-MM-DD, or the moment to act at as YYYY-MM-DD
HH:MM: a check wakes you at that minute with the item (at 09:00 of a date without a time), Night mode or not, and the
item stays among the due items of every check until you look at it — `checked_watch`, a `save_watch` with a new due or `delete_watch`; optional. `every`
is how often the item must be checked ("1d", "6h", "20m"); optional — leave it out for an item that only waits for its
due date. Cadence checks pause while Night mode is on and resume in the morning; a due time is the way to be woken
at a clock time during the night. The note runs to as many lines as it needs, up to 500 words — a longer one
is refused:
`save_watch("sofa-delivery", """...""", "2026-11-08")`, `save_watch("water-meter", """...""", "2026-10-17 17:00")`,
`save_watch("festival-tickets", """...""", "", "1d")`.

`checked_watch(name: str)` — you checked the item and nothing changed. It only records the time of the check; the
note stays as it is.

`delete_watch(name: str)` — it is done, delivered, paid, cancelled or no longer relevant.

An autonomous check is given the items that are due — by their due time or their cadence — in full, with when
each was last checked. Delete an item as soon as it is settled.
