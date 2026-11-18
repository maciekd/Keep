---
name: ferry
summary: "the next departures of the user's car ferry route (Trafikverket, the route in config), either direction"
commands: [ferry_departures]
---
The car ferry's timetable comes live from Trafikverket, for the route set in config, with its notices about delays
and cancellations:

`ferry_departures(origin: str = "")` — the next departures from each side of the route, or only from the harbor
`origin` names (part of the harbor's name is enough). Departures within the hour come with the minutes left. When
the user asks when the next ferry leaves, give the first one or two times from the side they ask about — from home
unless they say otherwise — and any notice.

### Stop-and-wait
`ferry_departures` returns data: emit it and stop generating; the result arrives in the next round.
