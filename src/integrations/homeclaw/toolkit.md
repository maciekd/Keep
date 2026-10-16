---
name: homeclaw
summary: "the HomeKit accessories HomeClaw lists: set the thermostat's target or mode or any other characteristic, and what changed in the house"
commands: [homeclaw_set, homeclaw_events]
---
### HomeKit accessories through HomeClaw
The HomeKit accessories the HomeClaw app sees — temperature, humidity and soil sensors, motion, the thermostat, covers,
the garage door, the irrigation valves, a few lights and the scenes — are in the HOME AUTOMATION DATA (the `home`
toolkit) with the rest of the house: switch, dim, open, close and stop them with the house commands (`turn_on`,
`turn_off`, `dim`, `stop`), run a scene with `execute`. A garage door opens with `turn_on` and closes with `turn_off`;
an irrigation valve starts watering with `turn_on` and stops with `turn_off`.

What those cannot express is written with (use this syntax exactly):

`homeclaw_set(name: str, characteristic: str, value: str)`

`name` is the accessory as listed, brackets included. `characteristic` is the HomeKit name in snake_case, `value` what
it should become:

- the thermostat: `target_temperature` in °C (`homeclaw_set("Tildes Element [Tildes Rum]", "target_temperature", "21")`),
  `target_heating_cooling` as `off`, `heat`, `cool` or `auto`
- a cover: `target_position`, 0 (closed) to 100 (open) — the same as `dim`
- a light: `brightness` 0–100, `color_temperature`, `hue`, `saturation`
- a garage door: `target_door_state` as `open` or `closed`

The write is verified against the accessory; the reply says the value it now reports, or why the app refused it (a
characteristic the accessory does not have, a value it cannot parse). It changes a setting on the device itself, so
confirm the value with the user first. The HOME AUTOMATION DATA shows the new state on its next refresh, within a
minute.

### What changed in the house
`homeclaw_events(hours: float)` — the house's event log for the last so many hours (24 when left out): doors
opened and closed, lights switched, covers moved, watering, motion, button presses, thermostat changes, with the time
of each. Sensor readings ticking and dimming steps are not in it. Use it for "did anyone open the garage", "when did
the watering run", "was there motion on the terrace tonight".

### Stop-and-wait
`homeclaw_set(...)` and `homeclaw_events(...)` return data: emit them and stop generating; the result arrives in the
next round.
