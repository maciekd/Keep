---
name: homeclaw
summary: "HomeKit through HomeClaw: covers, the garage door, irrigation, the thermostat, other lights, scenes, automations, what changed"
commands: [homekit_cover, homekit_garage, homekit_valve, homekit_thermostat, homekit_light, homekit_scene, homekit_events, homekit_automations]
---
The HOMEKIT (HOMECLAW) section of your prompt lists, room by room, the HomeKit accessories Keep's Hue and Plejd
integrations do not already bring, each line ending with its ref in brackets, e.g. `[A1B2C3D4]`. Several accessories
share a name, so pass the ref; a name works only when it is unique ("Name [Room]" narrows it to a room). A name that
fits several is refused with the candidates and their refs — pick one, never guess. Use this syntax exactly:

`homekit_cover(ref, position)` — a cover (awning, zip screen, blind): `open`, `close` or a position 0–100 (% open)
`homekit_garage(ref, open|close)` — the garage door; the ref may be left out when there is only one: `homekit_garage(open)`
`homekit_valve(ref, on|off)` — an irrigation valve or zone: on waters until turned off or the controller's own timer ends
`homekit_thermostat(ref, celsius)` — the thermostat's target temperature; the ref may be left out when there is one
`homekit_light(ref, on|off)` — one of the lights listed there (Hue and Plejd lights are in the home toolkit)
`homekit_scene(name)` — run a HomeKit scene, by the name listed under "Scenes"
`homekit_events(since)` — what changed in these accessories, oldest first; since is `30m`, `6h`, `2d` or an ISO time
(default `1h`). Sensor readings and light tweaks are not in it — the section has the current values.
`homekit_automations(name)` — the HomeKit automations: when each fires (a time, sunrise/sunset with offset, a button,
a sensor, people leaving or arriving), its conditions (weekdays, …), whether it is disabled and what it does, the
accessories grouped by setting ("off: A, B; on 30 %: C"). Without a name it lists them all; with one (part of a name)
it also spells out the named scenes they run. The section's "Automations" lists them a line each; call this for what
an automation actually does. Reading only — Keep cannot create, change, enable or disable automations.

Every command but the scene sends the change and waits for the accessory to confirm it; the reply says when the
accessory reports something else. Add `dry_run` as the last argument — `homekit_cover(A1B2C3D4, 40, dry_run)` — to
have HomeClaw check the request without acting; do that when unsure whether a value is accepted. Opening the garage
door and thermostat commands ask the user for approval first. A thermostat target outside the configured range is
refused and the reply names the range. When a command answers that HomeClaw is not running, tell the user the
HomeClaw app on the server's Mac must be started; do not retry.
