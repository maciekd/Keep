### The house
The HOME AUTOMATION DATA (sent with this reference, or in your system prompt) lists every light, action and variable
of the house with its state at the time it was sent. Values in square brackets are room names. Load this toolkit again
when you need the current state.

When the house runs on several home automation systems, a name that exists in more than one carries the system's name
as a last bracket, for example `Lamp [Kök] [Home Assistant]` — keep it, it says which system to send the command to.

### Light control commands
To manipulate lights use the following commands (use this syntax exactly): 

`turn_on(light_name: str)`
`turn_off(light_name: str)`
`dim(light_name: str, dim_percentage: int)`

Note that the dim level is not linear for many lights: 1% dim is clearly visible and 0% if light off. So a dim level
of 1% is often used to get nicely dimmed light. Do not ask if you should turn a lamp on before dimming it, just dim it. 
Dimming a lamp automatically turns it on. If you want to dim a lamp to 100% use the dim command, turn_on will just turn 
on the lamp with the last used dim level.

### Covers (blinds, zipscreens, awnings)
A cover is listed among the LIGHTS with its position as the value: `open`, `closed`, `40% open`, `opening` or
`closing`. Control it with the same commands — `turn_on` opens it fully, `turn_off` closes it, `dim(name, percentage)`
drives it to that position (100 is fully open, 0 closed) — and halt it on its way with:

`stop(cover_name: str)`

When the user refers to a room's lights collectively ("all the kitchen lights", "the bedroom lights"), first look
through the available ACTIONS for one that already does exactly that and prefer `execute(...)`-ing it. If there is no
matching action, apply the command to every light tagged with that room (each light whose name ends in the room tag,
e.g. `[Kök]`), not just a subset of them.

### Action execution command
To execute an action use the following syntax (use this syntax exactly):

`execute(action_name: str)`

Translate names of all lights, actions and variables if needed. But do not translate when using commands. When using
commands keep the names exactly as original, including the square brackets with the room name. REMEMBER to NEVER change
any part of the light or action parameter names.

NEVER user `turn_off` or `turn_on` to run actions, you must always use `execute` for actions. 

### Home automation logs
To fetch the home automation system's latest error logs use:

`home_automation_logs()`

### Stop-and-wait
`home_automation_logs()` returns data: emit it and stop generating; the result arrives in the next round.
