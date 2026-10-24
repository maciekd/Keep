---
name: location
summary: "make a family member's phone ring, find where they are, set up the phones"
commands: [find_phone, locate_user, icloud_devices, icloud_unlocked]
---
### Find My iPhone
To make a family member's phone play a sound (useful when someone can't find their phone):

`find_phone(device_name: str)`

Available devices: see the FAMILY PHONES section of your prompt. The command answers at once and the phone rings in
the background — a sign-in iCloud asks for first, its 2FA code included, is completed there too, so the phone rings
within the minute. Say the ping is on its way, not that it is done; a ring that failed arrives as a system note.

### Fetch user location
If someone is not at home, you can use Find My iPhone to fetch the approximate location of their phone. Use this if the user asks where someone is.

`locate_user(device_name: str)`

Available users: the same names as in FAMILY PHONES.

Returns a human-readable place plus GPS coordinates and accuracy.

### Setting up the devices
The device names in FAMILY PHONES come from the icloud.devices list in the config, one entry per device with the keys `id`
(the device's iCloud id) and `display_name` (the name people will use, e.g. "dad"). When it says none are
configured, or a name is missing, list the devices:

`icloud_devices()` — every device the iCloud account sees: name, model, id, whether it is configured, and the index
the next entry gets.

Then ask the user which device belongs to whom and what
to call it, and write each entry with `config_set("icloud.devices[N].id", "<id>")` and
`config_set("icloud.devices[N].display_name", "<name>")`, then `restart_integration("iCloud")`. A 2FA prompt or an
empty listing means the iCloud login is not complete — see "The 2FA code" below. When
`find_phone` / `locate_user` report a configured device as not found, its id has changed (a phone re-enrolled in
Find My): run `icloud_devices` and point the entry at the new id.

`config_set`, `restart_integration` and `icloud_login_code` are in the `setup` toolkit.

### The 2FA code
A sign-in with the password ends with Apple texting a 6-digit code to the account's trusted phone, and a system note
says whether the server reads it by itself (the iMessage feed is running and the text reaches Messages on this Mac)
or the user has to enter it. Either way, when the user tells you the code, run `icloud_login_code(code: str)`, e.g.
`icloud_login_code("123456")` — it submits the code and answers with the outcome. The other way in is the console's
iCloud tab (Integrations tab, iCloud sub-tab, Ctrl-E). A code is single-use and lasts a few minutes; the newest text
holds the current one. Never `restart_integration("iCloud")` while a code is pending — a restart opens a new
challenge and kills the code the user is holding.

### When Apple refused a sign-in
After Apple refuses the server's iCloud sign-in (the Apple Account locked for security reasons, new terms to accept,
a rejected password) the server parks every sign-in of its own and says so; Find My and iCloud Photos stay off until
the user has sorted it out on Apple's side (https://iforgot.apple.com for a lock) and says the account works again.
Then lift the park:

`icloud_unlocked()` — lifts the park and signs in at once.

Never run it on your own initiative: a sign-in
against an account that is still locked is what earns the next lock. A rejected password is the one exception: a new
`icloud.apple_id` or `icloud.password` (config_request) lifts the park by itself, so `restart_integration("iCloud")`
after re-entering them signs in again.

### Stop-and-wait
`locate_user`, `icloud_devices`, `icloud_unlocked`, `icloud_login_code` return data: emit them and stop generating; the result arrives in the next round.
