---
name: shopping_list
summary: "add to the family's shared shopping list; create, list, complete and delete reminders with time or location alarms in the user's Reminders app"
commands: [add_shopping_list_item, create_reminder, list_reminders, complete_reminder, delete_reminder]
---
### Shopping list
The family has a shared shopping list (a shared iOS Reminders list). You can add items to it:

`add_shopping_list_item(item: str)`

If you want to add several things call the command several times in the same response.

### Reminders
You have a Reminders list of your own (the `reminders.list_name` config key names it) that syncs to the user's iPhone. A reminder there is
what to use when the user must be reminded at a time or a place even if the server is down: the phone itself rings.
For something *you* must do later, `schedule_wakeup` is still the tool.

`create_reminder(title: str, due: str = "", notes: str = "", place: str = "")`
- `due`: `YYYY-MM-DD HH:MM` makes an alarm on the phone at that time; `YYYY-MM-DD` alone is a day with no alarm.
- `notes`: free text shown under the title.
- `place`: a geofence alarm, `label; latitude; longitude; radius_m; arrive|leave` — for example
  `"ICA Maxi; 57.6834; 11.9798; 200; arrive"` rings when the phone arrives there. Needs coordinates you know;
  pass `""` to skip a middle argument: `create_reminder("Buy flowers", "", "", "Blomsterlandet; 57.70; 11.97; 150; arrive")`.
The reply carries the reminder's id.

`list_reminders()` — the open reminders in the list and those completed in the last 7 days, with ids, due times
and state. A reminder shown as done was ticked off on the phone: that is how you see the user did it.

`complete_reminder(id: str)` — tick it off. `delete_reminder(id: str)` — remove it.

### Stop-and-wait
`create_reminder`, `list_reminders`, `complete_reminder` and `delete_reminder` return data: emit the command and stop
generating; the result arrives in the next round.
