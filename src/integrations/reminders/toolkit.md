---
name: shopping_list
summary: "add to the family's shared shopping list, tick a reminder off"
commands: [add_shopping_list_item, complete_reminder]
---
### Shopping list
The family has a shared shopping list (a shared iOS Reminders list). You can add items to it:

`add_shopping_list_item(item: str)`

If you want to add several things call the command several times in the same response.

### Completing a reminder
Any open reminder, in the shopping list or another Reminders list, is marked as completed by its title:

`complete_reminder(text: str)`

`text` is the reminder's title as the user says it; the match is case-insensitive against the open reminders. Exactly
one match is completed. Several matches (the same title in more than one list) are refused and named back to you with
their lists — nothing is completed, tell the user. No match is reported as such.

Reading the open reminders of every list takes a while: emit the command and stop generating; the result arrives in
the next round.
