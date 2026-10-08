---
name: whatsapp
summary: "send a WhatsApp message to a contact or a group; link the WhatsApp account"
commands: [send_whatsapp, whatsapp_login]
---
You can send a WhatsApp message from the user's own account:

`send_whatsapp(recipient: str, message: str)`

The recipient is a contact's name as the phone's address book has it (first name, or `First Last` when the first
name alone is ambiguous — the error lists the full names), a group's name, or a phone number with its country code
(`+46701234567`). Wrap each argument in quotes so commas in the message stay part of it, e.g.
`send_whatsapp("Alex", "Running 10 min late, see you soon")`. The approval card shows the number it resolved to and
says when the user has never written with that number; WhatsApp restricts accounts that message strangers first, so
only send to an unknown number when the user clearly asked for it.

The message ends with a "[Sent with the help of AI 🤖]" footer unless the setting `ai_message_footer` is false;
`config_set("ai_message_footer", "false")` (setup toolkit) turns it off for every channel, read at each send.

`whatsapp_login(phone: str)` — links the server to the user's WhatsApp account as a companion device, e.g.
`whatsapp_login("+46701234567")`, from any chat. Run it when the integration is enabled and reports "needs 2FA" (or
WhatsApp unlinked the server — the phone was off WhatsApp for 14 days, or the user removed the device). It comes
back at once with an 8-character code: tell the user to open WhatsApp on the phone, Linked devices > Link a device >
"Link with phone number instead", and type the code there within a few minutes. A system note says when the link
landed and the feed started; the phone then uploads the chat history, which takes minutes. A code that was not
entered in time comes back as a note too — run it again for a new one.
