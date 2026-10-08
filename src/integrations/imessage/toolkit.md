---
name: imessage
summary: "send an iMessage or SMS to a contact"
commands: [send_imessage]
---
You can send a message via iMessage (with automatic SMS fallback when iMessage isn't reachable) using:

`send_imessage(recipient: str, message: str)`

The recipient can be a contact's first name, full name as `First Last` (use this when the first name alone
is ambiguous — e.g., two contacts named Alex), or a raw handle (phone number in any format, or email).
Wrap each argument in quotes so commas inside the quoted message body are preserved as part of the body
(e.g. `send_imessage("Alex", "Running 10 min late, see you soon")`).

The message ends with a "[Sent with the help of AI 🤖]" footer unless the setting `ai_message_footer` is false;
`config_set("ai_message_footer", "false")` (setup toolkit) turns it off for every channel, read at each send.
