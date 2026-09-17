# Keep — AI assistance done right

Keep is a self-hosted AI assistant that runs on a Mac in your home and actually knows your life: your mail, your
photos, your calendar, your house, your car, and everything you have ever told it. 

It is built for people who live in the <b>Apple / iCloud ecosystem</b>, and it runs on <b>Apple Silicon Macs</b> only.

Refined over two years, its robust and actually WORKS. 


<p align="center">
  <img src="docs/images/example_2.png" width="32%" align="top" alt="Keep identifies a chair from a photo and finds the order and price in the mail">
  <img src="docs/images/example_3.png" width="32%" align="top" alt="Keep finds photos of a room in the photo library">
  <img src="docs/images/example_1.png" width="32%" align="top" alt="Keep completes an online order and summarizes the delivery">
</p>
todo: riktiga bilder

## Features

- 🧠 **Keep remembers**
  - Every night Keep goes into REM sleep, replays the day and curates its long-term memory<br>*"Remember that Sam hates cilantro."*
- 🔎 **Can look anything up — in your own life**
  - Every conversation, message, mail, photo and video you ever had is searchable.<br>*"What year did we renovate the kitchen?"*
- 📸 **Knows your photo library**
  - Every photo and video is described and indexed: what is in it, who, where, even what is said.<br>*"Find the video where Lily rides her bike on the beach."*
- ⏰ **Acts on its own**
  - Keep checks in every hour, schedules its own wake-ups and pings you when the moment comes.<br>*"Let me know when my wife is ten minutes out."*
- 🗣️ **Listens**
  - Fast voice-to-voice for the everyday stuff, and it recognizes who is speaking, so "my" means you.<br>*"Where's my phone?"*
- 🏠 **Runs the house**
  - Lights, music, cameras, rain radar and the car, all in one conversation.<br>*"Let me know when there's a package by the door"*
- ✉️ **Handles the boring parts**
  - Email with attachments, calendar, the shared shopping list, iMessages, even web forms and checkouts.<br>*"Reply to the plumber that Thursday works."*
- 🔐 **Never sees your secrets**
  - Passwords and card numbers are typed into a masked prompt and stay encrypted on your Mac; the AI only gets a placeholder.<br>*"Save my new card."*
- 🛠️ **Fixes itself**
  - Keep reads its own logs and source, finds the cause and opens a pull request for you to review.<br>*"Why was that so slow?"*

## Requirements

- An Apple Silicon (M-series) Mac that stays on as a server
- Python 3.12
- An API key for Anthropic

Everything else — iCloud, email, smart home, cameras, car — is optional, and Keep walks you through setting it up.

## Try it out!

Open Terminal and paste:

```bash
curl ..... TBD
```

## Privacy

Let's be honest: an assistant that reads your mail, sees your photos and listens in your kitchen is a bit of a privacy
nightmare. Here is where Keep stands today.

- **Your data lives on your Mac.** The database, photo index, logs and voice profiles never leave it, and
  there is no Keep cloud.
- **The thinking happens in Anthropics servers.** Make sure you have data retention off. 
- **Secrets stay out of it.** The Keep Vault keeps passwords, card numbers and the like encrypted on your Mac, and the AI
  only ever sees placeholders. It is a system for hiding secrets from the cloud, not for anonymizing everything you say.
- **The long game is 100% local.** Keep already runs on local models and once the SOTA local models are good enough we will move to local only.
