<p align="center"><img src="docs/images/banner.svg" width="100%" alt="Keep — personal AI done right"></p>
<br>
<p align="center">
A self-hosted personal AI assistant for you and your family
</p>
<br>

<table>
<tr>
<td align="center">
<img src="docs/images/video.gif" width="300" alt="A chat with Keep"><br>
</td>
</tr>
<tr>
<td>

### Keep
Keep knows you and your family. It has analyzed your messages and every photo you’ve taken, so it remembers all the names, all the places, all the plans and everything you need to do, without you explaining anything.

Runs on a Mac you own, talks to you by Telegram, iMessage, terminal and voice, and does things — not just chat: shopping, calendars, cameras, the car.

Keep stores everything on your Mac and thinks on Anthropic’s servers. That’s a deliberate trade-off until local models are good enough. Passwords and card numbers go in the Keep Vault, encrypted on your Mac: the model never sees them. Read the privacy section before you install.

Refined over two years of daily use in a real home.

### Requirements
An Apple Silicon Mac, signed in to your personal iCloud, to run the Keep server.

Everything else — email, smart home, cameras, cars — is optional, and Keep walks you through setting it up.

<img src="docs/images/banner.svg" align="left" width="2000" height="0" alt="">
</td>
</tr>
</table>

<br>

## Try it out!
You can have Keep running in minutes:
```bash
curl -fsSL https://raw.githubusercontent.com/maciekd/Keep/release/install.sh | bash
```
The installer is self-contained, it won't touch anything else on your Mac.

<br>

## Interfaces

<table>
<tr><td width="190">⌨️&nbsp;<b>Terminal&nbsp;UI</b></td><td><img src="docs/images/banner.svg" align="left" width="2000" height="0" alt="">Text chat in a Claude Code-style terminal UI.</td></tr>
<tr><td><img src="docs/images/telegram.svg" width="18" height="18" alt="" valign="middle">&nbsp;<b>Telegram</b></td><td>The same chat on your phone, with images and voice messages.</td></tr>
<tr><td>💬&nbsp;<b>iMessage</b></td><td>Keep as a contact of its own: give it an Apple ID and the family texts it like a friend, alone or in a group chat.</td></tr>
<tr><td>🎙️&nbsp;<b>Voice</b></td><td>Keep hears, thinks and answers in its own voice, fast enough to feel like a real conversation.</td></tr>

</table>

<br>

## Features

I asked my Keep to pick its own highlights from the past year. **All events are real.**

<table>
<tr>
<td width="50%" valign="top">

### 🍝 The pasta timing
I timed his pasta to his wife's drive home.

<details>
<summary>The full story</summary>
<br>

He was cooking, his wife driving home from the cinema. I tracked the car across town, said *hold the pasta* when she pulled over to drop off a friend, then counted her down to five minutes out.

</details>

</td>
<td width="50%" valign="top">

### 🛍️ Shopping, end to end
I buy things and book restaurants, checkout included.

<details>
<summary>The full story</summary>
<br>

Card with a 3DS push, **Klarna** and **BankID** all work: when a signature is needed, he gets a tap-to-sign link on Telegram.

- **A swivel armchair**: I saw in his mail that the first one had bounced back uncollected and reordered it. The price had gone up, so I used a 30% code that needed two items, added a wall charger and saved about 2,000 kr.
- **A radon test kit**: from *"how would we do that?"* to paid in two minutes.
- **A pharmacy order**: Klarna-only checkout, signed with BankID, evening courier slot picked.
- **A secret birthday present**: ordered, and kept secret.
- **An extra waste collection**: booked with the municipality in two minutes, plus a reminder to put the bin out.
- **A Stockholm restaurant**: table booked, calendar event added, newsletter box *unticked*.
- **The everyday stuff**: Birkenstocks, a USB-C adapter in minutes after he asked.

And I know when *not* to buy: I spotted a fake web store before paying.

</details>

</td>
</tr>
<tr>
<td valign="top">

### 🌧️ Cushions
I ping the family when rain is coming and the cushions are still out.

<details>
<summary>The full story</summary>
<br>

When the radar says rain within the hour, I check the cameras for the beige cushions on the black terrace furniture and ping the family. *Running quietly since May.*

</details>

</td>
<td valign="top">

### 📸 Needle, haystack
I found his old red BMW among **87,000 photos and videos**.

<details>
<summary>The full story</summary>
<br>

*"Show me a picture of my old BMW I had a long time ago."* Ten seconds later he had the red coupe on a beach, July 2004.

Every photo and video is described once (people, place, any text in the frame), so I can search back to 2003 in plain words:

- **The fridge and oven model numbers**, read off rating plates he photographed weeks earlier.
- **The hall gallery wall** in 2020, plus the 2014 shot of the frames going up with a laser level.
- **"A drone shot in Thailand way back"**: two clips from 2016–17.
- **The mouthwash**: he snapped the bottle, I read the label, and *"buy me three of these"* became a pharmacy order.

Ask by voice and the photo lands on the kitchen TV.

</details>

</td>
</tr>
<tr>
<td valign="top">

### 🐛 Fixing myself
I caught my own bug and shipped the fix as a PR.

<details>
<summary>The full story</summary>
<br>

Apple Store pages went blank: my self-analysis found my own cookie-banner remover deleting the whole page. And when the cushion rule didn't fire, I found I was reading the 3-hour forecast instead of the 15-minute radar and rewrote my own code.

</details>

</td>
<td valign="top">

### 🪪 BankID into the school portal
I got through a BankID login to summarize the school's weekly letter.

<details>
<summary>The full story</summary>
<br>

I drove the browser to the portal's eID page, pulled the autostart token off the auth endpoint and sent him a tappable BankID link on Telegram (*valid for 30 seconds*). Then I read the weekly letter and summarized what mattered to him.

</details>

</td>
</tr>
<tr>
<td valign="top">

### 🇮🇹 Letting the repairman in from Italy
The family was at Lake Como. The fridge technician was in Göteborg.

<details>
<summary>The full story</summary>
<br>

Gaggenau service was booked while the house stood empty. I sorted the two time windows from the confirmation mails, told him the plan the night before — "when the technician calls, ping me" — and unlocked the front door remotely when he did. Then watched the cameras until the van left. Two weeks later, when the tech was due again and the family was at sea without coverage, I noticed no one had come and told him to rebook.

</details>

</td>
<td valign="top">

### 🏊 Pool chemistry by camera
*There is no sensor:* I read the pool controller's display off a security camera.

<details>
<summary>The full story</summary>
<br>

"Tal." means the chlorine tank is empty; I first blamed a sensor fault, he corrected me, and I remembered. I caught the chlorine crash to 0.32 ppm on Jun 21 and to 0.02 a week later, and worked out that a blown-out display means the pump-room lamp is off: switch it on, read, switch it off.

</details>

</td>
</tr>
</table>

<br>

## Integrations

Every integration is off until you turn it on, and Keep walks you through the setup of each one.

<table>
<tr><td width="190">🍎&nbsp;<b>Apple</b></td><td><img src="docs/images/banner.svg" align="left" width="2000" height="0" alt="">iCloud Find My for the family's phones, Photos, iMessage, Reminders, and iCloud mail and calendars</td></tr>
<tr><td>✉️&nbsp;<b>Email</b></td><td>Any IMAP mailbox plus Gmail and Microsoft 365, many accounts side by side; Keep reads 2FA codes itself</td></tr>
<tr><td>📅&nbsp;<b>Calendars</b></td><td>CalDAV (iCloud, Fastmail, Nextcloud), Google and Microsoft 365 in one week view; Keep adds, moves and deletes events</td></tr>
<tr><td>🏠&nbsp;<b>Home</b></td><td>Home Assistant, Indigo, Philips Hue, Plejd and HomeKit (via HomeClaw) for lights, switches, locks, sensors, thermostats, covers and scenes</td></tr>
<tr><td>📹&nbsp;<b>Cameras</b></td><td>Arlo, Ring and UniFi Protect</td></tr>
<tr><td>🚗&nbsp;<b>Cars</b></td><td>Tesla, Volvo, Hyundai, Kia and Genesis; the whole family's cars as one fleet</td></tr>
<tr><td>🏊&nbsp;<b>Pool</b></td><td>Aseko pool units, through aseko.cloud</td></tr>
<tr><td>🎵&nbsp;<b>Music</b></td><td>Sonos and Spotify</td></tr>
<tr><td>🌦️&nbsp;<b>Weather</b></td><td>YR forecasts and rain radar</td></tr>
<tr><td>🌐&nbsp;<b>Web</b></td><td>Search, page reading, a real headless browser for sites that need clicking, and X</td></tr>
<tr><td>💬&nbsp;<b>Chat</b></td><td>Telegram and WhatsApp; Keep follows the family's group chats and can send on your behalf</td></tr>
<tr><td>🎙️&nbsp;<b>Voice</b></td><td>Fusion of local Whisper transcription and ElevenLabs, for <i>low latency</i> combined with <i>high accuracy</i></td></tr>
<tr><td>🛠️&nbsp;<b>Self&nbsp;Improvement</b></td><td>Keep reads its own logs and code through Claude Code, finds bugs and sends you the pull request</td></tr>
</table>

> [!TIP]
> Missing something? Ask Keep to write the integration itself: it analyses the code, builds it and opens the pull request for you (or me) to review.

<br>

## Architecture

<p align="center"><a href="https://keep-ai.dev/atlas/"><img src="docs/images/atlas.png" width="70%" alt="The Keep Atlas"></a></p>

Curious how it fits together? The **[Keep Atlas](https://keep-ai.dev/atlas/)** is a zoomable map of the whole server: every package, module and class with a one-line description, and a walk through one spoken request from the microphone to the speaker.

<br>

## Privacy

Let's be honest: an assistant that reads your mail, sees your photos and messages is a privacy minefield. Here is where Keep stands today.

- 💾 **Your data lives on your Mac.** The database, photo index, logs and voice profiles never leave it, and there is no Keep cloud.
- ☁️ **The thinking happens on Anthropic's servers.** Anthropic does not train on API data by default; but you need to be ok with this.
- 🔐 **Secrets stay out of it.** The Keep Vault keeps passwords, card numbers and the like encrypted on your Mac, and the AI only ever sees placeholders. It is a system for hiding secrets from the cloud, not for anonymizing everything you say.

The end game is 100% local. Keep already runs on local models and once the local SOTA models are good enough, we'll move to <i>local only</i>. Right now Fable from Anthropic is needed for a **<i>no-compromise solution</i>**.

> [!IMPORTANT]
> **Telemetry.** Keep sends usage events to the cloud: a random install id, whether the install succeeded (or the
> step it failed in), the Keep version, which interface and integrations are in use, the names of the commands a turn ran — never their arguments —, prompt size, etc. For the
> first 7 days after install the events also carry your queries and Keep's replies, so we can see where the first
> steps trip people up; after that they never do. Vault secrets are placeholders in the transcript. Turn it all off
> with `telemetry.enabled: false` in `config.yaml`. To have your data deleted, send us the
> `telemetry.id` from your `config.yaml`.
<br>

## License

Keep is free software under the [GNU Affero General Public License v3.0](https://github.com/maciekd/Keep/blob/master/LICENSE). You may run, study, change and
share it; if you distribute a modified Keep, or let others use one over a network, you must share your changes
under the same license. Copyright © 2024–2026 Maciek Drejak.
