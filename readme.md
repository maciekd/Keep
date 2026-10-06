<p align="center"><img src="docs/images/banner.svg" width="100%" alt="Keep — personal AI done right"></p>
<br>
<p align="center">
  A self-hosted personal AI assistant that <b>just works</b>.
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
Built for users in the <b>Apple ecosystem</b>, runs on an <b>Apple Silicon Mac</b> and is powered by <b>Fable from Anthropic</b>.

Refined over two years of daily use in a real home: it's robust and <b><i>just works</i></b>.

### Requirements
Apple Silicon Mac connected to your personal iCloud.

Everything else — email, smart home, cameras, car — is optional, and Keep walks you through setting it up.

<img src="docs/images/banner.svg" align="left" width="2000" height="0" alt="">
</td>
</tr>
</table>

<br>

## Try it out!

```bash
gh auth login && gh api -H "Accept: application/vnd.github.raw" "repos/maciekd/Keep/contents/install.sh?ref=release" | bash
```

> [!NOTE]
> Keep is in closed beta: the install only works for GitHub accounts invited as collaborators. `brew install gh` if you do not have `gh`. The installer is self-contained, it won't touch anything else on your Mac.

<br>

## Features

I asked Keep to pick her own highlights from the past year. **All events are real.**

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

### 🇩🇪 The Autobahn breakdown
I decoded a dashboard full of warnings from one photo.

<details>
<summary>The full story</summary>
<br>

The car lit up like a Christmas tree in northern Germany; he sent a photo of the alert screen. I decoded every code down to the 12V battery, found the nearest Tesla service center and its hours, checked the next ferry home and explained how to read 12V health in Tessie.

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

## Interfaces

<table>
<tr><td width="190">⌨️&nbsp;<b>Terminal&nbsp;UI</b></td><td><img src="docs/images/banner.svg" align="left" width="2000" height="0" alt="">Text chat in a Claude Code-style terminal UI.</td></tr>
<tr><td><img src="docs/images/telegram.svg" width="18" height="18" alt="" valign="middle">&nbsp;<b>Telegram</b></td><td>The same chat on your phone, with images and voice messages.</td></tr>
<tr><td>🎙️&nbsp;<b>Voice&#8209;to&#8209;voice</b></td><td>A highly optimized interface for talking to your home.</td></tr>
</table>

<br>

## Integrations

Every integration is off until you turn it on, and Keep walks you through the setup of each one.

<table>
<tr><td width="190">🍎&nbsp;<b>Apple</b></td><td><img src="docs/images/banner.svg" align="left" width="2000" height="0" alt="">iCloud (Find My for the family's phones), Photos library, iMessage, Reminders, iCloud Mail and CalDAV calendars</td></tr>
<tr><td>✉️&nbsp;<b>Email</b></td><td>Gmail and iCloud Mail; Keep also uses your inbox to handle 2FA codes on its own</td></tr>
<tr><td>🏠&nbsp;<b>Home</b></td><td>Home Assistant and Indigo for lights, switches, locks, sensors and scenes</td></tr>
<tr><td>📹&nbsp;<b>Cameras</b></td><td>Arlo and Ring</td></tr>
<tr><td>🚗&nbsp;<b>Car</b></td><td>Tesla, through tessie.com</td></tr>
<tr><td>🎵&nbsp;<b>Music</b></td><td>Sonos and Spotify</td></tr>
<tr><td>🌦️&nbsp;<b>Weather</b></td><td>YR forecasts and rain radar</td></tr>
<tr><td>🌐&nbsp;<b>Web</b></td><td>Search, page reading, a real headless browser for sites that need clicking, and X</td></tr>
<tr><td>💬&nbsp;<b>Chat</b></td><td>Telegram, with a feed of the family's group chats</td></tr>
<tr><td>🎙️&nbsp;<b>Voice</b></td><td>Fusion of local Whisper transcription and ElevenLabs, for <i>low latency</i> combined with <i>high accuracy</i></td></tr>
<tr><td>🛠️&nbsp;<b>Self&nbsp;Improvement</b></td><td>Keep reads its own logs and code through Claude Code, finds bugs and sends you the pull request</td></tr>
</table>

> [!TIP]
> Missing something? Ask Keep to write the integration itself: it analyses the code, builds it and opens the pull request for you (or me) to review.

<br>

## Architecture

<p align="center"><a href="https://maciekd.github.io/Keep/atlas/"><img src="docs/images/atlas.png" width="70%" alt="The Keep Atlas"></a></p>

Curious how it fits together? The **[Keep Atlas](https://maciekd.github.io/Keep/atlas/)** is a zoomable map of the whole server: every package, module and class with a one-line description, and a walk through one spoken request from the microphone to the speaker.

<br>

## Privacy

Let's be honest: an assistant that reads your mail, sees your photos and messages is a privacy minefield. Here is where Keep stands today.

- 💾 **Your data lives on your Mac.** The database, photo index, logs and voice profiles never leave it, and there is no Keep cloud.
- ☁️ **The thinking happens on Anthropic's servers.** Anthropic does not train on API data by default; but you need to be ok with this.
- 🔐 **Secrets stay out of it.** The Keep Vault keeps passwords, card numbers and the like encrypted on your Mac, and the AI only ever sees placeholders. It is a system for hiding secrets from the cloud, not for anonymizing everything you say.

The end game is 100% local. Keep already runs on local models and once the local SOTA models are good enough, we'll move to <i>local only</i>. Right now Fable from Anthropic is needed for a **<i>no-compromise solution</i>**.

> [!IMPORTANT]
> **Beta telemetry.** During the beta Keep sends anonymous usage events: a random install id, the Keep version, which
> interface and integrations are in use, which you switch on or off, which TUI tabs you open, and the names of the
> commands a turn ran — never their arguments. Until you enable a sensitive integration (mail, messages, calendars,
> photos, cameras, Find My) the events also carry your queries and Keep's replies; the moment you do, that is switched
> off for good. Vault secrets are placeholders in the transcript too. `telemetry.enabled: false` in `config.yaml` turns
> it all off.
