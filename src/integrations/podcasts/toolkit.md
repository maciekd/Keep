---
name: podcasts
summary: "the podcast episode reports (Moonshots): list them, read an explainer, LinkedIn ideas or TL;DR, search them, draft a LinkedIn post, regenerate one"
commands: [podcast_episodes, load_podcast_artifact, podcast_search, podcast_draft, podcast_regenerate]
---
### Podcasts
Every morning new episodes of the user's podcasts (Moonshots) are fetched, transcribed and turned into three
artifacts, saved in the show's folder, which syncs to the user's Obsidian vault: the `explainer` (the concepts and
references of the episode, chapter by chapter — not a recap), `linkedin_ideas` (three to five numbered content ideas,
each with a hook, an insight, a counter-take and its source timestamp) and the `tldr` (five bullets). A history row
"<show> #<number>: <title> — report ready" marks each finished report, and the user gets a message when a run starts
and ends. An episode is named by its slug, such as `moonshots-306`; the older episodes have the explainer alone.

`podcast_episodes(limit: int = 5)` — the newest episodes, one line each: slug, number and title, the date of the
report, the artifacts it has, and the run in progress when there is one.

`load_podcast_artifact(slug: str, kind: str)` — one artifact whole; `kind` is `explainer`, `linkedin_ideas` or `tldr`.

`podcast_search(query: str)` — the lines of every report and every kept transcript that contain all the words of
`query`, each with the episode, the artifact or transcript chapter it is in and a line of context.

`podcast_draft(slug: str, idea_no: int)` — expands LinkedIn idea number `idea_no` of the episode into a full post,
with the transcript around its source, and saves it beside the ideas. Show the user the post.

`podcast_regenerate(slug: str, kind: str)` — writes one artifact of an episode again from its transcript, on its own;
the user gets a message when it is done (an explainer takes several minutes). Episodes from before this pipeline
have no transcript here and cannot be regenerated.

### Stop-and-wait
`podcast_episodes`, `load_podcast_artifact`, `podcast_search` and `podcast_draft` return data: emit the command and
stop generating; the result arrives in the next round.
