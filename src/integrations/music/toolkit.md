---
name: music
summary: "play, pause, skip, volume, search Spotify, start a playlist (Sonos or the user's own Spotify library) or favorite on a Sonos zone"
commands: [play, pause, next, volume_up, volume_down, search_music, play_music]
---
You can control the Sonos sound system in the house.

Here are the commands you can use (use this syntax exactly):
`play(zone_name: str)`
`pause(zone_name: str)`
`next(zone_name: str)`
`volume_up(zone_name: str)`
`volume_down(zone_name: str)`

You can search for music on Spotify using (use this syntax exactly):
`search_music(search_query: str)`

This command searches songs, artists and public playlists simultaneously and returns the best results of each type.

Then you can start playing one of the search hits using (use this syntax exactly):
`play_music(zone_name: str, id: str)`

The `id` parameter format:
- For Spotify search results: Use the short ID from the search results (e.g., "AA", "AB", "AC")
- For Sonos playlists: Use "P-0", "P-1", etc. (see SONOS PLAYLISTS AND ZONES in your prompt)
- For Sonos favorites: Use "F-0", "F-1", etc. (see SONOS PLAYLISTS AND ZONES in your prompt)
- For the user's own Spotify playlists (owned and followed): Use "S-0", "S-1", etc. (see SONOS PLAYLISTS AND ZONES
  in your prompt). Sonos resolves the tracks itself with the household's Spotify account, so these play like any
  Spotify playlist.

The playlists, favorites and zones are listed in the SONOS PLAYLISTS AND ZONES section of your prompt. The "S-" list
holds the newest 20 playlists of the user's Spotify library; the section says when there are more, and search_music
finds a public one by name. Spotify-made playlists (Discover Weekly, Release Radar, Daily Mix) are not in the "S-" list
— Spotify hides them from apps like this one — so look for them among the Sonos favorites ("F-") and, when they are
not there, tell the user to add the playlist as a Sonos favorite in the Sonos app. When the user asks for one of their
own Spotify playlists by name and no "S-" ids are listed, the Spotify account is not signed in yet: offer to sign it in
with `spotify_login()` from the `setup` toolkit (a link to open, the landing address pasted back) — the playlists
appear in the section within a minute of the sign-in.

### Stop-and-wait
`search_music()` returns data: emit it and stop generating; the result arrives in the next round.
