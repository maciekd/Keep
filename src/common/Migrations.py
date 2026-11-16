"""One-off moves of what an install keeps on disk, run by the launcher before anything reads
it. Each entry can go once every install has started on the release that added it."""
from common import ConfigStore
from common.Paths import AISERVER_DIR
from common.logging import info, trace, error

# 2026-09-17: "Telegram Monitor" became "Telegram Feed" and the bot's section `telegram` became
# `telegram_bot`, so the two Telegram connections can no longer be taken for each other.
CONFIG_KEY_RENAMES = {
    "telegram": "telegram_bot",
    "telegram_monitor": "telegram_feed",
    "feeds.telegram_monitor": "feeds.telegram_feed",
}
# 2026-11-10: "iMessage" became "iMessage Feed" and "WhatsApp" "WhatsApp Feed" (and the bot,
# which is in no list, "Telegram Chat"), so the names read the same everywhere: a Chat is the
# assistant's account, a Feed the owner's.
# 2026-11-13: "Sonos & Spotify" became "Sonos": the Spotify account is core's and every speaker
# system's (HEOS too), so the integration is the speakers alone. Its section was `sonos` already.
INTEGRATION_RENAMES = {"Telegram Monitor": "Telegram Feed", "iMessage": "iMessage Feed", "WhatsApp": "WhatsApp Feed",
                       "Sonos & Spotify": "Sonos"}
# The Telethon session: without the move the feed would ask for a new login (code + 2FA).
FILE_RENAMES = {
    "telegram_monitor.session": "telegram_feed.session",
    "telegram_monitor.session-journal": "telegram_feed.session-journal",
}


def run():
    try:
        for change in ConfigStore.rename_keys(CONFIG_KEY_RENAMES, INTEGRATION_RENAMES):
            info(f"Migration: config.yaml {change}")
    except Exception as e:
        error(f"Migration: config.yaml keys not renamed ({type(e).__name__}: {e}) — rename them by hand: {CONFIG_KEY_RENAMES}")
    for old_name, new_name in FILE_RENAMES.items():
        old, new = AISERVER_DIR / old_name, AISERVER_DIR / new_name
        if not old.exists():
            trace(f"Migration: no {old_name} to move")
        elif new.exists():
            info(f"Migration: both {old_name} and {new_name} exist — {old_name} left in place")
        else:
            old.rename(new)
            info(f"Migration: {old_name} moved to {new_name}")
