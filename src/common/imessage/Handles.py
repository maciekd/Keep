"""The addresses Messages knows a person or a chat by: phone numbers, emails and group chat guids."""
import re
from typing import Optional


def normalize_phone(phone: str) -> str:
    """Normalize phone number: keep leading + and digits only."""
    if not phone:
        return phone
    return re.sub(r'[^\d+]', '', phone)


def phone_suffix(phone: str) -> Optional[str]:
    """Extract last 9 digits for matching local (0xxx) vs international (+46xxx) formats."""
    if not phone:
        return None
    digits = re.sub(r'\D', '', phone)
    return digits[-9:] if len(digits) >= 7 else None


def prefer_handle(raw_handle: str) -> Optional[str]:
    """Return a normalized handle suitable for AppleScript (E.164 phone or lowercase email)."""
    if not raw_handle:
        return None
    if "@" in raw_handle:
        return raw_handle.strip().lower()
    digits = re.sub(r'[^\d+]', '', raw_handle)
    return digits or None


def looks_like_handle(s: str) -> bool:
    """Heuristic: does this look like a phone number or email rather than a contact name?"""
    if not s:
        return False
    if "@" in s:
        return True
    digits = re.sub(r'\D', '', s)
    return len(digits) >= 7


def is_group_guid(handle: str) -> bool:
    """Whether a resolved handle is a group chat's guid (`any;+;chat…`) rather than a buddy —
    Messages marks a group with `;+;` and a 1:1 chat with `;-;`."""
    return ";+;" in (handle or "")
