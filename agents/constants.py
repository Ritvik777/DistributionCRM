"""Shared constants for agents and UI."""

import re

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

SEND_CONFIRM_PHRASES = frozenset(
    phrase.lower()
    for phrase in (
        "send it",
        "send now",
        "go ahead and send",
        "yes send",
        "please send",
    )
)

MAX_HISTORY_MESSAGES = 10
