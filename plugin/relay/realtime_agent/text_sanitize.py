"""Markdown / formatting sanitizer for assistant text before speech output.

Speech synthesizers (Kokoro, etc.) tend to spell out Markdown markup
character-by-character — a literal ``*word*`` arrives at the user as the
spoken phrase "asterisk word asterisk". This module strips those characters
so the spoken result keeps the words but loses the markup artifacts.

Lives next to the relay realtime-agent providers so any provider can reuse it.
The matching prompt-side guard (``_native_instructions`` in ``broker.py``) is
the first line of defense; this is the safety net.
"""

from __future__ import annotations

import re

# Characters that voice synthesizers tend to spell out literally:
#   `  backtick          -> "backtick"
#   *  asterisk          -> "asterisk"
#   _  underscore        -> "underscore"
#   ~  tilde             -> "tilde"
#   #  hash              -> "hash" / "number"
#   >  greater-than      -> "greater than"
#   [  ]  brackets       -> "open bracket" / "close bracket"
#   !  bang              -> "exclamation mark" (e.g. image alt: ![alt](url))
_TTS_STRIP_RE = re.compile(r"[`*_~#>\[\]|!]")


def sanitize_for_tts(text: str) -> str:
    """Strip Markdown markup that would otherwise be spoken verbatim.

    Replaces ``*word*``, ``**bold**``, ``# heading``, ``` `code` ```,
    ``> quote``, ``[link]`` and the leading ``!`` of an image alt with a single
    space, then collapses runs of whitespace. Empty input returns ``""``.
    """
    if not text:
        return ""
    cleaned = _TTS_STRIP_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned