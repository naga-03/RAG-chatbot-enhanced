from typing import Literal
from langdetect import detect, LangDetectException

LanguageCode = str | Literal["unknown"]

def detect_language(query: str) -> LanguageCode:
    """
    Detect the language of the given query.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'ta', 'hi')
        or 'unknown' if detection fails.
    """
    try:
        return detect(query)
    except LangDetectException:
        return "unknown"
