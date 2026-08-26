"""
Language display names.

The picker shows each language in its own script, because the person choosing
reads that language and not necessarily English. A listener who wants Arabic
should be looking for العربية, not for the word "Arabic".

`rtl` drives text direction on the button, so Arabic and Hebrew labels do not
render backwards next to Latin ones.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LanguageInfo:
    code: str
    native: str
    english: str
    rtl: bool = False


_LANGUAGES: dict[str, LanguageInfo] = {
    "ar": LanguageInfo("ar", "العربية", "Arabic", rtl=True),
    "de": LanguageInfo("de", "Deutsch", "German"),
    "en": LanguageInfo("en", "English", "English"),
    "es": LanguageInfo("es", "Español", "Spanish"),
    "fa": LanguageInfo("fa", "فارسی", "Persian", rtl=True),
    "fr": LanguageInfo("fr", "Français", "French"),
    "he": LanguageInfo("he", "עברית", "Hebrew", rtl=True),
    "hi": LanguageInfo("hi", "हिन्दी", "Hindi"),
    "it": LanguageInfo("it", "Italiano", "Italian"),
    "ja": LanguageInfo("ja", "日本語", "Japanese"),
    "ko": LanguageInfo("ko", "한국어", "Korean"),
    "ml": LanguageInfo("ml", "മലയാളം", "Malayalam"),
    "nl": LanguageInfo("nl", "Nederlands", "Dutch"),
    "pt": LanguageInfo("pt", "Português", "Portuguese"),
    "ru": LanguageInfo("ru", "Русский", "Russian"),
    "ta": LanguageInfo("ta", "தமிழ்", "Tamil"),
    "te": LanguageInfo("te", "తెలుగు", "Telugu"),
    "tr": LanguageInfo("tr", "Türkçe", "Turkish"),
    "ur": LanguageInfo("ur", "اردو", "Urdu", rtl=True),
    "zh": LanguageInfo("zh", "中文", "Chinese"),
}


def describe(code: str) -> LanguageInfo:
    """
    Look up a language, falling back to the raw code.

    An unknown code shows as itself rather than raising. A missing display name
    is a cosmetic problem; a join page that will not render is a room full of
    people hearing nothing.
    """
    return _LANGUAGES.get(code.lower()) or LanguageInfo(code, code.upper(), code.upper())


def known() -> list[str]:
    return sorted(_LANGUAGES)
