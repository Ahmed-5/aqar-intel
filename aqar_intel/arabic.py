"""Arabic/English text utilities.

Lexical retrieval (BM25) and field-level evaluation both need consistent
normalisation. Arabic in real documents varies in diacritics, alef forms,
taa-marbuta vs haa, and Arabic-Indic vs Western digits; without normalising
these, exact-match evaluation and keyword search silently underperform.
"""

from __future__ import annotations

import re
import unicodedata

_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")  # آ أ إ ٱ -> ا
_ARABIC_LETTER = re.compile(r"[\u0621-\u064A\u066E-\u06D3\u0671]")  # letters only, not punctuation (؟ ، ؛)
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[\u0621-\u064A\u066E-\u06D3\u0671]+|[A-Za-z]+|\d+(?:[.,]\d+)?")

_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
_EASTERN_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(_ARABIC_INDIC)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(_EASTERN_ARABIC_INDIC)})
_ARABIC_DECIMAL_SEP = "\u066B"  # ٫
_ARABIC_THOUSANDS_SEP = "\u066C"  # ٬

ARABIC_STOPWORDS = {
    "في", "من", "على", "إلى", "الى", "عن", "أن", "ان", "إن", "ما", "هل", "هو", "هي",
    "مع", "كم", "ماذا", "متى", "أين", "اين", "كيف", "هذا", "هذه", "ذلك", "التي",
    "الذي", "و", "أو", "او", "ثم", "لا", "نعم", "قد", "كان", "يكون", "لدى", "لدي",
}
ENGLISH_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "is", "are", "and", "or",
    "what", "how", "many", "much", "which", "do", "does", "i", "me", "my", "we", "with",
    "can", "there", "any", "be", "it", "this", "that",
}


def to_western_digits(text: str) -> str:
    """Convert Arabic-Indic (٠١٢) and Eastern Arabic-Indic (۰۱۲) digits to 0-9."""
    return (
        text.translate(_DIGIT_MAP)
        .replace(_ARABIC_DECIMAL_SEP, ".")
        .replace(_ARABIC_THOUSANDS_SEP, ",")
    )


def normalize_arabic(text: str) -> str:
    """Light, retrieval-friendly normalisation (does not stem)."""
    text = unicodedata.normalize("NFKC", text)
    text = to_western_digits(text)
    text = _TASHKEEL.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = _ALEF_VARIANTS.sub("\u0627", text)  # ا
    text = text.replace("\u0629", "\u0647")  # ة -> ه
    text = text.replace("\u0649", "\u064A")  # ى -> ي
    text = text.replace("\u0624", "\u0648")  # ؤ -> و
    text = text.replace("\u0626", "\u064A")  # ئ -> ي
    return text.lower()


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Tokenise mixed Arabic/English/numeric text after normalisation.

    Also strips the Arabic definite article ``ال`` from tokens so that
    "السعر" and "سعر" match. This is deliberately conservative: a full stemmer
    would raise recall but hurts precision on short domain terms.
    """
    norm = normalize_arabic(text)
    toks = []
    for t in _TOKEN.findall(norm):
        if _ARABIC_LETTER.search(t):
            if t.startswith("\u0627\u0644") and len(t) > 3:  # ال
                t = t[2:]
            if t.startswith("\u0648\u0627\u0644") and len(t) > 4:  # وال
                t = t[3:]
            if t.startswith("\u0628\u0627\u0644") and len(t) > 4:  # بال
                t = t[3:]
        if drop_stopwords and (t in ARABIC_STOPWORDS or t in ENGLISH_STOPWORDS):
            continue
        toks.append(t)
    return toks


def detect_language(text: str) -> str:
    """Return ``'ar'`` if Arabic letters dominate, else ``'en'``."""
    ar = len(_ARABIC_LETTER.findall(text))
    en = len(_LATIN_LETTER.findall(text))
    return "ar" if ar >= en and ar > 0 else "en"


def normalize_for_match(value: str | None) -> str:
    """Normalisation used when comparing extracted fields with ground truth."""
    if value is None:
        return ""
    v = normalize_arabic(str(value))
    v = re.sub(r"[\s\-_/,،:;]+", " ", v).strip()
    return v
