from aqar_intel.arabic import detect_language, normalize_for_match, to_western_digits, tokenize


def test_digit_conversion():
    assert to_western_digits("١,١٥١,٠٠٠") == "1,151,000"
    assert to_western_digits("۲۰۲۶") == "2026"


def test_tokenize_strips_article_and_normalises():
    assert tokenize("ما هي رسوم الحجز؟") == ["رسوم", "حجز"]
    assert tokenize("أسعار الفلل") == ["اسعار", "فلل"]


def test_language_detection():
    assert detect_language("كم عدد الفلل المتاحة؟") == "ar"
    assert detect_language("How many villas are available?") == "en"


def test_normalize_for_match_is_tolerant():
    assert normalize_for_match("أبراج الشاطئ") == normalize_for_match("ابراج الشاطي")
    assert normalize_for_match("AUF-2026-0143") == normalize_for_match("AUF 2026 0143")
