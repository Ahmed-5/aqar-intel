"""Fictional developer, projects and vocab shared by every generator.

Everything here is invented for demo purposes. Names, prices and policies do
not describe any real company or project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEVELOPER_AR = "شركة الأفق للتطوير العقاري"
DEVELOPER_EN = "Al-Ufuq Real Estate Development Co."
DEVELOPER_CR = "1010987654"  # commercial registration (fictional)
DEVELOPER_PHONE = "920012345"
DEVELOPER_EMAIL = "sales@alufuq-demo.example"


@dataclass
class Project:
    id: str
    name_ar: str
    name_en: str
    city_ar: str
    city_en: str
    district_ar: str
    district_en: str
    unit_types: list[str]  # subset of UNIT_TYPES keys
    base_price_sqm: int  # SAR per sqm, baseline
    launch: str  # YYYY-MM
    delivery: str  # YYYY-MM
    n_units: int
    amenities_ar: list[str] = field(default_factory=list)
    amenities_en: list[str] = field(default_factory=list)
    highlights_ar: str = ""
    highlights_en: str = ""


UNIT_TYPES = {
    "apartment": {"ar": "شقة", "en": "Apartment", "area": (95, 210), "beds": (2, 4), "mult": 1.00},
    "townhouse": {"ar": "تاون هاوس", "en": "Townhouse", "area": (220, 320), "beds": (3, 5), "mult": 1.08},
    "duplex": {"ar": "دوبلكس", "en": "Duplex", "area": (240, 360), "beds": (4, 5), "mult": 1.12},
    "villa": {"ar": "فيلا", "en": "Villa", "area": (320, 520), "beds": (4, 6), "mult": 1.20},
}

VIEWS = {
    "garden": {"ar": "حديقة", "en": "Garden", "mult": 1.04},
    "street": {"ar": "شارع", "en": "Street", "mult": 1.00},
    "city": {"ar": "المدينة", "en": "City", "mult": 1.03},
    "sea": {"ar": "البحر", "en": "Sea", "mult": 1.15},
    "park": {"ar": "المنتزه", "en": "Park", "mult": 1.06},
}

STATUSES = {
    "available": {"ar": "متاحة", "en": "Available"},
    "reserved": {"ar": "محجوزة", "en": "Reserved"},
    "sold": {"ar": "مباعة", "en": "Sold"},
}

PROJECTS: list[Project] = [
    Project(
        id="NRJ", name_ar="واحة النرجس", name_en="Narjis Oasis",
        city_ar="الرياض", city_en="Riyadh", district_ar="حي النرجس", district_en="Al Narjis",
        unit_types=["villa", "townhouse"], base_price_sqm=6200,
        launch="2024-02", delivery="2026-12", n_units=340,
        amenities_ar=["مسجد", "نادي رياضي", "مسارات مشي", "حدائق مركزية", "مدرسة"],
        amenities_en=["Mosque", "Fitness club", "Walking trails", "Central gardens", "School"],
        highlights_ar="مجتمع سكني متكامل شمال الرياض بالقرب من مطار الملك خالد وطريق الملك سلمان.",
        highlights_en="An integrated community in north Riyadh near King Khalid Airport and King Salman Road.",
    ),
    Project(
        id="CST", name_ar="أبراج الشاطئ", name_en="Coast Towers",
        city_ar="جدة", city_en="Jeddah", district_ar="حي الشاطئ", district_en="Ash Shati",
        unit_types=["apartment", "duplex"], base_price_sqm=9800,
        launch="2024-05", delivery="2027-06", n_units=420,
        amenities_ar=["مسبح لا متناهي", "صالة رياضية", "أمن على مدار الساعة", "مواقف ذكية", "تراس بحري"],
        amenities_en=["Infinity pool", "Gym", "24/7 security", "Smart parking", "Sea terrace"],
        highlights_ar="برجان سكنيان على كورنيش جدة بإطلالات بحرية مباشرة.",
        highlights_en="Twin residential towers on the Jeddah corniche with direct sea views.",
    ),
    Project(
        id="RWB", name_ar="حي الروابي", name_en="Al Rawabi District",
        city_ar="الدمام", city_en="Dammam", district_ar="حي الروابي", district_en="Al Rawabi",
        unit_types=["villa", "duplex"], base_price_sqm=4600,
        launch="2023-09", delivery="2026-06", n_units=260,
        amenities_ar=["مسجد", "ملاعب أطفال", "سوق مركزي", "مسارات دراجات"],
        amenities_en=["Mosque", "Playgrounds", "Central market", "Cycling lanes"],
        highlights_ar="فلل عائلية بمساحات واسعة على بعد دقائق من طريق الملك فهد.",
        highlights_en="Spacious family villas minutes from King Fahd Road.",
    ),
    Project(
        id="YSM", name_ar="ضاحية الياسمين", name_en="Yasmin Suburb",
        city_ar="الرياض", city_en="Riyadh", district_ar="حي الياسمين", district_en="Al Yasmin",
        unit_types=["apartment", "townhouse"], base_price_sqm=5800,
        launch="2024-09", delivery="2027-03", n_units=520,
        amenities_ar=["مركز تجاري", "مدرسة دولية", "حدائق", "نادي نسائي", "مسجد جامع"],
        amenities_en=["Retail centre", "International school", "Gardens", "Ladies' club", "Grand mosque"],
        highlights_ar="ضاحية متكاملة تستهدف الأسر الشابة ضمن برامج الدعم السكني.",
        highlights_en="An integrated suburb targeting young families under housing-support programmes.",
    ),
    Project(
        id="KHM", name_ar="مرسى الخبر", name_en="Khobar Marina",
        city_ar="الخبر", city_en="Al Khobar", district_ar="الكورنيش", district_en="Corniche",
        unit_types=["apartment", "duplex"], base_price_sqm=7900,
        launch="2025-01", delivery="2027-09", n_units=310,
        amenities_ar=["مرسى يخوت", "مطاعم", "نادي صحي", "مواقف تحت الأرض"],
        amenities_en=["Yacht marina", "Restaurants", "Health club", "Underground parking"],
        highlights_ar="شقق فاخرة على واجهة الخبر البحرية مع مرسى خاص.",
        highlights_en="Luxury waterfront apartments in Al Khobar with a private marina.",
    ),
    Project(
        id="AQQ", name_ar="وادي العقيق", name_en="Wadi Al Aqiq",
        city_ar="المدينة المنورة", city_en="Madinah", district_ar="العقيق", district_en="Al Aqiq",
        unit_types=["villa", "townhouse"], base_price_sqm=4200,
        launch="2025-04", delivery="2028-01", n_units=280,
        amenities_ar=["مسجد", "حدائق", "مركز خدمات", "مسارات مشي"],
        amenities_en=["Mosque", "Gardens", "Service centre", "Walking trails"],
        highlights_ar="مجتمع هادئ في وادي العقيق على بعد 15 دقيقة من المسجد النبوي.",
        highlights_en="A quiet community in Wadi Al Aqiq, 15 minutes from the Prophet's Mosque.",
    ),
]

PROJECT_BY_ID = {p.id: p for p in PROJECTS}

# Fictional buyer names for contracts
FIRST_NAMES_AR = ["محمد", "أحمد", "عبدالله", "خالد", "سعود", "فهد", "نورة", "سارة", "ريم", "هند", "عبدالعزيز", "تركي", "لمى", "منيرة", "بدر"]
FAMILY_NAMES_AR = ["العتيبي", "القحطاني", "الشمري", "الدوسري", "الحربي", "المطيري", "الزهراني", "الغامدي", "السبيعي", "العنزي", "الشهري", "البقمي"]
FIRST_NAMES_EN = ["Mohammed", "Ahmed", "Abdullah", "Khalid", "Saud", "Fahad", "Noura", "Sarah", "Reem", "Hind", "Abdulaziz", "Turki", "Lama", "Munira", "Badr"]
FAMILY_NAMES_EN = ["Al-Otaibi", "Al-Qahtani", "Al-Shammari", "Al-Dossari", "Al-Harbi", "Al-Mutairi", "Al-Zahrani", "Al-Ghamdi", "Al-Subaie", "Al-Anazi", "Al-Shehri", "Al-Buqami"]

BANKS_AR = ["مصرف الراجحي", "البنك الأهلي السعودي", "بنك الرياض", "البنك السعودي الفرنسي", "بنك البلاد"]
BANKS_EN = ["Al Rajhi Bank", "Saudi National Bank", "Riyad Bank", "Banque Saudi Fransi", "Bank AlBilad"]
