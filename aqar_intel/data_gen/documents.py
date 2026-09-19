"""Generate the bilingual document corpus used by the RAG assistant.

Documents are Markdown with a small YAML-ish front matter so the indexer can
attach metadata (project, language, doc type) to every chunk. Numbers are
pulled from the generated inventory database so documents and the SQL
inventory agree with each other.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import DB_PATH, DOCS_DIR
from .world import (
    DEVELOPER_AR, DEVELOPER_CR, DEVELOPER_EMAIL, DEVELOPER_EN, DEVELOPER_PHONE, PROJECTS, UNIT_TYPES, BANKS_AR, BANKS_EN,
)


def _fmt_sar(x: float) -> str:
    return f"{int(x):,}"


def _month_ar(ym: str) -> str:
    months = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
    y, m = ym.split("-")
    return f"{months[int(m) - 1]} {y}"


def _month_en(ym: str) -> str:
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    y, m = ym.split("-")
    return f"{months[int(m) - 1]} {y}"


def _front_matter(**kw) -> str:
    lines = ["---"] + [f"{k}: {v}" for k, v in kw.items()] + ["---", ""]
    return "\n".join(lines)


def _project_stats(con: sqlite3.Connection, pid: str) -> dict:
    rows = con.execute(
        """SELECT unit_type, COUNT(*), SUM(status='available'), MIN(area_sqm), MAX(area_sqm),
                  MIN(list_price_sar), MAX(list_price_sar)
           FROM units WHERE project_id=? GROUP BY unit_type""",
        (pid,),
    ).fetchall()
    total, avail = con.execute(
        "SELECT COUNT(*), SUM(status='available') FROM units WHERE project_id=?", (pid,)
    ).fetchone()
    return {"by_type": rows, "total": total, "available": avail}


# --------------------------------------------------------------------------- #
# Brochures
# --------------------------------------------------------------------------- #
def brochure_ar(p, stats) -> str:
    lines = [
        _front_matter(doc_type="brochure", project_id=p.id, project=p.name_ar, lang="ar"),
        f"# مشروع {p.name_ar} – {p.city_ar}",
        "",
        f"**المطور:** {DEVELOPER_AR}",
        f"**الموقع:** {p.district_ar}، {p.city_ar}",
        f"**تاريخ الإطلاق:** {_month_ar(p.launch)}",
        f"**موعد التسليم المتوقع:** {_month_ar(p.delivery)}",
        f"**إجمالي الوحدات:** {stats['total']} وحدة",
        "",
        "## نبذة عن المشروع",
        p.highlights_ar,
        "",
        "## أنواع الوحدات والمساحات",
    ]
    for utype, n, avail, amin, amax, pmin, pmax in stats["by_type"]:
        lines.append(
            f"- **{UNIT_TYPES[utype]['ar']}**: {n} وحدة، المساحات من {int(amin)} إلى {int(amax)} م²، "
            f"الأسعار تبدأ من {_fmt_sar(pmin)} ريال وتصل إلى {_fmt_sar(pmax)} ريال."
        )
    lines += [
        "",
        "## المرافق والخدمات",
        *[f"- {a}" for a in p.amenities_ar],
        "",
        "## مواصفات التشطيب",
        "- تشطيب فاخر بمواد عالية الجودة، أرضيات بورسلان ورخام في المداخل.",
        "- مطابخ مجهزة بخزائن وأسطح كوارتز.",
        "- تكييف مركزي (سبلت مخفي) في جميع الغرف.",
        "- نظام منزل ذكي أساسي (إضاءة وقفل ذكي).",
        "- ضمان إنشائي 10 سنوات وضمان تشطيبات سنة واحدة.",
        "",
        "## الموقع",
        f"يقع المشروع في {p.district_ar} بمدينة {p.city_ar}، مع سهولة الوصول إلى الطرق الرئيسية والخدمات.",
        "",
        f"للاستفسار والحجز: {DEVELOPER_PHONE} – {DEVELOPER_EMAIL}",
    ]
    return "\n".join(lines)


def brochure_en(p, stats) -> str:
    lines = [
        _front_matter(doc_type="brochure", project_id=p.id, project=p.name_en, lang="en"),
        f"# {p.name_en} – {p.city_en}",
        "",
        f"**Developer:** {DEVELOPER_EN}",
        f"**Location:** {p.district_en}, {p.city_en}",
        f"**Launch:** {_month_en(p.launch)}",
        f"**Expected handover:** {_month_en(p.delivery)}",
        f"**Total units:** {stats['total']}",
        "",
        "## About the project",
        p.highlights_en,
        "",
        "## Unit types and sizes",
    ]
    for utype, n, avail, amin, amax, pmin, pmax in stats["by_type"]:
        lines.append(
            f"- **{UNIT_TYPES[utype]['en']}**: {n} units, {int(amin)}–{int(amax)} sqm, "
            f"prices from SAR {_fmt_sar(pmin)} to SAR {_fmt_sar(pmax)}."
        )
    lines += [
        "",
        "## Amenities",
        *[f"- {a}" for a in p.amenities_en],
        "",
        "## Finishing specifications",
        "- Premium finishes: porcelain flooring, marble entrance lobbies.",
        "- Fitted kitchens with quartz worktops.",
        "- Concealed split air-conditioning in all rooms.",
        "- Basic smart-home package (lighting and smart lock).",
        "- 10-year structural warranty and 1-year finishing warranty.",
        "",
        "## Location",
        f"The project is located in {p.district_en}, {p.city_en}, with quick access to main roads and services.",
        "",
        f"Enquiries and booking: {DEVELOPER_PHONE} – {DEVELOPER_EMAIL}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Payment plans
# --------------------------------------------------------------------------- #
def payment_ar(p) -> str:
    return "\n".join([
        _front_matter(doc_type="payment_plan", project_id=p.id, project=p.name_ar, lang="ar"),
        f"# خطة السداد – مشروع {p.name_ar}",
        "",
        "## خطة السداد القياسية (البيع على الخارطة)",
        "| الدفعة | النسبة | التوقيت |",
        "|---|---|---|",
        "| دفعة الحجز | 10% | عند توقيع عقد البيع |",
        "| الدفعة الثانية | 15% | خلال 90 يوماً من توقيع العقد |",
        "| الدفعة الثالثة | 25% | عند إنجاز 50% من أعمال الإنشاء |",
        "| الدفعة الأخيرة | 50% | عند التسليم أو عبر التمويل البنكي |",
        "",
        "## رسوم الحجز",
        "رسوم الحجز 20,000 ريال، تُخصم من دفعة الحجز عند توقيع العقد، وتُسترد بالكامل إذا تم إلغاء الحجز خلال 14 يوماً.",
        "",
        "## التمويل العقاري",
        f"يتوفر تمويل عقاري من خلال شركائنا: {'، '.join(BANKS_AR)}. "
        "يمكن للعملاء المستفيدين من برنامج سكني الحصول على التمويل المدعوم وفق شروط البرنامج.",
        "",
        "## ضريبة التصرفات العقارية",
        "تُطبق ضريبة التصرفات العقارية بنسبة 5% من قيمة العقد. يُعفى المسكن الأول للمواطن من الضريبة حتى مليون ريال "
        "وفق ضوابط وزارة الإسكان، ويتم التحقق من الأهلية عبر منصة سكني.",
        "",
        "## الخصومات",
        "- خصم 2% عند السداد النقدي الكامل خلال 30 يوماً.",
        "- خصم 1% لموظفي الجهات الحكومية والشركات الشريكة.",
        "- لا يمكن الجمع بين أكثر من خصم واحد.",
        "",
        f"موعد التسليم المتوقع: {_month_ar(p.delivery)}.",
    ])


def payment_en(p) -> str:
    return "\n".join([
        _front_matter(doc_type="payment_plan", project_id=p.id, project=p.name_en, lang="en"),
        f"# Payment Plan – {p.name_en}",
        "",
        "## Standard off-plan payment schedule",
        "| Instalment | Share | Timing |",
        "|---|---|---|",
        "| Booking payment | 10% | On signing the sale contract |",
        "| Second payment | 15% | Within 90 days of signing |",
        "| Third payment | 25% | At 50% construction completion |",
        "| Final payment | 50% | On handover, or via bank financing |",
        "",
        "## Booking fee",
        "The booking fee is SAR 20,000. It is deducted from the booking payment on contract signing and is fully "
        "refundable if the booking is cancelled within 14 days.",
        "",
        "## Mortgage financing",
        f"Financing is available through our partner banks: {', '.join(BANKS_EN)}. "
        "Buyers eligible under the Sakani programme can obtain subsidised financing subject to programme rules.",
        "",
        "## Real Estate Transaction Tax (RETT)",
        "RETT of 5% applies to the contract value. A citizen's first home is exempt up to SAR 1,000,000 under the "
        "Ministry of Housing rules; eligibility is verified through the Sakani platform.",
        "",
        "## Discounts",
        "- 2% discount for full cash payment within 30 days.",
        "- 1% discount for government and partner-company employees.",
        "- Discounts cannot be combined.",
        "",
        f"Expected handover: {_month_en(p.delivery)}.",
    ])


# --------------------------------------------------------------------------- #
# Company-wide documents
# --------------------------------------------------------------------------- #
def faq_ar() -> str:
    return "\n".join([
        _front_matter(doc_type="faq", project_id="ALL", project="عام", lang="ar"),
        "# الأسئلة الشائعة",
        "",
        "## كيف أحجز وحدة؟",
        "1. اختر المشروع والوحدة عبر مركز المبيعات أو الموقع الإلكتروني.",
        "2. سدد رسوم الحجز (20,000 ريال) وقدّم صورة الهوية الوطنية أو الإقامة.",
        "3. وقّع عقد البيع خلال 14 يوماً من الحجز وسدد دفعة الحجز 10%.",
        "",
        "## ما المستندات المطلوبة لتوقيع العقد؟",
        "الهوية الوطنية أو الإقامة سارية المفعول، وخطاب الموافقة المبدئية من البنك في حال التمويل، ورقم الأهلية من منصة سكني إن وجد.",
        "",
        "## هل يمكن إلغاء الحجز؟",
        "نعم. يُسترد مبلغ الحجز بالكامل إذا تم الإلغاء خلال 14 يوماً. بعد توقيع العقد تُطبق سياسة الإلغاء: "
        "خصم 5% من قيمة العقد كرسوم إدارية إذا كان الإلغاء قبل بدء الإنشاء، و10% بعد ذلك.",
        "",
        "## هل المشاريع مرخصة للبيع على الخارطة؟",
        "نعم، جميع مشاريعنا مرخصة من برنامج البيع أو التأجير على الخارطة (وافي) وتُودع الدفعات في حساب ضمان مستقل.",
        "",
        "## ما هي رسوم الصيانة؟",
        "رسوم خدمات المجتمع السنوية 12 ريالاً لكل متر مربع للشقق، و8 ريالات لكل متر مربع للفلل والتاون هاوس، تُدفع بعد التسليم.",
        "",
        "## هل يمكن تعديل التصميم الداخلي؟",
        "يمكن اختيار أحد باقات التشطيب الثلاث (كلاسيك، مودرن، فاخر) قبل مرحلة التشطيب، ولا تُقبل تعديلات إنشائية.",
        "",
        "## هل تقبلون المستفيدين من برنامج سكني؟",
        "نعم، جميع مشاريعنا مسجلة في منصة سكني ويمكن استخدام القرض المدعوم ضمن الشروط.",
        "",
        "## كيف أتابع نسبة الإنجاز؟",
        "يتم إرسال تقرير إنجاز ربع سنوي بالصور عبر البريد الإلكتروني، ويمكن زيارة الموقع بموعد مسبق.",
    ])


def faq_en() -> str:
    return "\n".join([
        _front_matter(doc_type="faq", project_id="ALL", project="General", lang="en"),
        "# Frequently Asked Questions",
        "",
        "## How do I book a unit?",
        "1. Choose the project and unit through the sales centre or website.",
        "2. Pay the booking fee (SAR 20,000) and provide a copy of your national ID or Iqama.",
        "3. Sign the sale contract within 14 days of booking and pay the 10% booking payment.",
        "",
        "## What documents are required to sign?",
        "A valid national ID or Iqama, a bank pre-approval letter if financing, and the Sakani eligibility number if applicable.",
        "",
        "## Can I cancel a booking?",
        "Yes. The booking fee is fully refunded if you cancel within 14 days. After signing the contract, the "
        "cancellation policy applies: a 5% administrative charge before construction starts, 10% afterwards.",
        "",
        "## Are the projects licensed for off-plan sales?",
        "Yes. All projects are licensed under the Wafi off-plan sales programme and payments are held in an independent escrow account.",
        "",
        "## What are the service charges?",
        "Annual community service charges are SAR 12 per sqm for apartments and SAR 8 per sqm for villas and townhouses, payable after handover.",
        "",
        "## Can I customise the interior?",
        "You may choose one of three finishing packages (Classic, Modern, Premium) before the finishing stage. Structural changes are not accepted.",
        "",
        "## Do you accept Sakani beneficiaries?",
        "Yes. All projects are registered on the Sakani platform and the subsidised loan can be used subject to programme terms.",
        "",
        "## How can I track construction progress?",
        "A quarterly progress report with photos is emailed to buyers, and site visits can be arranged by appointment.",
    ])


def handover_ar() -> str:
    return "\n".join([
        _front_matter(doc_type="policy", project_id="ALL", project="عام", lang="ar"),
        "# سياسة التسليم والضمان",
        "",
        "## إجراءات التسليم",
        "1. إشعار العميل قبل 30 يوماً من موعد التسليم.",
        "2. معاينة الوحدة مع مهندس المشروع وتوثيق الملاحظات في محضر المعاينة.",
        "3. معالجة الملاحظات خلال 21 يوماً ثم إعادة المعاينة.",
        "4. سداد الدفعة الأخيرة أو استكمال إجراءات التمويل، ثم استلام المفاتيح وإفراغ الصك.",
        "",
        "## الضمانات",
        "- ضمان إنشائي (الهيكل الإنشائي) لمدة 10 سنوات من تاريخ التسليم.",
        "- ضمان الأعمال الكهربائية والميكانيكية لمدة سنتين.",
        "- ضمان التشطيبات والأعمال الظاهرة لمدة سنة واحدة.",
        "- لا يشمل الضمان الأضرار الناتجة عن سوء الاستخدام أو التعديلات التي ينفذها العميل.",
        "",
        "## التأخير في التسليم",
        "في حال تأخر التسليم عن الموعد المحدد في العقد لأكثر من 6 أشهر لأسباب تعود للمطور، يُعوض العميل بنسبة 0.5% من قيمة العقد عن كل شهر تأخير بحد أقصى 5%.",
    ])


def handover_en() -> str:
    return "\n".join([
        _front_matter(doc_type="policy", project_id="ALL", project="General", lang="en"),
        "# Handover and Warranty Policy",
        "",
        "## Handover procedure",
        "1. The buyer is notified 30 days before the handover date.",
        "2. The unit is inspected with the project engineer and any snags are recorded in the inspection report.",
        "3. Snags are rectified within 21 days, followed by a re-inspection.",
        "4. The final payment is settled (or financing completed), keys are handed over and the title deed is transferred.",
        "",
        "## Warranties",
        "- Structural warranty: 10 years from handover.",
        "- Electrical and mechanical works: 2 years.",
        "- Finishing and visible works: 1 year.",
        "- Damage from misuse or buyer modifications is excluded.",
        "",
        "## Late delivery",
        "If handover is delayed more than 6 months beyond the contract date for reasons attributable to the developer, "
        "the buyer is compensated 0.5% of the contract value per month of delay, capped at 5%.",
    ])


def about_ar() -> str:
    return "\n".join([
        _front_matter(doc_type="about", project_id="ALL", project="عام", lang="ar"),
        f"# عن {DEVELOPER_AR}",
        "",
        f"{DEVELOPER_AR} مطور عقاري سعودي (سجل تجاري {DEVELOPER_CR}) متخصص في تطوير المجتمعات السكنية المتكاملة "
        "في الرياض وجدة والمنطقة الشرقية والمدينة المنورة. نعمل بالشراكة مع وزارة الإسكان وبرنامج سكني لتوفير مساكن "
        "بجودة عالية وخطط سداد مرنة.",
        "",
        "## المشاريع الحالية",
        *[f"- {p.name_ar} – {p.city_ar} (تسليم {_month_ar(p.delivery)})" for p in PROJECTS],
        "",
        "## التواصل",
        f"- الهاتف الموحد: {DEVELOPER_PHONE}",
        f"- البريد الإلكتروني: {DEVELOPER_EMAIL}",
        "- مراكز المبيعات: الرياض (طريق الملك فهد)، جدة (طريق الأمير سلطان)، الخبر (الكورنيش).",
        "- ساعات العمل: السبت إلى الخميس من 9 صباحاً إلى 9 مساءً.",
    ])


def about_en() -> str:
    return "\n".join([
        _front_matter(doc_type="about", project_id="ALL", project="General", lang="en"),
        f"# About {DEVELOPER_EN}",
        "",
        f"{DEVELOPER_EN} (CR {DEVELOPER_CR}) is a Saudi developer of integrated residential communities in Riyadh, "
        "Jeddah, the Eastern Province and Madinah. We work with the Ministry of Housing and the Sakani programme to "
        "deliver quality homes with flexible payment plans.",
        "",
        "## Current projects",
        *[f"- {p.name_en} – {p.city_en} (handover {_month_en(p.delivery)})" for p in PROJECTS],
        "",
        "## Contact",
        f"- Unified number: {DEVELOPER_PHONE}",
        f"- Email: {DEVELOPER_EMAIL}",
        "- Sales centres: Riyadh (King Fahd Road), Jeddah (Prince Sultan Road), Al Khobar (Corniche).",
        "- Working hours: Saturday to Thursday, 9 am to 9 pm.",
    ])


# --------------------------------------------------------------------------- #
def generate_documents(docs_dir: Path = DOCS_DIR, db_path: Path = DB_PATH) -> dict:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for old in docs_dir.glob("*.md"):
        old.unlink()
    con = sqlite3.connect(db_path)
    written = []
    for p in PROJECTS:
        stats = _project_stats(con, p.id)
        for name, text in (
            (f"brochure_{p.id}_ar.md", brochure_ar(p, stats)),
            (f"brochure_{p.id}_en.md", brochure_en(p, stats)),
            (f"payment_{p.id}_ar.md", payment_ar(p)),
            (f"payment_{p.id}_en.md", payment_en(p)),
        ):
            (docs_dir / name).write_text(text, encoding="utf-8")
            written.append(name)
    for name, text in (
        ("faq_ar.md", faq_ar()), ("faq_en.md", faq_en()),
        ("handover_policy_ar.md", handover_ar()), ("handover_policy_en.md", handover_en()),
        ("about_ar.md", about_ar()), ("about_en.md", about_en()),
    ):
        (docs_dir / name).write_text(text, encoding="utf-8")
        written.append(name)
    con.close()
    return {"documents": len(written), "dir": str(docs_dir)}


if __name__ == "__main__":
    print(generate_documents())
