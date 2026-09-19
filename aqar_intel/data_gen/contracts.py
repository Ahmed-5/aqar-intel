"""Generate simulated off-plan sale contracts with ground-truth labels.

Four templates (three Arabic, one English) vary wording, layout, date
formats and even digit systems (Arabic-Indic digits in template B), so the
extractor is tested on realistic surface variation rather than a single
fixed layout. Every contract is paired with a ground-truth JSON record so
extraction accuracy can be measured, not eyeballed.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from ..arabic import to_western_digits
from ..config import CONTRACTS_DIR, DB_PATH
from .world import (
    BANKS_AR, BANKS_EN, DEVELOPER_AR, DEVELOPER_CR, DEVELOPER_EN, FAMILY_NAMES_AR, FAMILY_NAMES_EN,
    FIRST_NAMES_AR, FIRST_NAMES_EN, PROJECT_BY_ID, UNIT_TYPES,
)

_WESTERN_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
MONTHS_AR = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _ai(s: str) -> str:
    """Western digits -> Arabic-Indic digits."""
    return s.translate(_WESTERN_TO_ARABIC_INDIC)


def _sar(x: float) -> str:
    return f"{int(x):,}"


def _pick_unit(con: sqlite3.Connection, rng: random.Random) -> dict:
    rows = con.execute(
        "SELECT unit_id, project_id, unit_no, unit_type, area_sqm, list_price_sar, delivery_month FROM units WHERE status='sold'"
    ).fetchall()
    r = rng.choice(rows)
    return dict(zip(["unit_id", "project_id", "unit_no", "unit_type", "area_sqm", "list_price_sar", "delivery_month"], r))


def _payment_terms(rng: random.Random, total: int) -> dict:
    method = rng.choices(["installments", "bank_financing", "cash"], weights=[0.45, 0.4, 0.15])[0]
    if method == "cash":
        return dict(payment_method="cash", down_payment_sar=total, installments_count=0,
                    installment_amount_sar=0, bank_name=None)
    down = int(round(total * rng.choice([0.10, 0.15, 0.20]) / 1000) * 1000)
    remainder = total - down
    if method == "bank_financing":
        return dict(payment_method="bank_financing", down_payment_sar=down, installments_count=1,
                    installment_amount_sar=remainder, bank_name=rng.randrange(len(BANKS_AR)))
    n = rng.choice([4, 8, 12, 24])
    inst = remainder // n
    down = total - inst * n  # keep the identity total = down + n*inst exact
    return dict(payment_method="installments", down_payment_sar=down, installments_count=n,
                installment_amount_sar=inst, bank_name=None)


def _record(con: sqlite3.Connection, rng: random.Random, idx: int) -> dict:
    u = _pick_unit(con, rng)
    p = PROJECT_BY_ID[u["project_id"]]
    total = int(round(u["list_price_sar"] * rng.uniform(0.94, 1.0) / 1000) * 1000)
    dy, dm = int(u["delivery_month"][:4]), int(u["delivery_month"][5:])
    ddate = date(dy, dm, rng.choice([1, 15, 30 if dm != 2 else 28]))
    latest = min(date(2026, 8, 31), ddate - timedelta(days=60))
    cdate = date(2025, 1, 1) + timedelta(days=rng.randrange(0, max(1, (latest - date(2025, 1, 1)).days)))
    fi, li = rng.randrange(len(FIRST_NAMES_AR)), rng.randrange(len(FAMILY_NAMES_AR))
    nid = ("1" if rng.random() < 0.8 else "2") + "".join(str(rng.randrange(10)) for _ in range(9))
    phone = "05" + "".join(str(rng.randrange(10)) for _ in range(8))
    terms = _payment_terms(rng, total)
    return dict(
        contract_no=f"AUF-{cdate.year}-{idx:04d}",
        contract_date=cdate.isoformat(),
        buyer_name_ar=f"{FIRST_NAMES_AR[fi]} {FAMILY_NAMES_AR[li]}",
        buyer_name_en=f"{FIRST_NAMES_EN[fi]} {FAMILY_NAMES_EN[li]}",
        buyer_national_id=nid,
        buyer_phone=phone,
        project_id=p.id,
        project_name_ar=p.name_ar,
        project_name_en=p.name_en,
        city_ar=p.city_ar,
        city_en=p.city_en,
        unit_no=u["unit_no"],
        unit_type=u["unit_type"],
        area_sqm=float(u["area_sqm"]),
        total_price_sar=total,
        delivery_date=ddate.isoformat(),
        **{k: v for k, v in terms.items() if k != "bank_name"},
        bank_index=terms["bank_name"],
    )


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
def _date_slash(d: str) -> str:  # 14/03/2026
    y, m, dd = d.split("-")
    return f"{dd}/{m}/{y}"


def _date_words_ar(d: str) -> str:  # ١٤ مارس ٢٠٢٦
    y, m, dd = d.split("-")
    return _ai(f"{int(dd)} {MONTHS_AR[int(m) - 1]} {y}")


def _date_words_en(d: str) -> str:
    y, m, dd = d.split("-")
    return f"{int(dd)} {MONTHS_EN[int(m) - 1]} {y}"


def _payment_clause_ar(r: dict, digits=lambda s: s, subject: str = "الطرف الثاني") -> str:
    bank = BANKS_AR[r["bank_index"]] if r["bank_index"] is not None else None
    if r["payment_method"] == "cash":
        return digits(f"يُسدد {subject} كامل الثمن وقدره {_sar(r['total_price_sar'])} ريال نقداً عند توقيع هذا العقد.")
    if r["payment_method"] == "bank_financing":
        return digits(
            f"يُسدد {subject} دفعة مقدمة قدرها {_sar(r['down_payment_sar'])} ريال عند التوقيع، "
            f"ويُسدد المبلغ المتبقي وقدره {_sar(r['installment_amount_sar'])} ريال عن طريق التمويل العقاري من {bank}."
        )
    return digits(
        f"يُسدد {subject} دفعة مقدمة قدرها {_sar(r['down_payment_sar'])} ريال عند التوقيع، "
        f"ويُسدد المبلغ المتبقي على {r['installments_count']} أقساط متساوية قيمة كل قسط {_sar(r['installment_amount_sar'])} ريال."
    )


def template_a(r: dict) -> str:
    """Formal clause-based Arabic contract, Western digits, dd/mm/yyyy dates."""
    ut = UNIT_TYPES[r["unit_type"]]["ar"]
    return f"""عقد بيع وحدة سكنية على الخارطة

رقم العقد: {r['contract_no']}
تاريخ العقد: {_date_slash(r['contract_date'])}م

الطرف الأول (البائع): {DEVELOPER_AR}، سجل تجاري رقم {DEVELOPER_CR}، ويمثلها في التوقيع مدير المبيعات.
الطرف الثاني (المشتري): السيد/ {r['buyer_name_ar']}، هوية وطنية رقم {r['buyer_national_id']}، جوال {r['buyer_phone']}.

البند الأول – محل العقد:
باع الطرف الأول للطرف الثاني القابل لذلك الوحدة رقم {r['unit_no']} من نوع {ut} في مشروع {r['project_name_ar']} بمدينة {r['city_ar']}، بمساحة إجمالية قدرها {r['area_sqm']:g} متر مربع، وفق المخططات والمواصفات المرفقة.

البند الثاني – الثمن:
اتفق الطرفان على أن الثمن الإجمالي للوحدة هو {_sar(r['total_price_sar'])} ريال سعودي غير شامل ضريبة التصرفات العقارية.

البند الثالث – السداد:
{_payment_clause_ar(r)}

البند الرابع – التسليم:
يلتزم الطرف الأول بتسليم الوحدة جاهزة للسكن في موعد أقصاه {_date_slash(r['delivery_date'])}م، مع مراعاة أحكام القوة القاهرة.

البند الخامس – الضمانات:
يضمن الطرف الأول الهيكل الإنشائي لمدة عشر سنوات والتشطيبات لمدة سنة من تاريخ التسليم.

البند السادس – أحكام عامة:
تخضع أحكام هذا العقد لأنظمة برنامج البيع أو التأجير على الخارطة (وافي)، وتُودع المبالغ المسددة في حساب الضمان الخاص بالمشروع.

حُرر هذا العقد من نسختين بيد كل طرف نسخة للعمل بموجبها.

الطرف الأول: ____________        الطرف الثاني: ____________
"""


def template_b(r: dict) -> str:
    """Letter-style Arabic contract with Arabic-Indic digits and written-out dates."""
    ut = UNIT_TYPES[r["unit_type"]]["ar"]
    return f"""بسم الله الرحمن الرحيم

{DEVELOPER_AR}
عقد بيع وحدة عقارية

إنه في يوم {_date_words_ar(r['contract_date'])} تم الاتفاق بين كل من:

أولاً: {DEVELOPER_AR} (س.ت {_ai(DEVELOPER_CR)}) ويشار إليها فيما بعد بـ"المطور".
ثانياً: {r['buyer_name_ar']}، {'سعودي الجنسية' if r['buyer_national_id'].startswith('1') else 'مقيم'}، بموجب {'الهوية' if r['buyer_national_id'].startswith('1') else 'الإقامة'} رقم {_ai(r['buyer_national_id'])}، هاتف {_ai(r['buyer_phone'])}، ويشار إليه فيما بعد بـ"المشتري".

مرجع العقد: {r['contract_no']}

تمهيد: يمتلك المطور مشروع {r['project_name_ar']} الكائن في {r['city_ar']}، وقد رغب المشتري في شراء إحدى وحداته.

بناءً عليه اتفق الطرفان على ما يلي:

١. يبيع المطور للمشتري {ut} رقم {r['unit_no']} بمساحة {_ai(f"{r['area_sqm']:g}")} م٢ في مشروع {r['project_name_ar']}.
٢. قيمة البيع الإجمالية {_ai(_sar(r['total_price_sar']))} ريال سعودي.
٣. {_payment_clause_ar(r, digits=_ai, subject="المشتري")}
٤. موعد تسليم الوحدة: {_date_words_ar(r['delivery_date'])}.
٥. يلتزم المطور بضمان الهيكل الإنشائي عشر سنوات والتشطيبات سنة واحدة.
٦. أي نزاع ينشأ عن هذا العقد يُحال إلى لجان الفصل في المنازعات العقارية.

توقيع المطور: ____________
توقيع المشتري: ____________
"""


def template_c(r: dict) -> str:
    """Compact contract summary sheet (as produced by a CRM), ISO dates, key: value layout."""
    ut = UNIT_TYPES[r["unit_type"]]["ar"]
    bank = BANKS_AR[r["bank_index"]] if r["bank_index"] is not None else "لا ينطبق"
    method_ar = {"installments": "أقساط", "bank_financing": "تمويل بنكي", "cash": "نقدي"}[r["payment_method"]]
    return f"""ملخص عقد بيع – نظام إدارة المبيعات
{DEVELOPER_AR}

رقم العقد          : {r['contract_no']}
تاريخ التوقيع      : {r['contract_date']}
اسم المشتري        : {r['buyer_name_ar']}
رقم الهوية         : {r['buyer_national_id']}
رقم الجوال         : {r['buyer_phone']}
المشروع            : {r['project_name_ar']} – {r['city_ar']}
رقم الوحدة         : {r['unit_no']}
نوع الوحدة         : {ut}
المساحة            : {r['area_sqm']:g} م²
إجمالي الثمن       : {_sar(r['total_price_sar'])} ر.س
طريقة السداد       : {method_ar}
الدفعة المقدمة     : {_sar(r['down_payment_sar'])} ر.س
عدد الأقساط        : {r['installments_count']}
قيمة القسط         : {_sar(r['installment_amount_sar'])} ر.س
جهة التمويل        : {bank}
تاريخ التسليم      : {r['delivery_date']}

ملاحظات: الثمن غير شامل ضريبة التصرفات العقارية (5%). تُطبق سياسة الإلغاء المعتمدة.
"""


def template_d(r: dict) -> str:
    """English off-plan sale agreement."""
    ut = UNIT_TYPES[r["unit_type"]]["en"]
    bank = BANKS_EN[r["bank_index"]] if r["bank_index"] is not None else None
    if r["payment_method"] == "cash":
        pay = f"The Purchaser shall pay the full Purchase Price of SAR {_sar(r['total_price_sar'])} in cash upon signing this Agreement."
    elif r["payment_method"] == "bank_financing":
        pay = (f"The Purchaser shall pay a down payment of SAR {_sar(r['down_payment_sar'])} upon signing, and the balance of "
               f"SAR {_sar(r['installment_amount_sar'])} through mortgage financing provided by {bank}.")
    else:
        pay = (f"The Purchaser shall pay a down payment of SAR {_sar(r['down_payment_sar'])} upon signing, and the balance in "
               f"{r['installments_count']} equal instalments of SAR {_sar(r['installment_amount_sar'])} each.")
    return f"""OFF-PLAN SALE AGREEMENT

Agreement No.: {r['contract_no']}
Date: {_date_words_en(r['contract_date'])}

BETWEEN
(1) {DEVELOPER_EN}, Commercial Registration No. {DEVELOPER_CR} (the "Developer"); and
(2) Mr./Ms. {r['buyer_name_en']}, National ID No. {r['buyer_national_id']}, Mobile {r['buyer_phone']} (the "Purchaser").

1. SUBJECT
The Developer sells to the Purchaser {ut} No. {r['unit_no']} in the {r['project_name_en']} project, {r['city_en']}, with a gross area of {r['area_sqm']:g} square metres.

2. PURCHASE PRICE
The total Purchase Price is SAR {_sar(r['total_price_sar'])}, exclusive of Real Estate Transaction Tax.

3. PAYMENT
{pay}

4. HANDOVER
The Developer shall hand over the Unit ready for occupation no later than {_date_words_en(r['delivery_date'])}, subject to force majeure.

5. WARRANTIES
Ten-year structural warranty and one-year finishing warranty from the handover date.

6. GENERAL
This Agreement is governed by the Wafi off-plan sales regulations. All payments are deposited in the project escrow account.

Signed for the Developer: ____________      Signed by the Purchaser: ____________
"""


def template_e(r: dict) -> str:
    """Free-form sale confirmation letter (held-out layout: the rule-based baseline has no parser for it)."""
    ut = UNIT_TYPES[r["unit_type"]]["ar"]
    bank = BANKS_AR[r["bank_index"]] if r["bank_index"] is not None else None
    if r["payment_method"] == "cash":
        pay = "وقد سُدد كامل المبلغ نقداً عند التوقيع"
    elif r["payment_method"] == "bank_financing":
        pay = (f"وقد استلمنا دفعة أولى بمبلغ {_sar(r['down_payment_sar'])} ريال، على أن يُسدد المتبقي وقدره "
               f"{_sar(r['installment_amount_sar'])} ريال عبر تمويل عقاري من {bank}")
    else:
        pay = (f"وقد استلمنا دفعة أولى بمبلغ {_sar(r['down_payment_sar'])} ريال، ويُقسّط المتبقي على "
               f"{r['installments_count']} دفعات متساوية بقيمة {_sar(r['installment_amount_sar'])} ريال لكل دفعة")
    return f"""{DEVELOPER_AR}
إشعار تأكيد بيع وحدة

التاريخ: {_date_words_ar(r['contract_date'])}

عزيزنا العميل / {r['buyer_name_ar']}،

نشكر لكم ثقتكم بنا. يسرنا تأكيد إتمام بيع {ut} رقم {r['unit_no']} بمساحة {r['area_sqm']:g} متراً مربعاً ضمن مشروع {r['project_name_ar']} في مدينة {r['city_ar']}، وذلك بموجب عقد البيع رقم {r['contract_no']} الموقع في التاريخ أعلاه.

الثمن الإجمالي المتفق عليه هو {_sar(r['total_price_sar'])} ريال سعودي، {pay}. ومن المتوقع تسليم الوحدة بتاريخ {_date_words_ar(r['delivery_date'])} بإذن الله.

بياناتكم المسجلة لدينا: رقم الهوية {r['buyer_national_id']}، رقم الجوال {r['buyer_phone']}. نرجو التواصل معنا في حال وجود أي تعديل.

مع خالص التحية،
إدارة المبيعات
"""


TEMPLATES = {"A": template_a, "B": template_b, "C": template_c, "D": template_d, "E": template_e}


def ground_truth(r: dict, template: str) -> dict:
    lang = "en" if template == "D" else "ar"
    bank = None
    if r["bank_index"] is not None:
        bank = (BANKS_EN if lang == "en" else BANKS_AR)[r["bank_index"]]
    return dict(
        contract_no=r["contract_no"],
        contract_date=r["contract_date"],
        buyer_name=r["buyer_name_en"] if lang == "en" else r["buyer_name_ar"],
        buyer_national_id=r["buyer_national_id"],
        buyer_phone=r["buyer_phone"],
        project_name=r["project_name_en"] if lang == "en" else r["project_name_ar"],
        city=r["city_en"] if lang == "en" else r["city_ar"],
        unit_no=r["unit_no"],
        unit_type=r["unit_type"],
        area_sqm=r["area_sqm"],
        total_price_sar=r["total_price_sar"],
        payment_method=r["payment_method"],
        down_payment_sar=r["down_payment_sar"],
        installments_count=r["installments_count"],
        installment_amount_sar=r["installment_amount_sar"],
        bank_name=bank,
        delivery_date=r["delivery_date"],
        language=lang,
        template=template,
        planted_issue=r.get("planted_issue"),
    )


# contract index -> issue. Chosen so that the affected contracts are non-cash and spread across templates.
PLANTED_ISSUES = {8: "payments_do_not_sum", 21: "malformed_national_id", 34: "delivery_before_contract"}


def generate_contracts(n: int = 40, out_dir: Path = CONTRACTS_DIR, db_path: Path = DB_PATH, seed: int = 42) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.txt"):
        old.unlink()
    con = sqlite3.connect(db_path)
    truth: dict[str, dict] = {}
    templates = list(TEMPLATES)
    for i in range(1, n + 1):
        r = _record(con, rng, i)
        r["planted_issue"] = None
        if i in PLANTED_ISSUES:  # deliberately inconsistent records for the validator to catch
            r["planted_issue"] = PLANTED_ISSUES[i]
            if r["planted_issue"] == "payments_do_not_sum" and r["payment_method"] != "cash":
                r["installment_amount_sar"] += 5000
            elif r["planted_issue"] == "malformed_national_id":
                r["buyer_national_id"] = r["buyer_national_id"][:9]
            elif r["planted_issue"] == "delivery_before_contract":
                r["delivery_date"] = (date.fromisoformat(r["contract_date"]) - timedelta(days=45)).isoformat()
        t = templates[(i - 1) % len(templates)]
        text = TEMPLATES[t](r)
        fname = f"contract_{i:03d}_{t}.txt"
        (out_dir / fname).write_text(text, encoding="utf-8")
        truth[fname] = ground_truth(r, t)
    con.close()
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"contracts": n, "dir": str(out_dir), "templates": templates}


if __name__ == "__main__":
    print(generate_contracts())
