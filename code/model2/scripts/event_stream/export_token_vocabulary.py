"""Export the STREAM token vocabulary (union over the 5 fold vocabularies) with descriptions."""
import csv
import json
from pathlib import Path
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
DATA = ROOT / "data"
RES = ROOT / "result/event_stream/transformer"
OUT = ROOT / "result/supplementary"

SPECIAL = {
    "[PAD]": "Padding token (not attended)",
    "[UNK]": "Code not present in the training vocabulary",
    "[MASK]": "Placeholder for masked events in the masked-event modeling (MEM) objective",
    "[BASE]": "Dedicated surgery-index summary token; its output gives the baseline PO-AKI prediction",
    "[SURGERY]": "Generic index-surgery token replacing procedure identity (generic-surgery ablation only)",
}
BUCKET = {
    "COND_OTHERS": "Any condition whose CCS category is outside the retained CCS set",
    "MED_OTHERS": "Any medication whose RxCUI is outside the retained medication list",
    "LAB_OTHERS": "Any laboratory/vital-sign measurement outside the retained measurement list",
}
# OMOP standard Race/Gender concept names.
OMOP_DEMO = {
    "SEX|8507": "Male", "SEX|8532": "Female",
    "RACE|0": "Unknown / not reported",
    "RACE|8527": "White", "RACE|8516": "Black or African American", "RACE|8515": "Asian",
    "RACE|8557": "Native Hawaiian or Other Pacific Islander",
    "RACE|8657": "American Indian or Alaska Native",
    "RACE|38003574": "Asian Indian", "RACE|38003579": "Chinese", "RACE|38003581": "Filipino",
    "RACE|38003584": "Japanese", "RACE|38003585": "Korean", "RACE|38003592": "Vietnamese",
    "RACE|38003613": "Other Pacific Islander",
}
# Measurement names and units as recorded in the source lab table.
LAB_INFO = {
    "1004025": ("Heart rate", "beats/min"),
    "3000905": ("Leukocytes [#/volume] in Blood by Automated count", "10^3/µL"),
    "3000963": ("Hemoglobin [Mass/volume] in Blood", "g/dL"),
    "3004249": ("Systolic blood pressure", "mmHg"),
    "3004501": ("Glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
    "3005424": ("Body surface area", "m²"),
    "3006906": ("Calcium [Mass/volume] in Serum or Plasma", "mg/dL"),
    "3012888": ("Diastolic blood pressure", "mmHg"),
    "3013682": ("Urea nitrogen [Mass/volume] in Serum or Plasma", "mg/dL"),
    "3014576": ("Chloride [Moles/volume] in Serum or Plasma", "mmol/L"),
    "3015632": ("Carbon dioxide, total [Moles/volume] in Serum or Plasma", "mmol/L"),
    "3016723": ("Creatinine [Mass/volume] in Serum or Plasma", "mg/dL"),
    "3019550": ("Sodium [Moles/volume] in Serum or Plasma", "mmol/L"),
    "3020891": ("Body temperature", "°F"),
    "3023103": ("Potassium [Moles/volume] in Serum or Plasma", "mmol/L"),
    "3023314": ("Hematocrit [Volume Fraction] of Blood by Automated count", "%"),
    "3024561": ("Albumin [Mass/volume] in Serum or Plasma", "g/dL"),
    "3024929": ("Platelets [#/volume] in Blood by Automated count", "10^3/µL"),
    "3025315": ("Body weight", "lb"),
    "3036277": ("Body height", "in"),
    "3038553": ("Body mass index (BMI) [Ratio]", "kg/m²"),
}
CATEGORY_ORDER = ["Special", "Demographic", "Index procedure", "Condition", "Medication", "Measurement"]


def load_names():
    ccs = {}
    with open(DATA / "ccs_codes.csv") as f:
        for r in csv.DictReader(f):
            ccs[f"CCS|{r['ccs_code']}"] = " ".join(r["definition"].split())
    rx = {}
    with open(DATA / "rxcui_to_name.csv") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                rx[row[0]] = row[1]
    snomed = {k: v.removesuffix(" (procedure)") for k, v in json.loads((DATA / "snomed_procedure_names.json").read_text()).items()}
    return ccs, rx, snomed


def describe(tok, ccs, rx, snomed):
    """Return (category, vocabulary, code, description, value)."""
    if tok in SPECIAL:
        return "Special", "—", "—", SPECIAL[tok], "—"
    if tok == "AGE":
        return "Demographic", "—", "—", "Age at surgery", "Age (years), z-scored"
    if tok in BUCKET:
        cat = {"COND": "Condition", "MED": "Medication", "LAB": "Measurement"}[tok.split("_")[0]]
        return cat, "Other bucket", "—", BUCKET[tok], "Measured value, z-scored" if cat == "Measurement" else "—"
    vocab, code = tok.split("|", 1)
    if vocab == "SEX":
        return "Demographic", "OMOP Gender", code, f"Sex: {OMOP_DEMO[tok]}", "—"
    if vocab == "RACE":
        return "Demographic", "OMOP Race", code, f"Race: {OMOP_DEMO[tok]}", "—"
    if vocab == "SNOMED":
        return "Index procedure", "SNOMED CT", code, snomed.get(tok, ""), "—"
    if vocab == "CCS":
        return "Condition", "CCS (single-level diagnosis)", code, ccs.get(tok, ""), "—"
    if vocab == "RXCUI":
        return "Medication", "RxNorm ingredient (RxCUI)", code, rx.get(tok, ""), "Start/end as event type"
    if vocab == "LAB":
        name, unit = LAB_INFO.get(code, ("", ""))
        return "Measurement", "OMOP Measurement concept", code, name, f"Measured value ({unit}), z-scored"
    raise ValueError(tok)


def main():
    vocabs = [set(json.loads((RES / f"fold{k}/code_vocab.json").read_text())) for k in range(1, 6)]
    names = load_names()
    rows = []
    for tok in set.union(*vocabs):
        cat, voc, code, desc, val = describe(tok, *names)
        if not desc:
            raise ValueError(f"missing description for {tok}")
        rows.append({"Category": cat, "Token": tok, "Vocabulary": voc, "Code": code, "Description": desc,
                     "Value input": val})

    special_order = list(SPECIAL)
    demo_order = ["AGE", "SEX", "RACE"]

    def key(r):
        c, t = r["Code"], r["Token"]
        sub = special_order.index(t) if t in SPECIAL else demo_order.index(t.split("|")[0]) if r["Category"] == "Demographic" else 0
        return (CATEGORY_ORDER.index(r["Category"]), r["Vocabulary"] == "Other bucket", sub,
                int(c) if c.isdigit() else -1, t)

    rows.sort(key=key)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "STREAM_token_vocabulary.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    counts = {c: sum(r["Category"] == c for r in rows) for c in CATEGORY_ORDER}
    print(out, len(rows), counts)


if __name__ == "__main__":
    main()
