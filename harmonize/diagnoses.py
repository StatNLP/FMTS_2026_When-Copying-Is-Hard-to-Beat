"""
harmonize/diagnoses.py
Shared utilities for ICD-9/10 comorbidity scoring.

Implements:
  - Charlson Comorbidity Index (Quan 2005 ICD-9 and ICD-10 mappings)
  - SAPS-II chronic_disease component (metastatic cancer, haematologic
    malignancy, AIDS — Le Gall 1993)

References
----------
- Quan H. et al. (2005). "Coding Algorithms for Defining Comorbidities in
  ICD-9-CM and ICD-10 Administrative Data." Med Care 43(11):1130-9.
- Deyo R.A. et al. (1992). "Adapting a clinical comorbidity index for use
  with ICD-9-CM administrative databases." J Clin Epidemiol 45(6):613-9.
- Le Gall J.R. et al. (1993). "SAPS II." JAMA 270(24):2957-63.

ICD code format
---------------
Codes are passed as strings WITHOUT dots (e.g. '41001' not '410.01').
Normalisation is applied internally.
"""

from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(code: str) -> str:
    """Strip dots, spaces, leading/trailing zeros; uppercase."""
    return re.sub(r'[\.\s]', '', str(code).strip().upper())


def _startswith_any(code: str, prefixes: Sequence[str]) -> bool:
    return any(code.startswith(p) for p in prefixes)


# ---------------------------------------------------------------------------
# Charlson ICD-9 mapping (Quan 2005, Table 1)
# Weights follow Quan 2011 revised weights.
# ---------------------------------------------------------------------------

# Each entry: (condition_name, Charlson_weight, [ICD-9 prefixes/exact codes])
_CHARLSON_ICD9: list[tuple[str, int, list[str]]] = [
    ("myocardial_infarction",   1, ["410", "412"]),
    ("congestive_heart_failure", 1, ["4254", "4255", "4257", "4258", "4259",
                                     "428"]),
    ("peripheral_vascular",     1, ["0930", "4373", "440", "441",
                                     "4431", "4432", "4438", "4439",
                                     "4471", "5571", "5579", "V434"]),
    ("cerebrovascular",         1, ["36234", "430", "431", "432", "433",
                                     "434", "435", "436", "437", "438"]),
    ("dementia",                1, ["290", "2941", "3312"]),
    ("chronic_pulmonary",       1, ["4168", "4169", "490", "491", "492",
                                     "493", "494", "495", "496", "500",
                                     "501", "502", "503", "504", "505",
                                     "5064", "5081", "5088"]),
    ("rheumatic_disease",       1, ["4465", "7100", "7101", "7102", "7103",
                                     "7104", "7140", "7141", "7142", "71481",
                                     "725"]),
    ("peptic_ulcer",            1, ["531", "532", "533", "534"]),
    ("mild_liver_disease",      1, ["07022", "07023", "07032", "07033",
                                     "07044", "07054", "0706", "0709",
                                     "5710", "5712", "5713", "5715",
                                     "5716", "5718", "5719", "5733",
                                     "5734", "5738", "5739", "V427"]),
    ("diabetes_uncomplicated",  1, ["2500", "2501", "2502", "2503",
                                     "2508", "2509"]),
    ("diabetes_complicated",    2, ["2504", "2505", "2506", "2507"]),
    ("hemiplegia",              2, ["3341", "342", "343", "3440", "3441",
                                     "3442", "3443", "3444", "3445",
                                     "3446", "3449"]),
    ("renal_disease",           2, ["40301", "40311", "40391", "40402",
                                     "40403", "40412", "40413", "40492",
                                     "40493", "582", "5830", "5831",
                                     "5832", "5834", "5836", "5837",
                                     "585", "586", "5880", "V420",
                                     "V451", "V56"]),
    ("malignancy",              2, ["140", "141", "142", "143", "144",
                                     "145", "146", "147", "148", "149",
                                     "150", "151", "152", "153", "154",
                                     "155", "156", "157", "158", "159",
                                     "160", "161", "162", "163", "164",
                                     "165", "166", "167", "168", "169",
                                     "170", "171", "172", "174", "175",
                                     "176", "179", "180", "181", "182",
                                     "183", "184", "185", "186", "187",
                                     "188", "189", "190", "191", "192",
                                     "193", "194", "195",
                                     # Haematologic (Quan 2005: 200.0–208.9)
                                     "200", "201", "202", "203", "204",
                                     "205", "206", "207", "208"]),
    ("mod_severe_liver",        3, ["4560", "4561", "4562", "5722",
                                     "5723", "5724", "5725", "5726",
                                     "5727", "5728"]),
    ("metastatic",              6, ["196", "197", "198", "199"]),
    ("aids",                    6, ["042", "043", "044"]),
]


# ---------------------------------------------------------------------------
# Charlson ICD-10 mapping (Quan 2005, Table 2)
# ---------------------------------------------------------------------------

_CHARLSON_ICD10: list[tuple[str, int, list[str]]] = [
    ("myocardial_infarction",   1, ["I21", "I22", "I252"]),
    ("congestive_heart_failure", 1, ["I099", "I110", "I130", "I132",
                                      "I255", "I420", "I425", "I426",
                                      "I427", "I428", "I429", "I43",
                                      "I50", "P290"]),
    ("peripheral_vascular",     1, ["I70", "I71", "I731", "I738",
                                     "I739", "I771", "I790", "I792",
                                     "K551", "K558", "K559", "Z958",
                                     "Z959"]),
    ("cerebrovascular",         1, ["G45", "G46", "H340", "I60",
                                     "I61", "I62", "I63", "I64",
                                     "I65", "I66", "I67", "I68",
                                     "I69"]),
    ("dementia",                1, ["F00", "F01", "F02", "F03",
                                     "F051", "G30", "G311"]),
    ("chronic_pulmonary",       1, ["I278", "I279", "J40", "J41",
                                     "J42", "J43", "J44", "J45",
                                     "J46", "J47", "J60", "J61",
                                     "J62", "J63", "J64", "J65",
                                     "J66", "J67", "J684", "J701",
                                     "J703"]),
    ("rheumatic_disease",       1, ["M05", "M06", "M315", "M32",
                                     "M33", "M34", "M351", "M353",
                                     "M360"]),
    ("peptic_ulcer",            1, ["K25", "K26", "K27", "K28"]),
    ("mild_liver_disease",      1, ["B18", "K700", "K701", "K702",
                                     "K703", "K709", "K713", "K714",
                                     "K715", "K717", "K73", "K74",
                                     "K760", "K762", "K763", "K764",
                                     "K768", "K769", "Z944"]),
    ("diabetes_uncomplicated",  1, ["E100", "E101", "E106", "E108",
                                     "E109", "E110", "E111", "E116",
                                     "E118", "E119", "E120", "E121",
                                     "E126", "E128", "E129", "E130",
                                     "E131", "E136", "E138", "E139",
                                     "E140", "E141", "E146", "E148",
                                     "E149"]),
    ("diabetes_complicated",    2, ["E102", "E103", "E104", "E105",
                                     "E107", "E112", "E113", "E114",
                                     "E115", "E117", "E122", "E123",
                                     "E124", "E125", "E127", "E132",
                                     "E133", "E134", "E135", "E137",
                                     "E142", "E143", "E144", "E145",
                                     "E147"]),
    ("hemiplegia",              2, ["G041", "G114", "G801", "G802",
                                     "G81", "G82", "G830", "G831",
                                     "G832", "G833", "G834", "G839"]),
    ("renal_disease",           2, ["I120", "I131", "N032", "N033",
                                     "N034", "N035", "N036", "N037",
                                     "N052", "N053", "N054", "N055",
                                     "N056", "N057", "N18", "N19",
                                     "N250", "Z490", "Z491", "Z492",
                                     "Z940", "Z992"]),
    ("malignancy",              2, ["C00", "C01", "C02", "C03", "C04",
                                     "C05", "C06", "C07", "C08", "C09",
                                     "C10", "C11", "C12", "C13", "C14",
                                     "C15", "C16", "C17", "C18", "C19",
                                     "C20", "C21", "C22", "C23", "C24",
                                     "C25", "C26", "C30", "C31", "C32",
                                     "C33", "C34", "C37", "C38", "C39",
                                     "C40", "C41", "C43", "C45", "C46",
                                     "C47", "C48", "C49", "C50", "C51",
                                     "C52", "C53", "C54", "C55", "C56",
                                     "C57", "C58", "C60", "C61", "C62",
                                     "C63", "C64", "C65", "C66", "C67",
                                     "C68", "C69", "C70", "C71", "C72",
                                     "C73", "C74", "C75", "C76", "C81",
                                     "C82", "C83", "C84", "C85", "C88",
                                     "C90", "C91", "C92", "C93", "C94",
                                     "C95", "C96", "C97"]),
    ("mod_severe_liver",        3, ["I850", "I859", "I864", "I982",
                                     "K704", "K711", "K721", "K729",
                                     "K765", "K766", "K767"]),
    ("metastatic",              6, ["C77", "C78", "C79", "C80"]),
    ("aids",                    6, ["B20", "B21", "B22", "B24"]),
]


# ---------------------------------------------------------------------------
# Haematologic malignancy (for SAPS-II — scored 10 pts, distinct from the
# 9-pt metastatic solid tumour). The Charlson "malignancy" category lumps
# solid + haematologic together, so we detect haematologic separately here.
#   ICD-9:  200–208 (lymphoma, leukaemia, multiple myeloma)
#   ICD-10: C81–C96 (Hodgkin/non-Hodgkin lymphoma, leukaemia, myeloma)
# ---------------------------------------------------------------------------
_HAEM_ICD9 = [str(n) for n in range(200, 209)]            # 200..208
_HAEM_ICD10 = [f"C{n}" for n in range(81, 97)]            # C81..C96


def _detect_haematologic(norm_codes: list[str]) -> int:
    return int(
        _startswith_any_code(norm_codes, _HAEM_ICD9)
        or _startswith_any_code(norm_codes, _HAEM_ICD10)
    )


def _startswith_any_code(norm_codes: list[str], prefixes: Sequence[str]) -> bool:
    return any(_startswith_any(c, prefixes) for c in norm_codes)


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def _score_codes(
    raw_codes: Sequence[str],
    mapping: list[tuple[str, int, list[str]]],
) -> dict[str, int]:
    """
    Map a list of ICD codes to Charlson condition flags.

    Returns a dict {condition_name → present (0/1)} plus 'charlson_total'
    and 'haematologic_malignancy' (for the SAPS-II 10-pt component).
    Codes are normalised (no dots, uppercase) before matching.
    """
    norm_codes = [_norm(c) for c in raw_codes if c and str(c).strip()]
    result: dict[str, int] = {}
    total = 0
    for name, weight, prefixes in mapping:
        present = any(
            _startswith_any(c, prefixes) for c in norm_codes
        )
        result[name] = int(present)
        if present:
            total += weight
    result["charlson_total"] = total
    result["haematologic_malignancy"] = _detect_haematologic(norm_codes)
    return result


def charlson_icd9(codes: Sequence[str]) -> dict[str, int]:
    """Charlson comorbidity index from ICD-9 codes (Quan 2005)."""
    return _score_codes(codes, _CHARLSON_ICD9)


def charlson_icd10(codes: Sequence[str]) -> dict[str, int]:
    """Charlson comorbidity index from ICD-10 codes (Quan 2005)."""
    return _score_codes(codes, _CHARLSON_ICD10)


def charlson_mixed(codes: Sequence[str], versions: Sequence[int]) -> dict[str, int]:
    """
    Charlson index from mixed ICD-9/10 codes (MIMIC-IV pattern).
    `codes` and `versions` must be the same length.
    """
    icd9 = [c for c, v in zip(codes, versions) if v == 9]
    icd10 = [c for c, v in zip(codes, versions) if v == 10]
    r9  = _score_codes(icd9,  _CHARLSON_ICD9)
    r10 = _score_codes(icd10, _CHARLSON_ICD10)
    merged: dict[str, int] = {}
    for name, _, _ in _CHARLSON_ICD9:
        merged[name] = max(r9.get(name, 0), r10.get(name, 0))
    merged["charlson_total"] = sum(
        w for nm, w, _ in _CHARLSON_ICD9 if merged.get(nm, 0)
    )
    merged["haematologic_malignancy"] = max(
        r9.get("haematologic_malignancy", 0),
        r10.get("haematologic_malignancy", 0),
    )
    return merged


# ---------------------------------------------------------------------------
# SAPS-II chronic_disease component
# ---------------------------------------------------------------------------

def chronic_disease_sapsii(charlson: dict[str, int]) -> int:
    """
    SAPS-II chronic_disease points from a Charlson result dict.
    Le Gall 1993 (only these three conditions score; the highest applies):
      - AIDS (not just HIV+):     17 pts
      - Haematologic malignancy:  10 pts  (ICD-9 200-208 / ICD-10 C81-C96)
      - Metastatic cancer:         9 pts  (Charlson 'metastatic' flag)

    Note: a non-metastatic SOLID tumour scores 0 in SAPS-II (it is NOT a
    chronic-disease point), so we deliberately do not score the generic
    Charlson 'malignancy' flag — only metastatic or haematologic.
    """
    if charlson.get("aids", 0):
        return 17
    if charlson.get("haematologic_malignancy", 0):
        return 10
    if charlson.get("metastatic", 0):
        return 9
    return 0
