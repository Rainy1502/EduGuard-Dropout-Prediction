"""Shared utilities for the notebook and the Streamlit app.

Contains:
- Mappings from numeric codes to labels (based on the UCI dataset documentation
  "Predict Students' Dropout and Academic Success").
- The list of features used by the model.
- The feature-engineering function, which also runs inside the model pipeline,
  so the app only needs to send raw features.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------
MARITAL_STATUS = {
    1: "Single", 2: "Married", 3: "Widower", 4: "Divorced",
    5: "Facto union", 6: "Legally separated",
}

APPLICATION_MODE = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)",
}

COURSE = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)",
}

PREVIOUS_QUALIFICATION = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year)",
    38: "Basic education 2nd cycle (6th/7th/8th year)",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
}

NATIONALITY = {
    1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch",
    14: "English", 17: "Lithuanian", 21: "Angolan", 22: "Cape Verdean",
    24: "Guinean", 25: "Mozambican", 26: "Santomean", 32: "Turkish",
    41: "Brazilian", 62: "Romanian", 100: "Moldova (Republic of)",
    101: "Mexican", 103: "Ukrainian", 105: "Russian", 108: "Cuban",
    109: "Colombian",
}

YES_NO = {1: "Yes", 0: "No"}
GENDER = {1: "Male", 0: "Female"}
ATTENDANCE = {1: "Daytime", 0: "Evening"}

# Parents' education has many codes (>30), so they are grouped into
# broader levels to keep the analysis readable.
_PARENT_EDU_GROUPS = {
    "Higher education": [2, 3, 4, 5, 6, 40, 41, 43, 44],
    "Secondary education": [1, 9, 10, 12, 13, 14, 18, 20, 22, 25, 27, 31, 33, 39, 42],
    "Basic education": [11, 19, 26, 29, 30, 37, 38],
    "No formal schooling": [35, 36],
    "Unknown": [34],
}
PARENT_EDUCATION = {code: grp for grp, codes in _PARENT_EDU_GROUPS.items() for code in codes}

# Parents' occupation codes follow the Portuguese occupation classification
# (derived from ISCO). Codes 0-10 are major groups; 3-digit codes are sub-groups
# whose first digit after the leading 1 identifies the major group.
OCCUPATION_GROUPS = {
    0: "Student",
    1: "Directors & executive managers",
    2: "Intellectual & scientific specialists",
    3: "Intermediate-level technicians",
    4: "Administrative staff",
    5: "Personal services, security & sellers",
    6: "Farmers & skilled agricultural workers",
    7: "Skilled industry & construction workers",
    8: "Machine operators & assembly workers",
    9: "Unskilled workers",
    10: "Armed forces",
    90: "Other situation",
    99: "Unknown",
}


def occupation_group(code: int) -> str:
    """Map an occupation code (including 3-digit sub-groups) to its major group."""
    code = int(code)
    if code in OCCUPATION_GROUPS:
        return OCCUPATION_GROUPS[code]
    if 101 <= code <= 103:
        return OCCUPATION_GROUPS[10]
    if 110 <= code <= 199:
        return OCCUPATION_GROUPS.get((code - 100) // 10, "Unknown")
    return "Unknown"


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add label columns (suffix `_label`) for the categorical columns."""
    out = df.copy()
    out["Marital_status_label"] = out["Marital_status"].map(MARITAL_STATUS)
    out["Application_mode_label"] = out["Application_mode"].map(APPLICATION_MODE)
    out["Course_label"] = out["Course"].map(COURSE)
    out["Attendance_label"] = out["Daytime_evening_attendance"].map(ATTENDANCE)
    out["Previous_qualification_label"] = out["Previous_qualification"].map(PREVIOUS_QUALIFICATION)
    out["Nationality_label"] = out["Nacionality"].map(NATIONALITY)
    out["Mothers_education_label"] = out["Mothers_qualification"].map(PARENT_EDUCATION).fillna("Unknown")
    out["Fathers_education_label"] = out["Fathers_qualification"].map(PARENT_EDUCATION).fillna("Unknown")
    out["Mothers_occupation_label"] = out["Mothers_occupation"].map(occupation_group)
    out["Fathers_occupation_label"] = out["Fathers_occupation"].map(occupation_group)
    out["Gender_label"] = out["Gender"].map(GENDER)
    for col in ["Displaced", "Educational_special_needs", "Debtor",
                "Tuition_fees_up_to_date", "Scholarship_holder", "International"]:
        out[f"{col}_label"] = out[col].map(YES_NO)
    return out


# ---------------------------------------------------------------------------
# Model features
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["Marital_status", "Application_mode", "Course"]

NUMERIC_FEATURES = [
    "Daytime_evening_attendance",
    "Admission_grade",
    "Previous_qualification_grade",
    "Displaced",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Gender",
    "Scholarship_holder",
    "Age_at_enrollment",
    "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

ENGINEERED_FEATURES = ["Approval_rate_1st_sem", "Approval_rate_2nd_sem"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the ratio of courses passed / courses enrolled for each semester.

    If a student enrolled in no courses (enrolled = 0), the ratio is 0.
    Used inside the model pipeline (via FunctionTransformer).
    """
    out = df.copy()
    for sem in ["1st", "2nd"]:
        enrolled = out[f"Curricular_units_{sem}_sem_enrolled"]
        approved = out[f"Curricular_units_{sem}_sem_approved"]
        out[f"Approval_rate_{sem}_sem"] = np.where(
            enrolled > 0, approved / enrolled.where(enrolled > 0, 1), 0.0
        ).clip(0, 1)
    return out
