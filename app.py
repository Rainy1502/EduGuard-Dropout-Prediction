"""EduGuard - early warning prototype for students at risk of dropping out (Jaya Jaya Institut).

Run locally:
    streamlit run app.py
"""

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (APPLICATION_MODE, ATTENDANCE, CATEGORICAL_FEATURES, COURSE, GENDER,
                   MARITAL_STATUS, MODEL_FEATURES)

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="EduGuard | Jaya Jaya Institut", page_icon=str(BASE_DIR / "assets" / "favicon.svg"),
                   layout="wide", initial_sidebar_state="collapsed")

MODEL_PATH = BASE_DIR / "model" / "dropout_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"
SAMPLE_PATH = BASE_DIR / "data" / "sample_batch_input.csv"
DATA_PATH = BASE_DIR / "data" / "data.csv"
CLEAN_PATH = BASE_DIR / "data" / "students_clean.csv"
CSS_PATH = BASE_DIR / "assets" / "style.css"

# Palette: blue = UI accent; orange/green/blue = data statuses. No gradients.
C_DROP, C_ENR, C_GRAD = "#d95926", "#199e70", "#3987e5"
C_VIOLET, C_YELLOW = "#9085e9", "#c98500"  # colours for non-status cards
AVG_LINE = "#e6e9ef"  # average line on dropout-rate charts
C_NEUTRAL, INK, INK_SOFT, MUTED = "#3b4559", "#e6e9ef", "#cfd5e1", "#aeb8cc"
GRID = "rgba(255,255,255,0.06)"
FONT = "Plus Jakarta Sans, sans-serif"
STATUS_COLORS = {"Dropout": C_DROP, "Enrolled": C_ENR, "Graduate": C_GRAD}

# Risk levels (always shown together with a text label)
RISK_STYLE = {
    "Low": {"color": C_ENR, "action": "Routine monitoring"},
    "Medium": {"color": "#e0a526", "action": "Monitor closely"},
    "High": {"color": C_DROP, "action": "Priority for targeted guidance"},
}

LOGO_SVG = """<svg width="{size}" height="{size}" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-label="EduGuard">
<path d="M24 3 L41 9.5 V23 C41 34 33.6 41.6 24 45 C14.4 41.6 7 34 7 23 V9.5 Z" fill="#3987e5"/>
<path d="M24 15.5 L36.5 21.2 L24 26.9 L11.5 21.2 Z" fill="#0b1120"/>
<path d="M16.8 24 V29.6 C16.8 29.6 19.6 32.8 24 32.8 C28.4 32.8 31.2 29.6 31.2 29.6 V24 L24 27.3 Z" fill="#0b1120"/>
<path d="M34.6 22.3 V28.4" stroke="#0b1120" stroke-width="1.6" stroke-linecap="round"/>
</svg>"""

FEATURE_NAMES = {
    "Marital_status": "Marital status",
    "Application_mode": "Admission path",
    "Course": "Study program",
    "Daytime_evening_attendance": "Class time",
    "Admission_grade": "Admission grade",
    "Previous_qualification_grade": "Previous qualification grade",
    "Displaced": "Displaced (away from home)",
    "Debtor": "Debtor",
    "Tuition_fees_up_to_date": "Tuition paid up",
    "Gender": "Gender",
    "Scholarship_holder": "Scholarship holder",
    "Age_at_enrollment": "Age at enrollment",
    "Curricular_units_1st_sem_enrolled": "Courses enrolled (sem 1)",
    "Curricular_units_1st_sem_evaluations": "Courses evaluated (sem 1)",
    "Curricular_units_1st_sem_approved": "Courses passed (sem 1)",
    "Curricular_units_1st_sem_grade": "Average grade (sem 1)",
    "Curricular_units_2nd_sem_enrolled": "Courses enrolled (sem 2)",
    "Curricular_units_2nd_sem_evaluations": "Courses evaluated (sem 2)",
    "Curricular_units_2nd_sem_approved": "Courses passed (sem 2)",
    "Curricular_units_2nd_sem_grade": "Average grade (sem 2)",
    "Approval_rate_1st_sem": "Course pass rate (sem 1)",
    "Approval_rate_2nd_sem": "Course pass rate (sem 2)",
}

# Example profiles for a quick demo
DEFAULT_PROFILE = {
    "Marital_status": 1, "Application_mode": 1, "Course": 9500, "Daytime_evening_attendance": 1,
    "Admission_grade": 127.0, "Previous_qualification_grade": 133.0, "Displaced": 1, "Debtor": 0,
    "Tuition_fees_up_to_date": 1, "Gender": 0, "Scholarship_holder": 0, "Age_at_enrollment": 19,
    "Curricular_units_1st_sem_enrolled": 6, "Curricular_units_1st_sem_evaluations": 8,
    "Curricular_units_1st_sem_approved": 5, "Curricular_units_1st_sem_grade": 12.3,
    "Curricular_units_2nd_sem_enrolled": 6, "Curricular_units_2nd_sem_evaluations": 8,
    "Curricular_units_2nd_sem_approved": 5, "Curricular_units_2nd_sem_grade": 12.2,
}
HIGH_RISK_PROFILE = {
    **DEFAULT_PROFILE,
    "Marital_status": 2, "Application_mode": 39, "Course": 9991, "Daytime_evening_attendance": 0,
    "Admission_grade": 118.0, "Previous_qualification_grade": 120.0, "Displaced": 0, "Debtor": 1,
    "Tuition_fees_up_to_date": 0, "Gender": 1, "Age_at_enrollment": 31,
    "Curricular_units_1st_sem_approved": 2, "Curricular_units_1st_sem_grade": 10.5,
    "Curricular_units_2nd_sem_evaluations": 6, "Curricular_units_2nd_sem_approved": 1,
    "Curricular_units_2nd_sem_grade": 10.0,
}


# ===========================================================================
# Loader
# ===========================================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_students():
    return pd.read_csv(CLEAN_PATH)


model = load_model()
meta = load_metadata()
THRESHOLD = meta["threshold"]
RISK_LOW_MAX = meta["risk_low_max"]


# ===========================================================================
# Number formatting
# ===========================================================================
def fmt_int(n) -> str:
    return f"{int(n):,}"


def fmt_pct(x: float, decimals: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{decimals}f}%"


# ===========================================================================
# UI components (HTML)
# ===========================================================================
def html(markup: str):
    """Render HTML; lines are left-stripped so Markdown does not treat them as a code block."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def dot(color: str) -> str:
    return f'<span class="eg-dot" style="background:{color};"></span>'


def section(kicker: str, title: str, description: str | None = None, first: bool = False):
    desc = f"<p>{description}</p>" if description else ""
    html(f"""<div class="eg-section{' first' if first else ''}">
        <div class="eg-kicker">{kicker}</div><h2>{title}</h2>{desc}
    </div>""")


def stat_cards(cards: list[tuple]):
    """cards: (label, value, sub, color | None, meter_share | None). color=None = neutral card."""
    for col, (label, value, sub, color, share) in zip(st.columns(len(cards)), cards):
        border, value_color, meter_color = (color, color, color) if color else (C_NEUTRAL, INK, MUTED)
        meter = (f'<div class="eg-meter"><span style="width:{min(share, 1) * 100:.1f}%;'
                 f'background:{meter_color};"></span></div>' if share is not None else "")
        with col:
            html(f"""<div class="eg-stat" style="--accent:{border};">
                <div class="eg-stat-label">{label}</div>
                <div class="eg-stat-value" style="color:{value_color};">{value}</div>{meter}
                <div class="eg-stat-sub">{sub}</div>
            </div>""")


def chart_title(title: str, subtitle: str | None = None, rate_key: bool = False):
    """Chart title. rate_key=True also shows the legend explaining dropout-rate chart colours."""
    sub = f'<p class="eg-chart-sub">{subtitle}</p>' if subtitle else ""
    key = ""
    if rate_key:
        key = (f'<div class="eg-key"><span><i style="background:{C_DROP};"></i>above segment average</span>'
               f'<span><i style="background:{C_NEUTRAL};"></i>below average</span>'
               f'<span><i class="line"></i>segment average</span></div>')
    html(f'<p class="eg-chart-title">{title}</p>{sub}{key}')


def insight(stat: str, caption: str, actions: list[str], color: str = C_DROP):
    """Insight card: key number, one sentence (including a comparison), then actions."""
    items = "".join(f"<li>{a}</li>" for a in actions)
    html(f"""<div class="eg-insight" style="--accent:{color};">
        <div class="eg-insight-stat">{stat}</div>
        <div class="eg-insight-caption">{caption}</div>
        <div class="eg-actions"><div class="eg-actions-title">Actions</div><ul>{items}</ul></div>
    </div>""")


def spacer(height: str = "1rem"):
    html(f"<div style='height:{height}'></div>")


def navbar():
    m = meta["test_metrics"]
    html(f"""<div class="eg-nav">
        <div class="eg-brand">{LOGO_SVG.format(size=42)}
            <div><div class="eg-wordmark">Edu<span>Guard</span></div>
            <div class="eg-tagline">Jaya Jaya Institut · Student Dropout Early Warning</div></div>
        </div>
        <div class="eg-chip">{dot(C_ENR)}Active model: {meta['model_name']} · Recall {fmt_pct(m['recall'], 0)}</div>
    </div>""")


def hero(students: pd.DataFrame):
    n = len(students)
    counts = students["Status"].value_counts()
    strip = "".join(f'<div style="width:{counts.get(s, 0) / n * 100:.2f}%;background:{c};"></div>'
                    for s, c in STATUS_COLORS.items())
    legend = "".join(f"""<div class="eg-legend-item">
            <div class="lbl">{dot(c)}{s}</div>
            <div class="val" style="color:{c};">{fmt_pct(counts.get(s, 0) / n)}</div>
            <div class="sub">{fmt_int(counts.get(s, 0))} students</div>
        </div>""" for s, c in STATUS_COLORS.items())
    html(f"""<div class="eg-hero">
        <div>
            <div class="eg-kicker">Dropout early warning system</div>
            <h1>Find at-risk students <span class="eg-hl">before it's too&nbsp;late.</span></h1>
            <p>Track what drives dropout and spot students who need guidance early.</p>
        </div>
        <div class="eg-cohort">
            <div class="eg-cohort-title"><span>Final status of all students</span><strong>{fmt_int(n)}</strong></div>
            <div class="eg-strip">{strip}</div>
            <div class="eg-legend">{legend}</div>
            <div class="eg-cohort-note"><strong>Enrolled</strong> = graduating late ·
            <strong>Graduate</strong> = graduated</div>
        </div>
    </div>""")


# ===========================================================================
# Charts (Plotly)
# ===========================================================================
PLOT_CONFIG = {"displayModeBar": False}


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=24, r=28, t=28, b=8), separators=".,",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=13, family=FONT),
        hoverlabel=dict(font_size=13, font_family=FONT, bgcolor="#141d30", bordercolor="#2a3550", font_color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_SOFT)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    return fig


def show(fig: go.Figure):
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)


def rate_table(df: pd.DataFrame, col: str, min_n: int = 0, order: list | None = None) -> pd.DataFrame:
    t = df.groupby(col)["Is_dropout"].agg(rate="mean", n="size").reset_index()
    t = t[t["n"] >= min_n]
    if order is not None:
        t = t.set_index(col).reindex([o for o in order if o in set(t[col])]).reset_index()
    return t


def conditions_table(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    return pd.DataFrame([{"condition": k, "rate": df.loc[m, "Is_dropout"].mean(), "n": int(m.sum())}
                         for k, m in conditions.items() if m.sum() > 0])


SMALL_N = 30  # groups with fewer than 30 students are considered unrepresentative


def rate_chart(t: pd.DataFrame, col: str, overall: float, horizontal: bool = True,
               height: int = 330, sort: bool = True) -> go.Figure:
    """Dropout-rate bar chart. Orange = above the segment average, grey = below average.

    Groups with n < SMALL_N are drawn semi-transparent and labelled with their student count.
    """
    if sort:
        t = t.sort_values("rate", ascending=horizontal)
    colors = [C_DROP if r > overall else C_NEUTRAL for r in t["rate"]]
    opacity = [0.4 if n < SMALL_N else 1.0 for n in t["n"]]
    text = [fmt_pct(r, 0) + (f"  (n={n}, few)" if n < SMALL_N else "") for r, n in zip(t["rate"], t["n"])]
    custom = np.stack([t[col].astype(str), [fmt_pct(r) for r in t["rate"]], [fmt_int(v) for v in t["n"]]], axis=1)
    hover = "<b>%{customdata[0]}</b><br>Dropout rate: %{customdata[1]}<br>Students: %{customdata[2]}<extra></extra>"
    common = dict(marker_color=colors, marker_opacity=opacity, text=text, textposition="outside",
                  customdata=custom, hovertemplate=hover, cliponaxis=False,
                  textfont=dict(color=INK_SOFT, size=13))
    if horizontal:
        # Leading spaces add slack: Plotly measures the margin before the web font has loaded.
        labels = ["   " + str(v) for v in t[col]]
        fig = go.Figure(go.Bar(x=t["rate"], y=labels, orientation="h", **common))
    else:
        fig = go.Figure(go.Bar(x=t[col], y=t["rate"], **common))
    # Average line: bright and drawn above the bars; its label sits outside the plot so it never covers a bar.
    line = dict(line_dash="dash", line_color=AVG_LINE, line_width=1.6, layer="above")
    label = dict(text=f"<b>average {fmt_pct(overall, 0)}</b>", showarrow=False, font=dict(color=INK, size=12),
                 bgcolor="#1b2539", bordercolor=AVG_LINE, borderwidth=1, borderpad=4)
    fig = style_fig(fig, height)
    if horizontal:
        fig.add_vline(x=overall, **line)
        fig.add_annotation(x=overall, xref="x", y=1, yref="paper", yanchor="bottom", yshift=6, **label)
        fig.update_xaxes(tickformat=".0%", range=[0, 1.1])
        fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SOFT))
        fig.update_layout(margin=dict(t=44))
    else:
        fig.add_hline(y=overall, **line)
        fig.add_annotation(x=1, xref="paper", xanchor="left", xshift=8, y=overall, yref="y", **label)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
        fig.update_xaxes(type="category", showgrid=False, tickfont=dict(color=INK_SOFT))
        fig.update_layout(margin=dict(r=130))
    fig.update_traces(marker_line_width=0, width=0.6, selector=dict(type="bar"))
    return fig


def status_bar_chart(df: pd.DataFrame, value_cols: dict, y_title: str, y_max: float,
                     height: int = 320) -> go.Figure:
    """Grouped bar chart of metric averages per status. All metrics in one chart must share a unit."""
    grouped = df.groupby("Status")[list(value_cols)].mean()
    fig = go.Figure()
    for status in STATUS_COLORS:
        if status not in grouped.index:
            continue
        vals = grouped.loc[status]
        fig.add_bar(name=status, x=list(value_cols.values()), y=vals.values, marker_color=STATUS_COLORS[status],
                    text=[f"{v:.1f}" for v in vals.values], textposition="outside",
                    cliponaxis=False, textfont=dict(color=INK_SOFT),
                    hovertemplate=f"<b>{status}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    fig.update_xaxes(showgrid=False, tickfont=dict(color=INK_SOFT))
    fig.update_yaxes(title_text=y_title, title_font_size=12, range=[0, y_max])
    return style_fig(fig, height)


# ===========================================================================
# Model helper
# ===========================================================================
def risk_level(prob: float) -> str:
    if prob >= THRESHOLD:
        return "High"
    if prob >= RISK_LOW_MAX:
        return "Medium"
    return "Low"


def predict(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[MODEL_FEATURES])[:, 1]


@st.cache_resource
def reference_sample(n: int = 200) -> pd.DataFrame:
    """Sample of students from the modeling data (Dropout + Graduate), used as the comparison group."""
    data = pd.read_csv(DATA_PATH, sep=";")
    data = data[data["Status"].isin(["Dropout", "Graduate"])]
    return data[MODEL_FEATURES].sample(n, random_state=42).reset_index(drop=True)


def format_value(feature: str, value) -> str:
    """Format a feature value so it reads well on the contribution chart."""
    lookups = {"Course": COURSE, "Application_mode": APPLICATION_MODE, "Marital_status": MARITAL_STATUS,
               "Daytime_evening_attendance": ATTENDANCE, "Gender": GENDER}
    if feature in lookups:
        return lookups[feature].get(int(value), str(value))
    if feature in {"Displaced", "Debtor", "Tuition_fees_up_to_date", "Scholarship_holder"}:
        return "Yes" if int(value) == 1 else "No"
    if feature.startswith("Approval_rate"):
        return fmt_pct(value, 0)
    if feature.endswith("_grade"):
        return f"{value:.1f}"
    return f"{int(value)}"


def explain(row: pd.DataFrame) -> pd.Series:
    """Contribution of each feature through a what-if analysis (works for any model type).

    contribution = this student's probability - mean probability when one feature is swapped for
    other students' values (a sample of the modeling data) while every other feature stays the same.
    Positive values raise the risk, negative values lower it.
    """
    values = row[MODEL_FEATURES].iloc[0].to_dict()
    ref = reference_sample()
    variants = []
    for f in MODEL_FEATURES:
        v = pd.DataFrame([values] * len(ref))
        v[f] = ref[f].values
        variants.append(v)
    probs = predict(pd.concat(variants, ignore_index=True)).reshape(len(MODEL_FEATURES), len(ref))
    contrib = predict(row)[0] - probs.mean(axis=1)
    labels = [f"{FEATURE_NAMES[f]}: {format_value(f, values[f])}" for f in MODEL_FEATURES]
    return pd.Series(contrib, index=labels)


def recommendations(row: dict, prob: float) -> list[str]:
    recs = []
    rate1 = row["Curricular_units_1st_sem_approved"] / max(row["Curricular_units_1st_sem_enrolled"], 1)
    rate2 = row["Curricular_units_2nd_sem_approved"] / max(row["Curricular_units_2nd_sem_enrolled"], 1)
    if row["Curricular_units_2nd_sem_enrolled"] == 0:
        recs.append("<strong>No courses taken in semester 2.</strong> Contact the student now, a strong sign of quitting.")
    if min(rate1, rate2) < 0.5:
        recs.append("<strong>Low course pass rate.</strong> Intensive advisor mentoring and peer tutoring.")
    if row["Tuition_fees_up_to_date"] == 0:
        recs.append("<strong>Overdue tuition.</strong> Offer instalments or relief through the finance office.")
    if row["Debtor"] == 1:
        recs.append("<strong>In debt.</strong> Financial counselling and a rescheduled payment plan.")
    if row["Scholarship_holder"] == 0 and prob >= RISK_LOW_MAX:
        recs.append("<strong>No scholarship yet.</strong> Share available scholarship opportunities.")
    if row["Age_at_enrollment"] >= 25 or row["Daytime_evening_attendance"] == 0:
        recs.append("<strong>Mature / evening student.</strong> Flexible scheduling and time-management counselling.")
    if not recs:
        recs.append("No prominent risk factors. Continue routine monitoring.")
    return recs


def gauge(prob: float, level: str) -> go.Figure:
    color = RISK_STYLE[level]["color"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 42, "color": INK, "family": FONT}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": MUTED, "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "threshold": {"line": {"color": INK, "width": 2}, "value": THRESHOLD * 100},
        },
    ))
    return style_fig(fig, 230).update_layout(margin=dict(l=28, r=28, t=16, b=0))


def contribution_chart(contrib: pd.Series, n: int = 8) -> go.Figure:
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(n).iloc[::-1]
    colors = [C_DROP if v > 0 else C_GRAD for v in top.values]
    direction = ["raises risk" if v > 0 else "lowers risk" for v in top.values]
    fig = go.Figure(go.Bar(x=top.values, y=["\u00a0\u00a0\u00a0" + s for s in top.index], orientation="h",
                           marker_color=colors, width=0.6, customdata=np.stack([top.index, direction], axis=1),
                           hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    # The raw numbers mean nothing to users; only the direction and relative bar length matter.
    fig.update_xaxes(showticklabels=False, showgrid=False,
                     title_text="lowers risk  \u2190   \u2192  raises risk", title_font_size=12)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SOFT))
    return style_fig(fig, 350)


def read_uploaded_csv(uploaded) -> pd.DataFrame:
    """Read an uploaded CSV; the delimiter (comma/semicolon) is detected automatically."""
    raw = uploaded.getvalue().decode("utf-8-sig")
    header = raw.splitlines()[0] if raw else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(io.StringIO(raw), sep=sep)


def load_profile(profile: dict):
    for key, value in profile.items():
        st.session_state[key] = value


# ===========================================================================
# Main layout
# ===========================================================================
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

students = load_students()
navbar()
hero(students)

if "Course" not in st.session_state:
    load_profile(DEFAULT_PROFILE)

tab_dash, tab_single, tab_batch, tab_about = st.tabs(
    ["Dashboard", "Single Prediction", "Batch Prediction", "About the Model"])

# ===========================================================================
# TAB 1 - DASHBOARD
# ===========================================================================
with tab_dash:
    section("Monitoring", "Student dropout dashboard", first=True)

    f1, f2, f3 = st.columns([2.2, 1, 1])
    course_filter = f1.multiselect("Study program", sorted(students["Course_label"].unique()),
                                   placeholder="All study programs")
    gender_filter = f2.segmented_control("Gender", ["Female", "Male"], selection_mode="multi",
                                         default=["Female", "Male"])
    time_filter = f3.segmented_control("Class time", ["Daytime", "Evening"], selection_mode="multi",
                                       default=["Daytime", "Evening"])

    df = students
    if course_filter:
        df = df[df["Course_label"].isin(course_filter)]
    df = df[df["Gender_label"].isin(gender_filter or []) & df["Attendance_label"].isin(time_filter or [])]

    if df.empty:
        st.warning("No students match these filters. Adjust the filters above.")
    else:
        n = len(df)
        overall = df["Is_dropout"].mean()
        counts = df["Status"].value_counts()

        late_fee = int((df["Tuition_fees_up_to_date"] == 0).sum())

        spacer("0.6rem")
        stat_cards([
            ("Students", fmt_int(n), f"of {fmt_int(len(students))} students", C_VIOLET, n / len(students)),
            ("Dropout", fmt_pct(overall), f"{fmt_int(counts.get('Dropout', 0))} students", C_DROP, overall),
            ("Enrolled", fmt_pct(counts.get("Enrolled", 0) / n),
             f"{fmt_int(counts.get('Enrolled', 0))} students", C_ENR, counts.get("Enrolled", 0) / n),
            ("Graduate", fmt_pct(counts.get("Graduate", 0) / n),
             f"{fmt_int(counts.get('Graduate', 0))} students", C_GRAD, counts.get("Graduate", 0) / n),
            ("Overdue tuition", fmt_int(late_fee), f"{fmt_pct(late_fee / n)} of segment", C_YELLOW, late_fee / n),
        ])

        section("Overview", "Conditions that most often lead to dropout")
        chart_title("Dropout rate by student condition", rate_key=True)
        profile = conditions_table(df, {
            "Overdue tuition": df["Tuition_fees_up_to_date"] == 0,
            "0 courses passed in semester 2": df["Curricular_units_2nd_sem_approved"] == 0,
            "Debtor": df["Debtor"] == 1,
            "Enrolled at age ≥ 25": df["Age_at_enrollment"] >= 25,
            "Over 23 years old path": df["Application_mode"] == 39,
            "Male": df["Gender"] == 1,
            "Evening classes": df["Daytime_evening_attendance"] == 0,
            "Scholarship holder (comparison)": df["Scholarship_holder"] == 1,
        })
        show(rate_chart(profile, "condition", overall, height=340))

        # ── 01 Academic ─────────────────────────────────────────────
        section("Business question 01 · Academic", "Are early-semester results an early warning sign?")
        appr = rate_table(df, "Approved_2nd_group", order=["0", "1-2", "3-4", "5-6", ">6"])
        ar = appr.set_index("Approved_2nd_group")["rate"]
        a1, a2 = st.columns([1.7, 1], gap="large")
        with a1:
            chart_title("Dropout rate by courses passed in semester 2")
            show(rate_chart(appr, "Approved_2nd_group", overall, horizontal=False, sort=False))
        with a2:
            insight(fmt_pct(ar.get("0", np.nan), 0),
                    "of students who passed no courses in semester 2 dropped out "
                    f"(vs {fmt_pct(ar.get('5-6', np.nan), 0)} of those who passed 5-6)",
                    ["Track courses passed every semester", "A pass rate below 50% requires mentoring"])
        spacer("1.4rem")
        b1, b2 = st.columns(2, gap="large")
        with b1:
            chart_title("Average courses passed per status")
            show(status_bar_chart(df, {
                "Curricular_units_1st_sem_approved": "Semester 1",
                "Curricular_units_2nd_sem_approved": "Semester 2",
            }, y_title="Courses passed", y_max=8))
        with b2:
            chart_title("Average semester grade per status", "Grade scale 0-20")
            show(status_bar_chart(df, {
                "Curricular_units_1st_sem_grade": "Semester 1",
                "Curricular_units_2nd_sem_grade": "Semester 2",
            }, y_title="Average grade", y_max=20))

        # ── 02 Financial ───────────────────────────────────────────
        section("Business question 02 · Financial", "How much does financial condition matter?")
        fin = conditions_table(df, {
            "Overdue tuition": df["Tuition_fees_up_to_date"] == 0,
            "Tuition paid up": df["Tuition_fees_up_to_date"] == 1,
            "Debtor": df["Debtor"] == 1,
            "No debt": df["Debtor"] == 0,
            "No scholarship": df["Scholarship_holder"] == 0,
            "Scholarship holder": df["Scholarship_holder"] == 1,
        })
        fr = fin.set_index("condition")["rate"]
        d1, d2 = st.columns([1, 1.7], gap="large")
        with d1:
            insight(fmt_pct(fr.get("Overdue tuition", np.nan), 0),
                    "of students with overdue tuition dropped out "
                    f"(vs {fmt_pct(fr.get('Scholarship holder', np.nan), 0)} of scholarship holders)",
                    ["Tuition instalments or relief", "Expand scholarships"])
        with d2:
            chart_title("Dropout rate by financial condition")
            show(rate_chart(fin, "condition", overall, sort=False).update_yaxes(autorange="reversed"))

        # ── 03 Demographic──────────────────────────────────────────
        section("Business question 03 · Demographic", "Who is most vulnerable to dropping out?")
        age = rate_table(df, "Age_group", order=["<=20", "21-24", "25-29", "30-39", "40+"])
        agr = age.set_index("Age_group")["rate"]
        e1, e2 = st.columns([1.7, 1], gap="large")
        with e1:
            chart_title("Dropout rate by age at enrollment")
            show(rate_chart(age, "Age_group", overall, horizontal=False, sort=False))
        with e2:
            insight(fmt_pct(agr.get("25-29", np.nan), 0),
                    "of students who enrolled at age 25-29 dropped out "
                    f"(vs {fmt_pct(agr.get('<=20', np.nan), 0)} of those aged ≤20)",
                    ["Flexible or online classes", "Time-management counselling"])

        # ── 04 Study program──────────────────────────────────────
        section("Business question 04 · Program & admission path", "Which programs and admission paths need attention?")
        course = rate_table(df, "Course_label")
        big = course[course["n"] >= 30].sort_values("rate", ascending=False)
        g1, g2 = st.columns([1, 1.7], gap="large")
        with g1:
            if big.empty:
                insight("-", "no study program has at least 30 students in this segment",
                        ["Widen the filter selection"])
            else:
                top = big.iloc[0]
                insight(fmt_pct(top["rate"], 0),
                        f"of {top['Course_label']} students dropped out, the highest of any program (min. 30 students)",
                        ["Dedicated mentoring per program", "Review the first-year curriculum"])
        with g2:
            chart_title("Dropout rate by study program", f"Faded = fewer than {SMALL_N} students")
            show(rate_chart(course, "Course_label", overall, height=max(340, 26 * len(course))))
        spacer("1.4rem")
        chart_title(f"Dropout rate by admission path (min. {SMALL_N} students)")
        mode = rate_table(df, "Application_mode_label", min_n=30)
        if mode.empty:
            st.caption("No admission path has at least 30 students in this segment.")
        else:
            show(rate_chart(mode, "Application_mode_label", overall, height=max(260, 32 * len(mode))))

# ===========================================================================
# TAB 2 - INDIVIDUAL PREDICTION
# ===========================================================================
with tab_single:
    section("Single prediction", "How likely is this student to drop out?", first=True)
    legend = "".join(
        f'<span class="eg-chip">{dot(RISK_STYLE[lvl]["color"])}{lvl} {rng}</span>'
        for lvl, rng in [("Low", f"&lt; {fmt_pct(RISK_LOW_MAX, 0)}"),
                         ("Medium", f"{fmt_pct(RISK_LOW_MAX, 0)} - {fmt_pct(THRESHOLD, 0)}"),
                         ("High", f"≥ {fmt_pct(THRESHOLD, 0)}")])
    html(f'<div class="eg-legend-row">{legend}</div>')

    b1, b2, _ = st.columns([1, 1, 2.5])
    b1.button("Fill low-risk example", on_click=load_profile, args=(DEFAULT_PROFILE,), width="stretch")
    b2.button("Fill high-risk example", on_click=load_profile, args=(HIGH_RISK_PROFILE,), width="stretch")

    with st.form("single_form"):
        html('<div class="eg-form-title">Academic performance</div>')
        s1, s2 = st.columns(2, gap="large")
        with s1:
            st.caption("Semester 1")
            st.number_input("Courses enrolled", 0, 30, key="Curricular_units_1st_sem_enrolled")
            st.number_input("Courses evaluated (exams taken)", 0, 50, key="Curricular_units_1st_sem_evaluations")
            st.number_input("Courses passed", 0, 30, key="Curricular_units_1st_sem_approved")
            st.number_input("Average grade (0-20)", 0.0, 20.0, step=0.1, format="%.1f",
                            key="Curricular_units_1st_sem_grade")
        with s2:
            st.caption("Semester 2")
            st.number_input("Courses enrolled ", 0, 30, key="Curricular_units_2nd_sem_enrolled")
            st.number_input("Courses evaluated (exams taken) ", 0, 50, key="Curricular_units_2nd_sem_evaluations")
            st.number_input("Courses passed ", 0, 30, key="Curricular_units_2nd_sem_approved")
            st.number_input("Average grade (0-20) ", 0.0, 20.0, step=0.1, format="%.1f",
                            key="Curricular_units_2nd_sem_grade")

        st.divider()
        html('<div class="eg-form-title">Financial</div>')
        f1, f2, f3 = st.columns(3)
        yes_no = [1, 0]
        f1.radio("Tuition paid up?", yes_no, format_func=lambda v: "Yes" if v else "No",
                 horizontal=True, key="Tuition_fees_up_to_date")
        f2.radio("In debt?", yes_no, format_func=lambda v: "Yes" if v else "No",
                 horizontal=True, key="Debtor")
        f3.radio("Scholarship holder?", yes_no, format_func=lambda v: "Yes" if v else "No",
                 horizontal=True, key="Scholarship_holder")

        st.divider()
        html('<div class="eg-form-title">Profile & admission</div>')
        p1, p2, p3 = st.columns(3)
        with p1:
            st.selectbox("Study program", sorted(COURSE, key=COURSE.get), format_func=COURSE.get, key="Course")
            st.selectbox("Admission path", list(APPLICATION_MODE), format_func=APPLICATION_MODE.get,
                         key="Application_mode")
            st.radio("Class time", [1, 0], format_func=ATTENDANCE.get, horizontal=True,
                     key="Daytime_evening_attendance")
        with p2:
            st.number_input("Age at enrollment", 16, 80, key="Age_at_enrollment")
            st.radio("Gender", [0, 1], format_func=GENDER.get, horizontal=True, key="Gender")
            st.selectbox("Marital status", list(MARITAL_STATUS), format_func=MARITAL_STATUS.get,
                         key="Marital_status")
        with p3:
            st.number_input("Admission grade (0-200)", 0.0, 200.0, step=0.1, format="%.1f", key="Admission_grade")
            st.number_input("Previous qualification grade (0-200)", 0.0, 200.0, step=0.1, format="%.1f",
                            key="Previous_qualification_grade")
            st.radio("Displaced (away from home)?", yes_no, format_func=lambda v: "Yes" if v else "No",
                     horizontal=True, key="Displaced")

        submitted = st.form_submit_button("Predict dropout risk", type="primary", width="stretch")

    if submitted:
        row = {f: st.session_state[f] for f in MODEL_FEATURES}
        issues = [
            f"Semester {s}: courses passed cannot exceed courses enrolled."
            for s, key in [("1", "1st"), ("2", "2nd")]
            if row[f"Curricular_units_{key}_sem_approved"] > row[f"Curricular_units_{key}_sem_enrolled"]
        ]
        if issues:
            for msg in issues:
                st.error(msg)
        else:
            row_df = pd.DataFrame([row])
            prob = float(predict(row_df)[0])
            level = risk_level(prob)
            color = RISK_STYLE[level]["color"]
            verdict = "Likely to drop out" if prob >= THRESHOLD else "Unlikely to drop out"

            section("Analysis", "Prediction result")
            r1, r2 = st.columns([1, 1.5], gap="large")
            with r1:
                html(f"""<div class="eg-result" style="--accent:{color};">
                    <div class="eg-stat-label">Risk level</div>
                    <div class="eg-result-level">{dot(color)}{level} risk</div>
                    <div class="eg-result-note">{verdict} · {RISK_STYLE[level]['action']}</div>
                </div>""")
                show(gauge(prob, level))
            with r2:
                chart_title("Factors that most affect the prediction", "Compared with other students")
                show(contribution_chart(explain(row_df)))
                st.caption("Bar length = how much the dropout probability changes when this feature is swapped "
                           "for other students' values while every other feature stays the same.")

            section("Follow-up", "Recommendations for the academic advisor")
            html("".join(f'<div class="eg-rec"><span class="eg-rec-num">{i:02d}</span><span>{rec}</span></div>'
                         for i, rec in enumerate(recommendations(row, prob), 1)))

# ===========================================================================
# TAB 3 - BATCH PREDICTION
# ===========================================================================
with tab_batch:
    section("Batch prediction", "Screen many students at once",
            f"A CSV with the {len(MODEL_FEATURES)} columns from the template.", first=True)
    sample_df = pd.read_csv(SAMPLE_PATH)
    u1, u2 = st.columns([2, 1], gap="large")
    with u1:
        uploaded = st.file_uploader("Upload CSV (comma or semicolon separated)", type="csv")
    with u2:
        spacer("1.7rem")
        st.download_button("Download CSV template", sample_df.to_csv(index=False).encode("utf-8"),
                           file_name="sample_batch_input.csv", mime="text/csv", width="stretch")
        with st.popover("View required columns", width="stretch"):
            st.dataframe(pd.DataFrame({"column": MODEL_FEATURES,
                                       "description": [FEATURE_NAMES[f] for f in MODEL_FEATURES]}),
                         hide_index=True)

    if uploaded is None:
        use_sample = st.toggle("Use sample data (25 Enrolled students)", value=True)
        batch_df = sample_df.copy() if use_sample else None
    else:
        batch_df = read_uploaded_csv(uploaded)

    if batch_df is not None:
        missing = [c for c in MODEL_FEATURES if c not in batch_df.columns]
        if missing:
            st.error("These columns are missing: " + ", ".join(missing))
        else:
            result = batch_df.copy()
            result["Dropout_probability"] = predict(result)
            result["Risk_level"] = result["Dropout_probability"].map(risk_level)
            result["Prediction"] = np.where(result["Dropout_probability"] >= THRESHOLD, "Dropout", "Graduate")
            result = result.sort_values("Dropout_probability", ascending=False)

            counts = result["Risk_level"].value_counts()
            total = len(result)
            spacer("0.8rem")
            stat_cards([("Students", fmt_int(total), "analysed", C_VIOLET, None)] + [
                (f"{lvl} risk", fmt_int(counts.get(lvl, 0)),
                 f"{fmt_pct(counts.get(lvl, 0) / total, 0)} of total",
                 RISK_STYLE[lvl]["color"], counts.get(lvl, 0) / total)
                for lvl in ["High", "Medium", "Low"]
            ])
            spacer("1.6rem")

            chosen = st.pills("Filter by risk level", ["High", "Medium", "Low"],
                              selection_mode="multi", default=["High", "Medium", "Low"])
            view = result[result["Risk_level"].isin(chosen or [])].copy()
            view["Study_program"] = view["Course"].map(COURSE)
            front = [c for c in ["Student_id"] if c in view.columns] + [
                "Dropout_probability", "Risk_level", "Prediction", "Study_program", "Age_at_enrollment",
                "Tuition_fees_up_to_date", "Debtor", "Scholarship_holder",
                "Curricular_units_1st_sem_approved", "Curricular_units_2nd_sem_approved"]
            st.dataframe(
                view[front + [c for c in view.columns if c not in front]],
                hide_index=True,
                column_config={
                    "Dropout_probability": st.column_config.ProgressColumn(
                        "Dropout prob.", format="percent", min_value=0, max_value=1),
                    "Risk_level": "Risk",
                    "Study_program": "Study program",
                    "Age_at_enrollment": "Age",
                    "Tuition_fees_up_to_date": st.column_config.CheckboxColumn("Tuition paid"),
                    "Debtor": st.column_config.CheckboxColumn("Debtor"),
                    "Scholarship_holder": st.column_config.CheckboxColumn("Scholarship"),
                    "Curricular_units_1st_sem_approved": "Passed sem 1",
                    "Curricular_units_2nd_sem_approved": "Passed sem 2",
                },
            )
            st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"),
                               file_name="dropout_predictions.csv", mime="text/csv", type="primary")

# ===========================================================================
# TAB 4 - ABOUT THE MODEL
# ===========================================================================
with tab_about:
    section("About the model", "How reliable are EduGuard's predictions?", "Measured on the 20% test set.", first=True)
    m = meta["test_metrics"]
    stat_cards([
        ("Recall", fmt_pct(m["recall"], 0),
         f"{round(m['recall'] * 10)} of 10 dropouts detected", C_DROP, m["recall"]),
        ("Precision", fmt_pct(m["precision"], 0),
         f"{round(m['precision'] * 10)} of 10 risk flags correct", C_GRAD, m["precision"]),
        ("F1-score", f"{m['f1']:.2f}", "recall & precision combined", C_ENR, m["f1"]),
        ("ROC-AUC", f"{m['roc_auc']:.2f}", "0.5 = random guess, 1 = perfect",
         C_VIOLET, m["roc_auc"]),
        ("Accuracy", fmt_pct(m["accuracy"], 0),
         "of all predictions correct", C_YELLOW, m["accuracy"]),
    ])
    spacer("1.6rem")
    t1, t2 = st.columns([1, 1.5], gap="large")
    with t1:
        chart_title("How EduGuard makes a prediction")
        steps = [
            ("Data", f"{fmt_int(meta['n_model_rows'])} students who already dropped out or graduated, "
                     f"{len(MODEL_FEATURES)} academic, financial, and profile features. "
                     f"Enrolled students are not used for training."),
            ("Processing", "Compute course pass rates, standardise scales, encode categories as numbers."),
            ("Model", f"{meta['model_name']}, trained on {fmt_int(meta['n_train'])} students (80%)."),
            ("Decision", f"≥ {fmt_pct(THRESHOLD, 0)} = high risk (catches "
                         f"{round(m['recall'] * 10)} of 10 dropouts), "
                         f"{fmt_pct(RISK_LOW_MAX, 0)}-{fmt_pct(THRESHOLD, 0)} = medium."),
        ]
        html("".join(f'<div class="eg-rec"><span class="eg-rec-num">{i:02d}</span>'
                     f'<span><strong>{t}.</strong> {d}</span></div>' for i, (t, d) in enumerate(steps, 1)))
    with t2:
        chart_title("Influence of each feature", "Relative scale, strongest = 100 (permutation importance). "
                    "≈ 0 = almost no influence.")
        raw = pd.Series(meta["feature_importance"]).rename(index=FEATURE_NAMES).iloc[::-1]
        rel = (raw.clip(lower=0) / raw.max() * 100)
        # Raw values < 0.001 (including slightly negative ones) are within random variation -> "≈ 0".
        labels = ["\u00a0" + (f"{r:.0f}" if v >= 0.001 else "\u2248 0") for v, r in zip(raw.values, rel.values)]
        y_labels = ["\u00a0\u00a0\u00a0" + s for s in rel.index]
        fig = go.Figure(go.Bar(x=rel.values, y=y_labels, orientation="h", width=0.62, marker_color=C_GRAD,
                               customdata=list(rel.index), hovertemplate="%{customdata}: %{x:.1f}<extra></extra>"))
        fig.add_scatter(x=rel.values, y=y_labels, mode="text", text=labels, hoverinfo="skip",
                        textposition="middle right", textfont=dict(color=INK_SOFT), cliponaxis=False,
                        showlegend=False)
        fig.update_xaxes(range=[0, 112], title_text="Relative influence (0-100)", title_font_size=12)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SOFT))
        show(style_fig(fig, 26 * len(rel) + 80))
    st.caption("Predictions support decisions; they are not verdicts.")

html("""<div class="eg-footer">
    <span>EduGuard · Jaya Jaya Institut</span>
    <span>Final Project · Belajar Penerapan Data Science</span>
</div>""")
