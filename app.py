"""EduGuard - prototype sistem deteksi dini mahasiswa berisiko dropout (Jaya Jaya Institut).

Jalankan secara lokal:
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
SAMPLE_PATH = BASE_DIR / "data" / "contoh_input_batch.csv"
DATA_PATH = BASE_DIR / "data" / "data.csv"
CLEAN_PATH = BASE_DIR / "data" / "students_clean.csv"
CSS_PATH = BASE_DIR / "assets" / "style.css"

# Palet: biru = aksen UI; oranye/hijau/biru = status data. Tanpa gradasi.
C_DROP, C_ENR, C_GRAD = "#d95926", "#199e70", "#3987e5"
C_VIOLET, C_YELLOW = "#9085e9", "#c98500"  # warna kartu non-status
AVG_LINE = "#e6e9ef"  # garis rata-rata di grafik dropout rate
C_NEUTRAL, INK, INK_SOFT, MUTED = "#3b4559", "#e6e9ef", "#cfd5e1", "#aeb8cc"
GRID = "rgba(255,255,255,0.06)"
FONT = "Plus Jakarta Sans, sans-serif"
STATUS_COLORS = {"Dropout": C_DROP, "Enrolled": C_ENR, "Graduate": C_GRAD}

# Tingkat risiko (selalu disertai label teks)
RISK_STYLE = {
    "Rendah": {"color": C_ENR, "action": "Pemantauan rutin"},
    "Sedang": {"color": "#e0a526", "action": "Pantau lebih dekat"},
    "Tinggi": {"color": C_DROP, "action": "Prioritas bimbingan khusus"},
}

LOGO_SVG = """<svg width="{size}" height="{size}" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-label="EduGuard">
<path d="M24 3 L41 9.5 V23 C41 34 33.6 41.6 24 45 C14.4 41.6 7 34 7 23 V9.5 Z" fill="#3987e5"/>
<path d="M24 15.5 L36.5 21.2 L24 26.9 L11.5 21.2 Z" fill="#0b1120"/>
<path d="M16.8 24 V29.6 C16.8 29.6 19.6 32.8 24 32.8 C28.4 32.8 31.2 29.6 31.2 29.6 V24 L24 27.3 Z" fill="#0b1120"/>
<path d="M34.6 22.3 V28.4" stroke="#0b1120" stroke-width="1.6" stroke-linecap="round"/>
</svg>"""

FEATURE_NAMES = {
    "Marital_status": "Status pernikahan",
    "Application_mode": "Jalur masuk",
    "Course": "Program studi",
    "Daytime_evening_attendance": "Waktu kuliah",
    "Admission_grade": "Nilai masuk",
    "Previous_qualification_grade": "Nilai pendidikan sebelumnya",
    "Displaced": "Mahasiswa perantau",
    "Debtor": "Memiliki utang",
    "Tuition_fees_up_to_date": "Uang kuliah lunas",
    "Gender": "Gender",
    "Scholarship_holder": "Penerima beasiswa",
    "Age_at_enrollment": "Usia saat mendaftar",
    "Curricular_units_1st_sem_enrolled": "MK diambil (smt 1)",
    "Curricular_units_1st_sem_evaluations": "MK dievaluasi (smt 1)",
    "Curricular_units_1st_sem_approved": "MK lulus (smt 1)",
    "Curricular_units_1st_sem_grade": "Rata-rata nilai (smt 1)",
    "Curricular_units_2nd_sem_enrolled": "MK diambil (smt 2)",
    "Curricular_units_2nd_sem_evaluations": "MK dievaluasi (smt 2)",
    "Curricular_units_2nd_sem_approved": "MK lulus (smt 2)",
    "Curricular_units_2nd_sem_grade": "Rata-rata nilai (smt 2)",
    "Approval_rate_1st_sem": "Rasio kelulusan MK (smt 1)",
    "Approval_rate_2nd_sem": "Rasio kelulusan MK (smt 2)",
}

# Profil contoh untuk demo cepat
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
# Format angka (gaya Indonesia)
# ===========================================================================
def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_pct(x: float, decimals: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{decimals}f}%".replace(".", ",")


# ===========================================================================
# Komponen UI (HTML)
# ===========================================================================
def html(markup: str):
    """Render HTML; baris diratakan kiri agar tidak dianggap blok kode oleh Markdown."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def dot(color: str) -> str:
    return f'<span class="eg-dot" style="background:{color};"></span>'


def section(kicker: str, title: str, description: str | None = None, first: bool = False):
    desc = f"<p>{description}</p>" if description else ""
    html(f"""<div class="eg-section{' first' if first else ''}">
        <div class="eg-kicker">{kicker}</div><h2>{title}</h2>{desc}
    </div>""")


def stat_cards(cards: list[tuple]):
    """cards: (label, value, sub, color | None, meter_share | None). color=None = kartu netral."""
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
    """Judul grafik. rate_key=True menampilkan legenda cara membaca grafik dropout rate."""
    sub = f'<p class="eg-chart-sub">{subtitle}</p>' if subtitle else ""
    key = ""
    if rate_key:
        key = (f'<div class="eg-key"><span><i style="background:{C_DROP};"></i>di atas rata-rata segmen</span>'
               f'<span><i style="background:{C_NEUTRAL};"></i>di bawah rata-rata</span>'
               f'<span><i class="line"></i>rata-rata segmen</span></div>')
    html(f'<p class="eg-chart-title">{title}</p>{sub}{key}')


def insight(stat: str, caption: str, actions: list[str], color: str = C_DROP):
    """Kartu temuan: angka kunci, satu kalimat (termasuk pembanding), lalu aksi."""
    items = "".join(f"<li>{a}</li>" for a in actions)
    html(f"""<div class="eg-insight" style="--accent:{color};">
        <div class="eg-insight-stat">{stat}</div>
        <div class="eg-insight-caption">{caption}</div>
        <div class="eg-actions"><div class="eg-actions-title">Aksi</div><ul>{items}</ul></div>
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
        <div class="eg-chip">{dot(C_ENR)}Model aktif: {meta['model_name']} · Recall {fmt_pct(m['recall'], 0)}</div>
    </div>""")


def hero(students: pd.DataFrame):
    n = len(students)
    counts = students["Status"].value_counts()
    strip = "".join(f'<div style="width:{counts.get(s, 0) / n * 100:.2f}%;background:{c};"></div>'
                    for s, c in STATUS_COLORS.items())
    legend = "".join(f"""<div class="eg-legend-item">
            <div class="lbl">{dot(c)}{s}</div>
            <div class="val" style="color:{c};">{fmt_pct(counts.get(s, 0) / n)}</div>
            <div class="sub">{fmt_int(counts.get(s, 0))} mahasiswa</div>
        </div>""" for s, c in STATUS_COLORS.items())
    html(f"""<div class="eg-hero">
        <div>
            <div class="eg-kicker">Sistem peringatan dini dropout</div>
            <h1>Temukan mahasiswa berisiko <span class="eg-hl">sebelum terlambat.</span></h1>
            <p>Pantau penyebab dropout dan temukan mahasiswa yang perlu bimbingan lebih awal.</p>
        </div>
        <div class="eg-cohort">
            <div class="eg-cohort-title"><span>Status akhir seluruh mahasiswa</span><strong>{fmt_int(n)}</strong></div>
            <div class="eg-strip">{strip}</div>
            <div class="eg-legend">{legend}</div>
            <div class="eg-cohort-note"><strong>Enrolled</strong> = terlambat lulus ·
            <strong>Graduate</strong> = lulus</div>
        </div>
    </div>""")


# ===========================================================================
# Grafik (Plotly)
# ===========================================================================
PLOT_CONFIG = {"displayModeBar": False}


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=24, r=28, t=28, b=8), separators=",.",
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
    return pd.DataFrame([{"kondisi": k, "rate": df.loc[m, "Is_dropout"].mean(), "n": int(m.sum())}
                         for k, m in conditions.items() if m.sum() > 0])


SMALL_N = 30  # kelompok dengan mahasiswa < 30 dianggap kurang representatif


def rate_chart(t: pd.DataFrame, col: str, overall: float, horizontal: bool = True,
               height: int = 330, sort: bool = True) -> go.Figure:
    """Bar dropout rate. Oranye = di atas rata-rata segmen, abu-abu = di bawah rata-rata.

    Kelompok dengan n < SMALL_N digambar transparan dan diberi label jumlah mahasiswanya.
    """
    if sort:
        t = t.sort_values("rate", ascending=horizontal)
    colors = [C_DROP if r > overall else C_NEUTRAL for r in t["rate"]]
    opacity = [0.4 if n < SMALL_N else 1.0 for n in t["n"]]
    text = [fmt_pct(r, 0) + (f"  (n={n}, sedikit)" if n < SMALL_N else "") for r, n in zip(t["rate"], t["n"])]
    custom = np.stack([t[col].astype(str), [fmt_pct(r) for r in t["rate"]], [fmt_int(v) for v in t["n"]]], axis=1)
    hover = "<b>%{customdata[0]}</b><br>Dropout rate: %{customdata[1]}<br>Mahasiswa: %{customdata[2]}<extra></extra>"
    common = dict(marker_color=colors, marker_opacity=opacity, text=text, textposition="outside",
                  customdata=custom, hovertemplate=hover, cliponaxis=False,
                  textfont=dict(color=INK_SOFT, size=13))
    if horizontal:
        # Spasi di depan label memberi ruang cadangan: Plotly menghitung margin sebelum web font termuat.
        labels = ["   " + str(v) for v in t[col]]
        fig = go.Figure(go.Bar(x=t["rate"], y=labels, orientation="h", **common))
    else:
        fig = go.Figure(go.Bar(x=t[col], y=t["rate"], **common))
    # Garis rata-rata: terang, digambar di atas batang; labelnya di luar area plot agar tidak menimpa batang.
    line = dict(line_dash="dash", line_color=AVG_LINE, line_width=1.6, layer="above")
    label = dict(text=f"<b>rata-rata {fmt_pct(overall, 0)}</b>", showarrow=False, font=dict(color=INK, size=12),
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
    """Grouped bar rata-rata metrik per status. Semua metrik dalam satu grafik harus bersatuan sama."""
    grouped = df.groupby("Status")[list(value_cols)].mean()
    fig = go.Figure()
    for status in STATUS_COLORS:
        if status not in grouped.index:
            continue
        vals = grouped.loc[status]
        fig.add_bar(name=status, x=list(value_cols.values()), y=vals.values, marker_color=STATUS_COLORS[status],
                    text=[f"{v:.1f}".replace(".", ",") for v in vals.values], textposition="outside",
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
        return "Tinggi"
    if prob >= RISK_LOW_MAX:
        return "Sedang"
    return "Rendah"


def predict(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[MODEL_FEATURES])[:, 1]


@st.cache_resource
def reference_sample(n: int = 200) -> pd.DataFrame:
    """Sampel mahasiswa dari data pemodelan (Dropout + Graduate), dipakai sebagai pembanding."""
    data = pd.read_csv(DATA_PATH, sep=";")
    data = data[data["Status"].isin(["Dropout", "Graduate"])]
    return data[MODEL_FEATURES].sample(n, random_state=42).reset_index(drop=True)


def format_value(feature: str, value) -> str:
    """Format nilai fitur agar mudah dibaca pada grafik kontribusi."""
    lookups = {"Course": COURSE, "Application_mode": APPLICATION_MODE, "Marital_status": MARITAL_STATUS,
               "Daytime_evening_attendance": ATTENDANCE, "Gender": GENDER}
    if feature in lookups:
        return lookups[feature].get(int(value), str(value))
    if feature in {"Displaced", "Debtor", "Tuition_fees_up_to_date", "Scholarship_holder"}:
        return "Ya" if int(value) == 1 else "Tidak"
    if feature.startswith("Approval_rate"):
        return fmt_pct(value, 0)
    if feature.endswith("_grade"):
        return f"{value:.1f}".replace(".", ",")
    return f"{int(value)}"


def explain(row: pd.DataFrame) -> pd.Series:
    """Kontribusi tiap fitur dengan analisis what-if (berlaku untuk jenis model apa pun).

    kontribusi = probabilitas mahasiswa ini - rata-rata probabilitas jika nilai satu fitur diganti
    dengan nilai mahasiswa lain (sampel data pemodelan), sementara fitur lain tetap.
    Nilai positif berarti fitur tersebut menaikkan risiko, negatif menurunkan risiko.
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
        recs.append("<strong>Tidak mengambil MK di semester 2.</strong> Segera hubungi, indikasi kuat berhenti.")
    if min(rate1, rate2) < 0.5:
        recs.append("<strong>Kelulusan MK rendah.</strong> Bimbingan intensif dosen wali dan tutor sebaya.")
    if row["Tuition_fees_up_to_date"] == 0:
        recs.append("<strong>Uang kuliah menunggak.</strong> Tawarkan cicilan atau keringanan lewat bagian keuangan.")
    if row["Debtor"] == 1:
        recs.append("<strong>Memiliki utang.</strong> Konseling finansial dan jadwal ulang pembayaran.")
    if row["Scholarship_holder"] == 0 and prob >= RISK_LOW_MAX:
        recs.append("<strong>Belum menerima beasiswa.</strong> Informasikan peluang beasiswa.")
    if row["Age_at_enrollment"] >= 25 or row["Daytime_evening_attendance"] == 0:
        recs.append("<strong>Mahasiswa dewasa / kelas malam.</strong> Jadwal fleksibel dan konseling waktu.")
    if not recs:
        recs.append("Tidak ada faktor risiko menonjol. Lanjutkan pemantauan rutin.")
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
    direction = ["menaikkan risiko" if v > 0 else "menurunkan risiko" for v in top.values]
    fig = go.Figure(go.Bar(x=top.values, y=["\u00a0\u00a0\u00a0" + s for s in top.index], orientation="h",
                           marker_color=colors, width=0.6, customdata=np.stack([top.index, direction], axis=1),
                           hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    # Angka mentahnya tidak perlu dibaca pengguna; yang penting arah dan panjang relatif batang.
    fig.update_xaxes(showticklabels=False, showgrid=False,
                     title_text="menurunkan risiko  \u2190   \u2192  menaikkan risiko", title_font_size=12)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SOFT))
    return style_fig(fig, 350)


def read_uploaded_csv(uploaded) -> pd.DataFrame:
    """Baca CSV unggahan; pemisah (koma/titik koma) dideteksi otomatis."""
    raw = uploaded.getvalue().decode("utf-8-sig")
    header = raw.splitlines()[0] if raw else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(io.StringIO(raw), sep=sep)


def load_profile(profile: dict):
    for key, value in profile.items():
        st.session_state[key] = value


# ===========================================================================
# Layout utama
# ===========================================================================
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

students = load_students()
navbar()
hero(students)

if "Course" not in st.session_state:
    load_profile(DEFAULT_PROFILE)

tab_dash, tab_single, tab_batch, tab_about = st.tabs(
    ["Dashboard", "Prediksi Individu", "Prediksi Batch", "Tentang Model"])

# ===========================================================================
# TAB 1 - DASHBOARD
# ===========================================================================
with tab_dash:
    section("Monitoring", "Dashboard dropout mahasiswa", first=True)

    f1, f2, f3 = st.columns([2.2, 1, 1])
    course_filter = f1.multiselect("Program studi", sorted(students["Course_label"].unique()),
                                   placeholder="Semua program studi")
    gender_filter = f2.segmented_control("Gender", ["Female", "Male"], selection_mode="multi",
                                         default=["Female", "Male"],
                                         format_func=lambda v: {"Female": "Perempuan", "Male": "Laki-laki"}[v])
    time_filter = f3.segmented_control("Waktu kuliah", ["Daytime", "Evening"], selection_mode="multi",
                                       default=["Daytime", "Evening"],
                                       format_func=lambda v: {"Daytime": "Siang", "Evening": "Malam"}[v])

    df = students
    if course_filter:
        df = df[df["Course_label"].isin(course_filter)]
    df = df[df["Gender_label"].isin(gender_filter or []) & df["Attendance_label"].isin(time_filter or [])]

    if df.empty:
        st.warning("Tidak ada mahasiswa yang cocok dengan filter. Ubah pilihan filter di atas.")
    else:
        n = len(df)
        overall = df["Is_dropout"].mean()
        counts = df["Status"].value_counts()

        late_fee = int((df["Tuition_fees_up_to_date"] == 0).sum())

        spacer("0.6rem")
        stat_cards([
            ("Mahasiswa", fmt_int(n), f"dari {fmt_int(len(students))} mahasiswa", C_VIOLET, n / len(students)),
            ("Dropout", fmt_pct(overall), f"{fmt_int(counts.get('Dropout', 0))} mahasiswa", C_DROP, overall),
            ("Enrolled", fmt_pct(counts.get("Enrolled", 0) / n),
             f"{fmt_int(counts.get('Enrolled', 0))} mahasiswa", C_ENR, counts.get("Enrolled", 0) / n),
            ("Graduate", fmt_pct(counts.get("Graduate", 0) / n),
             f"{fmt_int(counts.get('Graduate', 0))} mahasiswa", C_GRAD, counts.get("Graduate", 0) / n),
            ("Menunggak UKT", fmt_int(late_fee), f"{fmt_pct(late_fee / n)} dari segmen", C_YELLOW, late_fee / n),
        ])

        section("Ringkasan", "Kondisi yang paling sering berujung dropout")
        chart_title("Dropout rate menurut kondisi mahasiswa", rate_key=True)
        profile = conditions_table(df, {
            "UKT menunggak": df["Tuition_fees_up_to_date"] == 0,
            "0 MK lulus di semester 2": df["Curricular_units_2nd_sem_approved"] == 0,
            "Memiliki utang": df["Debtor"] == 1,
            "Usia masuk ≥ 25 tahun": df["Age_at_enrollment"] >= 25,
            "Jalur Over 23 years old": df["Application_mode"] == 39,
            "Laki-laki": df["Gender"] == 1,
            "Kuliah malam": df["Daytime_evening_attendance"] == 0,
            "Penerima beasiswa (pembanding)": df["Scholarship_holder"] == 1,
        })
        show(rate_chart(profile, "kondisi", overall, height=340))

        # ── 01 Akademik ────────────────────────────────────────────
        section("Pertanyaan bisnis 01 · Akademik", "Apakah nilai semester awal bisa jadi sinyal dini?")
        appr = rate_table(df, "Approved_2nd_group", order=["0 MK", "1-2 MK", "3-4 MK", "5-6 MK", ">6 MK"])
        ar = appr.set_index("Approved_2nd_group")["rate"]
        a1, a2 = st.columns([1.7, 1], gap="large")
        with a1:
            chart_title("Dropout rate berdasarkan jumlah MK lulus di semester 2")
            show(rate_chart(appr, "Approved_2nd_group", overall, horizontal=False, sort=False))
        with a2:
            insight(fmt_pct(ar.get("0 MK", np.nan), 0),
                    "mahasiswa yang tidak lulus satu pun MK di semester 2 dropout "
                    f"(vs {fmt_pct(ar.get('5-6 MK', np.nan), 0)} yang lulus 5-6 MK)",
                    ["Pantau jumlah MK lulus tiap semester", "Rasio lulus di bawah 50% wajib bimbingan"])
        spacer("1.4rem")
        b1, b2 = st.columns(2, gap="large")
        with b1:
            chart_title("Rata-rata jumlah MK lulus per status")
            show(status_bar_chart(df, {
                "Curricular_units_1st_sem_approved": "Semester 1",
                "Curricular_units_2nd_sem_approved": "Semester 2",
            }, y_title="Jumlah MK lulus", y_max=8))
        with b2:
            chart_title("Rata-rata nilai semester per status", "Skala nilai 0-20")
            show(status_bar_chart(df, {
                "Curricular_units_1st_sem_grade": "Semester 1",
                "Curricular_units_2nd_sem_grade": "Semester 2",
            }, y_title="Nilai rata-rata", y_max=20))

        # ── 02 Finansial ───────────────────────────────────────────
        section("Pertanyaan bisnis 02 · Finansial", "Seberapa besar pengaruh kondisi finansial?")
        fin = conditions_table(df, {
            "UKT menunggak": df["Tuition_fees_up_to_date"] == 0,
            "UKT lunas": df["Tuition_fees_up_to_date"] == 1,
            "Memiliki utang": df["Debtor"] == 1,
            "Tidak memiliki utang": df["Debtor"] == 0,
            "Bukan penerima beasiswa": df["Scholarship_holder"] == 0,
            "Penerima beasiswa": df["Scholarship_holder"] == 1,
        })
        fr = fin.set_index("kondisi")["rate"]
        d1, d2 = st.columns([1, 1.7], gap="large")
        with d1:
            insight(fmt_pct(fr.get("UKT menunggak", np.nan), 0),
                    "mahasiswa yang menunggak UKT dropout "
                    f"(vs {fmt_pct(fr.get('Penerima beasiswa', np.nan), 0)} penerima beasiswa)",
                    ["Skema cicilan atau keringanan UKT", "Perluas beasiswa"])
        with d2:
            chart_title("Dropout rate berdasarkan kondisi finansial")
            show(rate_chart(fin, "kondisi", overall, sort=False).update_yaxes(autorange="reversed"))

        # ── 03 Demografi ───────────────────────────────────────────
        section("Pertanyaan bisnis 03 · Demografi", "Siapa yang paling rentan dropout?")
        age = rate_table(df, "Age_group", order=["<=20", "21-24", "25-29", "30-39", "40+"])
        agr = age.set_index("Age_group")["rate"]
        e1, e2 = st.columns([1.7, 1], gap="large")
        with e1:
            chart_title("Dropout rate per kelompok usia saat mendaftar")
            show(rate_chart(age, "Age_group", overall, horizontal=False, sort=False))
        with e2:
            insight(fmt_pct(agr.get("25-29", np.nan), 0),
                    "mahasiswa yang masuk di usia 25-29 tahun dropout "
                    f"(vs {fmt_pct(agr.get('<=20', np.nan), 0)} usia ≤20 tahun)",
                    ["Kelas fleksibel atau daring", "Konseling manajemen waktu"])

        # ── 04 Program studi ───────────────────────────────────────
        section("Pertanyaan bisnis 04 · Prodi & jalur masuk", "Prodi dan jalur masuk mana yang perlu perhatian?")
        course = rate_table(df, "Course_label")
        big = course[course["n"] >= 30].sort_values("rate", ascending=False)
        g1, g2 = st.columns([1, 1.7], gap="large")
        with g1:
            if big.empty:
                insight("-", "tidak ada prodi dengan minimal 30 mahasiswa pada segmen ini",
                        ["Perluas pilihan filter"])
            else:
                top = big.iloc[0]
                insight(fmt_pct(top["rate"], 0),
                        f"mahasiswa {top['Course_label']} dropout, tertinggi di antara prodi (min. 30 mahasiswa)",
                        ["Program mentoring khusus per prodi", "Evaluasi kurikulum tahun pertama"])
        with g2:
            chart_title("Dropout rate per program studi", f"Transparan = kurang dari {SMALL_N} mahasiswa")
            show(rate_chart(course, "Course_label", overall, height=max(340, 26 * len(course))))
        spacer("1.4rem")
        chart_title(f"Dropout rate per jalur masuk (min. {SMALL_N} mahasiswa)")
        mode = rate_table(df, "Application_mode_label", min_n=30)
        if mode.empty:
            st.caption("Tidak ada jalur masuk dengan minimal 30 mahasiswa pada segmen ini.")
        else:
            show(rate_chart(mode, "Application_mode_label", overall, height=max(260, 32 * len(mode))))

# ===========================================================================
# TAB 2 - PREDIKSI INDIVIDU
# ===========================================================================
with tab_single:
    section("Prediksi individu", "Seberapa besar risiko dropout mahasiswa ini?", first=True)
    legend = "".join(
        f'<span class="eg-chip">{dot(RISK_STYLE[lvl]["color"])}{lvl} {rng}</span>'
        for lvl, rng in [("Rendah", f"&lt; {fmt_pct(RISK_LOW_MAX, 0)}"),
                         ("Sedang", f"{fmt_pct(RISK_LOW_MAX, 0)} - {fmt_pct(THRESHOLD, 0)}"),
                         ("Tinggi", f"≥ {fmt_pct(THRESHOLD, 0)}")])
    html(f'<div class="eg-legend-row">{legend}</div>')

    b1, b2, _ = st.columns([1, 1, 2.5])
    b1.button("Isi contoh risiko rendah", on_click=load_profile, args=(DEFAULT_PROFILE,), width="stretch")
    b2.button("Isi contoh risiko tinggi", on_click=load_profile, args=(HIGH_RISK_PROFILE,), width="stretch")

    with st.form("single_form"):
        html('<div class="eg-form-title">Performa akademik</div>')
        s1, s2 = st.columns(2, gap="large")
        with s1:
            st.caption("Semester 1")
            st.number_input("Jumlah MK diambil", 0, 30, key="Curricular_units_1st_sem_enrolled")
            st.number_input("Jumlah MK dievaluasi (ujian diikuti)", 0, 50, key="Curricular_units_1st_sem_evaluations")
            st.number_input("Jumlah MK lulus", 0, 30, key="Curricular_units_1st_sem_approved")
            st.number_input("Rata-rata nilai (0-20)", 0.0, 20.0, step=0.1, format="%.1f",
                            key="Curricular_units_1st_sem_grade")
        with s2:
            st.caption("Semester 2")
            st.number_input("Jumlah MK diambil ", 0, 30, key="Curricular_units_2nd_sem_enrolled")
            st.number_input("Jumlah MK dievaluasi (ujian diikuti) ", 0, 50, key="Curricular_units_2nd_sem_evaluations")
            st.number_input("Jumlah MK lulus ", 0, 30, key="Curricular_units_2nd_sem_approved")
            st.number_input("Rata-rata nilai (0-20) ", 0.0, 20.0, step=0.1, format="%.1f",
                            key="Curricular_units_2nd_sem_grade")

        st.divider()
        html('<div class="eg-form-title">Finansial</div>')
        f1, f2, f3 = st.columns(3)
        yes_no = [1, 0]
        f1.radio("Uang kuliah lunas?", yes_no, format_func=lambda v: "Ya" if v else "Tidak",
                 horizontal=True, key="Tuition_fees_up_to_date")
        f2.radio("Memiliki utang?", yes_no, format_func=lambda v: "Ya" if v else "Tidak",
                 horizontal=True, key="Debtor")
        f3.radio("Penerima beasiswa?", yes_no, format_func=lambda v: "Ya" if v else "Tidak",
                 horizontal=True, key="Scholarship_holder")

        st.divider()
        html('<div class="eg-form-title">Profil & pendaftaran</div>')
        p1, p2, p3 = st.columns(3)
        with p1:
            st.selectbox("Program studi", sorted(COURSE, key=COURSE.get), format_func=COURSE.get, key="Course")
            st.selectbox("Jalur masuk", list(APPLICATION_MODE), format_func=APPLICATION_MODE.get,
                         key="Application_mode")
            st.radio("Waktu kuliah", [1, 0], format_func=ATTENDANCE.get, horizontal=True,
                     key="Daytime_evening_attendance")
        with p2:
            st.number_input("Usia saat mendaftar", 16, 80, key="Age_at_enrollment")
            st.radio("Gender", [0, 1], format_func=GENDER.get, horizontal=True, key="Gender")
            st.selectbox("Status pernikahan", list(MARITAL_STATUS), format_func=MARITAL_STATUS.get,
                         key="Marital_status")
        with p3:
            st.number_input("Nilai masuk (0-200)", 0.0, 200.0, step=0.1, format="%.1f", key="Admission_grade")
            st.number_input("Nilai pendidikan sebelumnya (0-200)", 0.0, 200.0, step=0.1, format="%.1f",
                            key="Previous_qualification_grade")
            st.radio("Mahasiswa perantau?", yes_no, format_func=lambda v: "Ya" if v else "Tidak",
                     horizontal=True, key="Displaced")

        submitted = st.form_submit_button("Prediksi risiko dropout", type="primary", width="stretch")

    if submitted:
        row = {f: st.session_state[f] for f in MODEL_FEATURES}
        issues = [
            f"Semester {s}: jumlah MK lulus melebihi MK diambil."
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
            verdict = "Berpotensi dropout" if prob >= THRESHOLD else "Tidak berpotensi dropout"

            section("Hasil analisis", "Hasil prediksi")
            r1, r2 = st.columns([1, 1.5], gap="large")
            with r1:
                html(f"""<div class="eg-result" style="--accent:{color};">
                    <div class="eg-stat-label">Tingkat risiko</div>
                    <div class="eg-result-level">{dot(color)}Risiko {level}</div>
                    <div class="eg-result-note">{verdict} · {RISK_STYLE[level]['action']}</div>
                </div>""")
                show(gauge(prob, level))
            with r2:
                chart_title("Faktor yang paling memengaruhi prediksi", "Dibandingkan mahasiswa lain")
                show(contribution_chart(explain(row_df)))
                st.caption("Panjang batang = seberapa besar probabilitas dropout berubah jika nilai fitur itu "
                           "diganti nilai mahasiswa lain, sementara fitur lain tetap.")

            section("Tindak lanjut", "Rekomendasi untuk dosen wali")
            html("".join(f'<div class="eg-rec"><span class="eg-rec-num">{i:02d}</span><span>{rec}</span></div>'
                         for i, rec in enumerate(recommendations(row, prob), 1)))

# ===========================================================================
# TAB 3 - PREDIKSI BATCH
# ===========================================================================
with tab_batch:
    section("Prediksi batch", "Skrining banyak mahasiswa sekaligus",
            f"CSV berisi {len(MODEL_FEATURES)} kolom sesuai template.", first=True)
    sample_df = pd.read_csv(SAMPLE_PATH)
    u1, u2 = st.columns([2, 1], gap="large")
    with u1:
        uploaded = st.file_uploader("Unggah CSV (pemisah koma atau titik koma)", type="csv")
    with u2:
        spacer("1.7rem")
        st.download_button("Unduh template CSV", sample_df.to_csv(index=False).encode("utf-8"),
                           file_name="contoh_input_batch.csv", mime="text/csv", width="stretch")
        with st.popover("Lihat kolom yang dibutuhkan", width="stretch"):
            st.dataframe(pd.DataFrame({"kolom": MODEL_FEATURES,
                                       "keterangan": [FEATURE_NAMES[f] for f in MODEL_FEATURES]}),
                         hide_index=True)

    if uploaded is None:
        use_sample = st.toggle("Gunakan data contoh (25 mahasiswa Enrolled)", value=True)
        batch_df = sample_df.copy() if use_sample else None
    else:
        batch_df = read_uploaded_csv(uploaded)

    if batch_df is not None:
        missing = [c for c in MODEL_FEATURES if c not in batch_df.columns]
        if missing:
            st.error("Kolom berikut tidak ditemukan: " + ", ".join(missing))
        else:
            result = batch_df.copy()
            result["Probabilitas_dropout"] = predict(result)
            result["Tingkat_risiko"] = result["Probabilitas_dropout"].map(risk_level)
            result["Prediksi"] = np.where(result["Probabilitas_dropout"] >= THRESHOLD, "Dropout", "Graduate")
            result = result.sort_values("Probabilitas_dropout", ascending=False)

            counts = result["Tingkat_risiko"].value_counts()
            total = len(result)
            spacer("0.8rem")
            stat_cards([("Mahasiswa", fmt_int(total), "dianalisis", C_VIOLET, None)] + [
                (f"Risiko {lvl}", fmt_int(counts.get(lvl, 0)),
                 f"{fmt_pct(counts.get(lvl, 0) / total, 0)} dari total",
                 RISK_STYLE[lvl]["color"], counts.get(lvl, 0) / total)
                for lvl in ["Tinggi", "Sedang", "Rendah"]
            ])
            spacer("1.6rem")

            chosen = st.pills("Filter tingkat risiko", ["Tinggi", "Sedang", "Rendah"],
                              selection_mode="multi", default=["Tinggi", "Sedang", "Rendah"])
            view = result[result["Tingkat_risiko"].isin(chosen or [])].copy()
            view["Program_studi"] = view["Course"].map(COURSE)
            front = [c for c in ["Student_id"] if c in view.columns] + [
                "Probabilitas_dropout", "Tingkat_risiko", "Prediksi", "Program_studi", "Age_at_enrollment",
                "Tuition_fees_up_to_date", "Debtor", "Scholarship_holder",
                "Curricular_units_1st_sem_approved", "Curricular_units_2nd_sem_approved"]
            st.dataframe(
                view[front + [c for c in view.columns if c not in front]],
                hide_index=True,
                column_config={
                    "Probabilitas_dropout": st.column_config.ProgressColumn(
                        "Prob. dropout", format="percent", min_value=0, max_value=1),
                    "Tingkat_risiko": "Risiko",
                    "Program_studi": "Program studi",
                    "Age_at_enrollment": "Usia",
                    "Tuition_fees_up_to_date": st.column_config.CheckboxColumn("UKT lunas"),
                    "Debtor": st.column_config.CheckboxColumn("Berutang"),
                    "Scholarship_holder": st.column_config.CheckboxColumn("Beasiswa"),
                    "Curricular_units_1st_sem_approved": "MK lulus smt 1",
                    "Curricular_units_2nd_sem_approved": "MK lulus smt 2",
                },
            )
            st.download_button("Unduh hasil prediksi", result.to_csv(index=False).encode("utf-8"),
                               file_name="hasil_prediksi_dropout.csv", mime="text/csv", type="primary")

# ===========================================================================
# TAB 4 - TENTANG MODEL
# ===========================================================================
with tab_about:
    section("Tentang model", "Seberapa andal prediksi EduGuard?", "Diukur pada 20% data uji.", first=True)
    m = meta["test_metrics"]
    stat_cards([
        ("Recall", fmt_pct(m["recall"], 0),
         f"{round(m['recall'] * 10)} dari 10 dropout terdeteksi", C_DROP, m["recall"]),
        ("Precision", fmt_pct(m["precision"], 0),
         f"{round(m['precision'] * 10)} dari 10 tanda risiko benar", C_GRAD, m["precision"]),
        ("F1-score", f"{m['f1']:.2f}".replace(".", ","), "gabungan recall & precision", C_ENR, m["f1"]),
        ("ROC-AUC", f"{m['roc_auc']:.2f}".replace(".", ","), "0,5 = tebakan acak, 1 = sempurna",
         C_VIOLET, m["roc_auc"]),
        ("Akurasi", fmt_pct(m["accuracy"], 0),
         "seluruh prediksi benar", C_YELLOW, m["accuracy"]),
    ])
    spacer("1.6rem")
    t1, t2 = st.columns([1, 1.5], gap="large")
    with t1:
        chart_title("Cara EduGuard membuat prediksi")
        steps = [
            ("Data", f"{fmt_int(meta['n_model_rows'])} mahasiswa yang sudah Dropout atau Graduate, "
                     f"{len(MODEL_FEATURES)} fitur akademik, finansial, dan profil. "
                     f"Mahasiswa Enrolled tidak dipakai untuk melatih model."),
            ("Olah data", "Hitung rasio kelulusan MK, samakan skala, ubah kategori jadi angka."),
            ("Model", f"{meta['model_name']}, dilatih pada {fmt_int(meta['n_train'])} mahasiswa (80%)."),
            ("Keputusan", f"≥ {fmt_pct(THRESHOLD, 0)} = risiko tinggi (menangkap "
                          f"{round(m['recall'] * 10)} dari 10 dropout), "
                          f"{fmt_pct(RISK_LOW_MAX, 0)}-{fmt_pct(THRESHOLD, 0)} = sedang."),
        ]
        html("".join(f'<div class="eg-rec"><span class="eg-rec-num">{i:02d}</span>'
                     f'<span><strong>{t}.</strong> {d}</span></div>' for i, (t, d) in enumerate(steps, 1)))
    with t2:
        chart_title("Pengaruh setiap fitur", "Skala relatif, terkuat = 100 (permutation importance). "
                    "≈ 0 = hampir tidak berpengaruh.")
        raw = pd.Series(meta["feature_importance"]).rename(index=FEATURE_NAMES).iloc[::-1]
        rel = (raw.clip(lower=0) / raw.max() * 100)
        # Nilai mentah < 0,001 (termasuk yang sedikit negatif) berada di dalam variasi acak -> "≈ 0".
        labels = ["\u00a0" + (f"{r:.0f}" if v >= 0.001 else "\u2248 0") for v, r in zip(raw.values, rel.values)]
        y_labels = ["\u00a0\u00a0\u00a0" + s for s in rel.index]
        fig = go.Figure(go.Bar(x=rel.values, y=y_labels, orientation="h", width=0.62, marker_color=C_GRAD,
                               customdata=list(rel.index), hovertemplate="%{customdata}: %{x:.1f}<extra></extra>"))
        fig.add_scatter(x=rel.values, y=y_labels, mode="text", text=labels, hoverinfo="skip",
                        textposition="middle right", textfont=dict(color=INK_SOFT), cliponaxis=False,
                        showlegend=False)
        fig.update_xaxes(range=[0, 112], title_text="Pengaruh relatif (0-100)", title_font_size=12)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SOFT))
        show(style_fig(fig, 26 * len(rel) + 80))
    st.caption("Prediksi adalah alat bantu keputusan, bukan vonis.")

html("""<div class="eg-footer">
    <span>EduGuard · Jaya Jaya Institut</span>
    <span>Proyek Akhir Belajar Penerapan Data Science</span>
</div>""")
