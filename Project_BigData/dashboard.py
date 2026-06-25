import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Prediksi Kategori SKS Optimal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #f0f4f9; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

    .sidebar-filter-title {
        color: #3b7dd8; font-weight: 700; font-size: 0.85rem;
        letter-spacing: 1px; text-transform: uppercase;
        display: flex; align-items: center; gap: 6px; margin-bottom: 0.5rem;
    }
    .sidebar-info-title {
        color: #5b6abf; font-weight: 700; font-size: 0.85rem;
        letter-spacing: 1px; text-transform: uppercase;
        display: flex; align-items: center; gap: 6px; margin: 1rem 0 0.5rem;
    }
    .info-row { display: flex; justify-content: space-between; font-size: 0.82rem;
                color: #444; padding: 2px 0; }
    .info-val { font-weight: 600; color: #222; }
    .sidebar-footer {
        background: linear-gradient(135deg, #eef2ff, #dbeafe);
        border-radius: 10px; padding: 10px 12px; margin-top: 1rem;
        font-size: 0.78rem; color: #3b5998;
    }

    /* KPI Cards */
    .kpi-card {
        background: white; border-radius: 14px; padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07); height: 110px;
        display: flex; align-items: center; gap: 14px;
    }
    .kpi-icon { font-size: 2rem; }
    .kpi-label { font-size: 0.72rem; color: #888; margin-bottom: 2px; }
    .kpi-value { font-size: 1.7rem; font-weight: 700; color: #1a1a2e; line-height: 1.1; }
    .kpi-sub { font-size: 0.72rem; color: #aaa; }

    /* Chart cards */
    .chart-card {
        background: white; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        margin-bottom: 14px;
    }
    .chart-title {
        font-size: 0.88rem; font-weight: 700; color: #1a1a2e; margin-bottom: 8px;
    }

    /* Segment table */
    .seg-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .seg-table th { color: #888; font-weight: 500; padding: 4px 8px; text-align: left; }
    .seg-table td { padding: 6px 8px; vertical-align: middle; }
    .seg-badge {
        border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; font-weight: 600;
        display: inline-flex; align-items: center; gap: 5px; white-space: nowrap;
    }

    h1.dash-title { font-size: 2rem; font-weight: 800; color: #1a1a2e; margin-bottom: 2px; }
    p.dash-subtitle { color: #888; font-size: 0.88rem; margin-top: 0; }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    [data-testid="stHeader"] { background-color: transparent; }
    div[data-testid="stSlider"] > div { padding-top: 0; }
    div[data-testid="stSelectbox"] > div { font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ─── DATA & MODELS ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("df_final_clean__5_.csv")

@st.cache_resource
def train_models(_df):
    features = ['GPA', 'StudyTimeWeekly', 'Absences', 'PerformanceScore',
                'raisedhands', 'VisITedResources', 'AnnouncementsView',
                'Discussion', 'StudentAbsenceDays', 'EngagementScore']
    X = _df[features]
    y = _df['SKS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    try:
        sm = SMOTE(random_state=42, k_neighbors=2)
        X_res, y_res = sm.fit_resample(X_train, y_train)
    except Exception:
        X_res, y_res = X_train, y_train

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_res, y_res)
    y_pred_rf = rf.predict(X_test)

    metrics = {
        "DT": {
            "Accuracy":  round(accuracy_score(y_test, y_pred_dt)*100, 2),
            "Precision": round(precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)*100, 2),
            "Recall":    round(recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)*100, 2),
            "F1-Score":  round(f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)*100, 2),
        },
        "RF": {
            "Accuracy":  round(accuracy_score(y_test, y_pred_rf)*100, 2),
            "Precision": round(precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)*100, 2),
            "Recall":    round(recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)*100, 2),
            "F1-Score":  round(f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)*100, 2),
        }
    }
    fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return rf, dt, metrics, fi, features

@st.cache_data
def compute_clusters(df_hash, df):
    feats = ['GPA', 'StudyTimeWeekly', 'Absences', 'EngagementScore', 'PerformanceScore']
    X = df[feats].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df2 = df.copy()
    df2['Cluster_raw'] = labels
    eng_mean = df2.groupby('Cluster_raw')['EngagementScore'].mean().sort_values(ascending=False)
    cmap = {eng_mean.index[0]: 'High Engaged', eng_mean.index[1]: 'Mid Engaged', eng_mean.index[2]: 'Low Engaged'}
    df2['ClusterLabel'] = df2['Cluster_raw'].map(cmap)
    return df2

df_raw = load_data()
rf_model, dt_model, metrics, fi, feature_cols = train_models(df_raw)
df_clustered = compute_clusters(id(df_raw), df_raw)

SKS_COLOR_LIST = ["#6366f1", "#22d3ee", "#f97316", "#a855f7", "#10b981"]
CLUSTER_COLORS = {"High Engaged": "#6366f1", "Mid Engaged": "#22d3ee", "Low Engaged": "#a855f7"}

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-filter-title">🔽 FILTER DATA</div>', unsafe_allow_html=True)

    sks_opts = ["Semua"] + sorted(df_raw['SKS'].unique().tolist())
    sel_sks_cat   = st.selectbox("SKS Category", sks_opts)
    sel_sks_total = st.selectbox("SKS (Total)",   sks_opts)

    gpa_min, gpa_max = float(df_raw['GPA'].min()), float(df_raw['GPA'].max())
    gpa_range = st.slider("GPA Range", gpa_min, gpa_max, (round(gpa_min,2), round(gpa_max,2)), step=0.01)

    abs_min, abs_max = int(df_raw['Absences'].min()), int(df_raw['Absences'].max())
    abs_range = st.slider("Absences Range", abs_min, abs_max, (abs_min, abs_max))

    eng_min, eng_max = float(df_raw['EngagementScore'].min()), float(df_raw['EngagementScore'].max())
    eng_range = st.slider("Engagement Score Range", round(eng_min,2), round(eng_max,2),
                          (round(eng_min,2), round(eng_max,2)), step=0.01)

    sad_min, sad_max = int(df_raw['StudentAbsenceDays'].min()), int(df_raw['StudentAbsenceDays'].max())
    sad_range = st.slider("Student Absence Days", sad_min, sad_max, (sad_min, sad_max))

    if st.button("🔄 Reset Filter", use_container_width=True):
        st.rerun()

    st.markdown('<div class="sidebar-info-title">ℹ️ INFO DATASET</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-row"><span>Total Mahasiswa</span><span class="info-val">1,000</span></div>
    <div class="info-row"><span>Jumlah Fitur</span><span class="info-val">16</span></div>
    <div style="margin-top:8px; font-size:0.78rem; color:#888;">
      <b>Sumber Dataset</b><br>Student Performance Dataset &amp; xAPI-Edu-Data
    </div>
    <div style="font-size:0.78rem; color:#888; margin-top:4px;">
      <b>Terakhir Diperbarui</b><br>Mei 2025
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-footer">🎓 Dashboard ini membantu mengambil keputusan akademik berbasis data.</div>',
                unsafe_allow_html=True)

# ─── FILTER DATA ─────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_sks_cat   != "Semua": df = df[df['SKS'] == int(sel_sks_cat)]
if sel_sks_total != "Semua": df = df[df['SKS'] == int(sel_sks_total)]
df = df[df['GPA'].between(gpa_range[0], gpa_range[1])]
df = df[df['Absences'].between(abs_range[0], abs_range[1])]
df = df[df['EngagementScore'].between(eng_range[0], eng_range[1])]
df = df[df['StudentAbsenceDays'].between(sad_range[0], sad_range[1])]

df_cf = compute_clusters(tuple(df.index.tolist()), df) if len(df) >= 3 else df_clustered

total_mhs   = len(df)
pct_total   = round(total_mhs / len(df_raw) * 100, 1)
avg_gpa     = df['GPA'].mean()     if total_mhs else 0
std_gpa     = df['GPA'].std()      if total_mhs else 0
avg_eng     = df['EngagementScore'].mean() if total_mhs else 0
std_eng     = df['EngagementScore'].std()  if total_mhs else 0
avg_abs     = df['Absences'].mean() if total_mhs else 0
std_abs     = df['Absences'].std()  if total_mhs else 0
top_sks     = df['SKS'].value_counts().idxmax() if total_mhs else "-"
top_sks_pct = round(df['SKS'].value_counts().max() / total_mhs * 100, 1) if total_mhs else 0

# ─── HEADER (judul saja, tanpa model badge) ──────────────────────────────────
st.markdown('<h1 class="dash-title">Dashboard Prediksi Kategori SKS Optimal</h1>', unsafe_allow_html=True)
st.markdown('<p class="dash-subtitle">Analisis Mahasiswa Berdasarkan Akademik &amp; Perilaku Belajar</p>', unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─── KPI ROW ─────────────────────────────────────────────────────────────────
def kpi_card(icon, label, value, sub, bg="#ffffff"):
    return f"""<div class="kpi-card" style="background:{bg}">
        <div class="kpi-icon">{icon}</div>
        <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
    </div>"""

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.markdown(kpi_card("👥", "Jumlah Mahasiswa",        f"{total_mhs:,}",   f"{pct_total}% dari total data", "#eef2ff"), unsafe_allow_html=True)
with k2: st.markdown(kpi_card("🎓", "Rata-rata GPA",           f"{avg_gpa:.2f}",   f"± {std_gpa:.2f}",             "#f0fdf4"), unsafe_allow_html=True)
with k3: st.markdown(kpi_card("📊", "Rata-rata Engagement",    f"{avg_eng:.2f}",   f"± {std_eng:.2f}",             "#faf5ff"), unsafe_allow_html=True)
with k4: st.markdown(kpi_card("👤", "Rata-rata Absences",      f"{avg_abs:.2f}",   f"± {std_abs:.2f}",             "#fff7ed"), unsafe_allow_html=True)
with k5: st.markdown(kpi_card("⭐", "Kategori SKS Terbanyak",  f"{top_sks} SKS",   f"({top_sks_pct}%)",            "#fefce8"), unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─── ROW 1: Charts 1, 2, 3 ───────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="chart-card"><div class="chart-title">📊 1. Distribusi Kategori SKS</div>', unsafe_allow_html=True)
    sks_dist  = df['SKS'].value_counts().sort_index()
    text_vals = [f"{v} ({round(v/total_mhs*100,1)}%)" for v in sks_dist.values] if total_mhs else []
    fig1 = go.Figure(go.Bar(
        x=sks_dist.values, y=[f"{s} SKS" for s in sks_dist.index], orientation='h',
        marker_color=SKS_COLOR_LIST[:len(sks_dist)],
        text=text_vals, textposition='outside',
        hovertemplate='%{y}: %{x} mahasiswa<extra></extra>'
    ))
    fig1.update_layout(margin=dict(l=0,r=80,t=10,b=30), height=200,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(title="Jumlah Mahasiswa", showgrid=True, gridcolor='#f0f0f0'),
        yaxis=dict(title="SKS Category"), font=dict(size=11))
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card"><div class="chart-title">📈 2. Rata-rata GPA per Kategori SKS</div>', unsafe_allow_html=True)
    gpa_per_sks = df.groupby('SKS')['GPA'].mean().sort_index()
    fig2 = go.Figure(go.Bar(
        x=[f"{s} SKS" for s in gpa_per_sks.index], y=gpa_per_sks.values,
        marker_color=SKS_COLOR_LIST[:len(gpa_per_sks)],
        text=[f"{v:.2f}" for v in gpa_per_sks.values], textposition='outside',
        hovertemplate='%{x}: GPA %{y:.2f}<extra></extra>'
    ))
    fig2.update_layout(margin=dict(l=0,r=10,t=10,b=30), height=200,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(title="SKS Category"),
        yaxis=dict(title="Rata-rata GPA", showgrid=True, gridcolor='#f0f0f0'),
        font=dict(size=11))
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="chart-card"><div class="chart-title">👤 3. Rata-rata Absences per Kategori SKS</div>', unsafe_allow_html=True)
    abs_per_sks = df.groupby('SKS')['Absences'].mean().sort_index()
    fig3 = go.Figure(go.Bar(
        x=[f"{s} SKS" for s in abs_per_sks.index], y=abs_per_sks.values,
        marker_color=SKS_COLOR_LIST[:len(abs_per_sks)],
        text=[f"{v:.2f}" for v in abs_per_sks.values], textposition='outside',
        hovertemplate='%{x}: Absences %{y:.2f}<extra></extra>'
    ))
    fig3.update_layout(margin=dict(l=0,r=10,t=10,b=30), height=200,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(title="SKS Category"),
        yaxis=dict(title="Rata-rata Absences", showgrid=True, gridcolor='#f0f0f0'),
        font=dict(size=11))
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ─── ROW 2: Charts 4, 5, 6 ───────────────────────────────────────────────────
c4, c5, c6 = st.columns(3)

with c4:
    st.markdown('<div class="chart-card"><div class="chart-title">⭐ 4. Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    fi_sorted = fi.sort_values(ascending=True)
    fi_pct    = (fi_sorted / fi_sorted.sum() * 100).round(2)
    n = len(fi_pct)
    bar_colors_fi = [f"hsl({int(i * 120 / max(n-1,1))}, 80%, 55%)" for i in range(n)]
    fig4 = go.Figure(go.Bar(
        x=fi_pct.values, y=fi_pct.index.tolist(),
        orientation='h', marker_color=bar_colors_fi,
        text=[f"{v}%" for v in fi_pct.values], textposition='outside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
    ))
    fig4.update_layout(margin=dict(l=0,r=55,t=10,b=30), height=250,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(title="Importance (%)", showgrid=True, gridcolor='#f0f0f0'),
        yaxis=dict(title=""), font=dict(size=10))
    st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c5:
    st.markdown('<div class="chart-card"><div class="chart-title">🥧 5. Distribusi Tingkat Engagement Mahasiswa</div>', unsafe_allow_html=True)
    if 'ClusterLabel' in df_cf.columns:
        cl_counts = df_cf['ClusterLabel'].value_counts()
    else:
        cl_counts = pd.Series({'High Engaged': 0, 'Mid Engaged': 0, 'Low Engaged': 0})
    labels_c  = cl_counts.index.tolist()
    colors_c  = [CLUSTER_COLORS.get(l, "#aaa") for l in labels_c]
    fig5 = go.Figure(go.Pie(
        labels=labels_c, values=cl_counts.values,
        hole=0.45, marker_colors=colors_c,
        textinfo='percent',
        hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
    ))
    fig5.update_layout(margin=dict(l=0,r=0,t=10,b=10), height=250,
        paper_bgcolor='white',
        legend=dict(orientation='v', x=1.0, y=0.5, font=dict(size=10)),
        font=dict(size=11))
    st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c6:
    st.markdown('<div class="chart-card"><div class="chart-title">👥 6. Segmentasi Mahasiswa (K-Means = 3)</div>', unsafe_allow_html=True)
    if 'ClusterLabel' in df_cf.columns:
        cl_vc = df_cf['ClusterLabel'].value_counts()
    else:
        cl_vc = pd.Series({'High Engaged': 0, 'Mid Engaged': 0, 'Low Engaged': 0})
    order      = ['High Engaged', 'Mid Engaged', 'Low Engaged']
    cl_data    = {k: int(cl_vc.get(k, 0)) for k in order}
    total_cl   = sum(cl_data.values())
    max_count  = max(cl_data.values()) if max(cl_data.values()) > 0 else 1
    badge_hex  = {"High Engaged": "#6366f1", "Mid Engaged": "#22d3ee", "Low Engaged": "#a855f7"}

    rows_html = ""
    for label, count in cl_data.items():
        pct   = round(count / total_cl * 100, 1) if total_cl > 0 else 0
        bar_w = int(count / max_count * 100)
        color = badge_hex[label]
        rows_html += f"""
        <tr>
          <td style="white-space:nowrap">
            <span class="seg-badge" style="background:{color}22;color:{color}">
              &#x1F465; {label}
            </span>
          </td>
          <td style="width:45%">
            <div style="background:#f0f0f0;border-radius:6px;height:14px">
              <div style="background:{color};width:{bar_w}%;height:14px;border-radius:6px"></div>
            </div>
          </td>
          <td style="text-align:right;font-weight:600;padding-left:8px">{count}</td>
          <td style="text-align:right;color:#888;padding-left:4px">{pct}%</td>
        </tr>"""

    rows_html += f"""
    <tr style="border-top:1px solid #e0e0e0;font-weight:600">
      <td colspan="2" style="padding-top:6px">Total</td>
      <td style="text-align:right;padding-top:6px">{total_cl:,}</td>
      <td style="text-align:right;color:#888;padding-top:6px">100%</td>
    </tr>"""

    st.markdown(f"""
    <table class="seg-table">
      <thead>
        <tr>
          <th>Cluster</th><th></th>
          <th style="text-align:right">Jumlah Mahasiswa</th>
          <th style="text-align:right">Persentase</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ─── ROW 3: Charts 7 & 8 ─────────────────────────────────────────────────────
c7, c8 = st.columns([1, 1.6])

with c7:
    st.markdown('<div class="chart-card"><div class="chart-title">📊 7. Perbandingan Performa Model</div>', unsafe_allow_html=True)
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    dt_vals = [metrics["DT"][m] for m in metric_names]
    rf_vals = [metrics["RF"][m] for m in metric_names]
    fig7 = go.Figure()
    fig7.add_trace(go.Bar(
        name="Decision Tree", x=metric_names, y=dt_vals,
        marker_color="#93c5fd",
        text=[f"{v}" for v in dt_vals], textposition='outside',
        hovertemplate='DT %{x}: %{y}%<extra></extra>'
    ))
    fig7.add_trace(go.Bar(
        name="Random Forest (SMOTE)", x=metric_names, y=rf_vals,
        marker_color="#2dd4bf",
        text=[f"{v}" for v in rf_vals], textposition='outside',
        hovertemplate='RF %{x}: %{y}%<extra></extra>'
    ))
    fig7.update_layout(
        barmode='group',
        margin=dict(l=0,r=0,t=10,b=30), height=280,
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(title="Skor (%)", range=[0,115], showgrid=True, gridcolor='#f0f0f0'),
        legend=dict(orientation='h', x=0, y=1.15, font=dict(size=10)),
        font=dict(size=11))
    st.plotly_chart(fig7, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c8:
    st.markdown('<div class="chart-card"><div class="chart-title">🎯 8. Prediksi Rekomendasi SKS (Input Mahasiswa)</div>', unsafe_allow_html=True)

    pi1, pi2 = st.columns(2)
    with pi1:
        gpa_in   = st.slider("GPA", 2.0, 4.0, 3.25, 0.01, key="p_gpa")
        abs_in   = st.slider("Absences", 0, 93, 10, key="p_abs")
        eng_in   = st.slider("Engagement Score", 17.92, 97.03, 65.25, 0.01, key="p_eng")
    with pi2:
        study_in = st.slider("Study Time Weekly (jam)", 0, 40, 15, key="p_study")
        perf_in  = st.slider("Performance Score", 0, 100, 72, key="p_perf")

    sample_input = pd.DataFrame([[
        gpa_in, float(study_in), abs_in, float(perf_in),
        40, 50, 30, 20, 10, eng_in
    ]], columns=feature_cols)

    pred_class  = rf_model.predict(sample_input)[0]
    pred_proba  = rf_model.predict_proba(sample_input)[0]
    classes     = rf_model.classes_

    # — Hanya probability bars, tanpa kartu kuning —
    st.markdown(f"<div style='font-size:0.82rem;font-weight:700;margin:8px 0 4px'>Rekomendasi: <span style='color:#d97706;font-size:1.1rem'>{pred_class} SKS</span></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;font-weight:600;margin-bottom:6px;color:#555'>Probabilitas Tiap Kelas</div>", unsafe_allow_html=True)

    bar_colors_pred = ["#a855f7", "#f97316", "#22d3ee", "#6366f1", "#10b981"]
    for i, (cls, prob) in enumerate(zip(classes, pred_proba)):
        pct   = round(prob * 100, 2)
        color = bar_colors_pred[i % len(bar_colors_pred)]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <div style="width:52px;font-size:0.78rem;font-weight:600;color:{color}">{cls} SKS</div>
          <div style="flex:1;background:#f0f0f0;border-radius:6px;height:16px">
            <div style="background:{color};width:{min(pct,100)}%;height:16px;border-radius:6px"></div>
          </div>
          <div style="width:52px;font-size:0.78rem;text-align:right;font-weight:600">{pct}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""<div style="margin-top:10px;font-size:0.78rem;color:#4ade80;background:#f0fdf4;
        border-radius:8px;padding:8px 10px;">
        ✅ Mahasiswa dengan profil ini direkomendasikan mengambil <b>{pred_class} SKS</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
