import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from model import train_models, predict_crop_yield, evaluate_field_validation
from utils import (
    clean_dataset,
    get_crop_status,
    get_recommendation,
    get_harvest_progress,
    get_health_ratio,
    build_data_summary,
    plot_growth_chart,
    plot_yield_distribution,
    plot_correlation_heatmap,
    feature_distribution,
    load_css,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "agroforecast_dataset.csv"
FIELD_DATA_PATH = BASE_DIR / "Dataset" / "agroforecast_field_validation.csv"
CSS_PATH = BASE_DIR / "style.css"

st.set_page_config(
    page_title="AgroForecast",
    page_icon="🌿",
    layout="wide",
)

load_css(CSS_PATH)


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def build_model_pipeline(df):
    cleaned = clean_dataset(df)
    return train_models(cleaned)


@st.cache_data(show_spinner=False)
def load_validation_data():
    if FIELD_DATA_PATH.exists():
        return pd.read_csv(FIELD_DATA_PATH)
    return None


def render_sidebar():
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color:white; margin-bottom:0;'>AgroForecast</h1>
            <p style='color:#d8f3dc; margin-top:0.2rem; font-size:0.95rem;'>AI-based Harvest Prediction System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Dashboard",
        ["Dashboard", "Prediksi", "Dataset"],
        index=0,
    )
    return menu


def render_header(summary):
    st.markdown(
        """
        <div class='header-panel'>
            <h1>AgroForecast</h1>
            <p>AI Smart Farming untuk monitoring, analisis, dan prediksi panen Cabai, Tomat, dan Terong.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-card-title'>Umur Tanaman Median</div>
            <div class='metric-card-value'>{summary['avg_age']} Hari</div>
            <div class='metric-card-subtitle'>Data sampel dari dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-card-title'>Tinggi Rata-rata</div>
            <div class='metric-card-value'>{summary['avg_height']} cm</div>
            <div class='metric-card-subtitle'>Trend pertumbuhan</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-card-title'>Estimasi Hasil Panen</div>
            <div class='metric-card-value'>{summary['avg_yield']} kg</div>
            <div class='metric-card-subtitle'>Per tanaman</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status = "Optimal" if summary["avg_yield"] > 2.0 else "Stabil"
    col4.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-card-title'>Status Kebun</div>
            <div class='metric-card-value'>{status}</div>
            <div class='metric-card-subtitle'>Analisis AI sederhana</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_dashboard(df):
    st.title("Dashboard Modern")
    st.write("Ringkasan AI-based Harvest Prediction System untuk data pertumbuhan tanaman.")

    summary = build_data_summary(df)
    render_header(summary)

    with st.container():
        st.subheader("Pertumbuhan dan Kinerja Tanaman")
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(plot_growth_chart(df), use_container_width=True)
        with right:
            st.markdown(
                """
                <div class='glass-panel'>
                    <h3>Status Varietas</h3>
                    <p>Cabai, Tomat, dan Terong dipantau secara real time berdasarkan kondisi lingkungan.</p>
                    <ul style='padding-left: 1rem; color:#2c5d3b;'>
                        <li>Analisis kualitas air dan tanah</li>
                        <li>Rata-rata hasil panen</li>
                        <li>Estimasi fase panen</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("---")

    with st.container():
        st.subheader("Visualisasi Data")
        tab1, tab2 = st.tabs(["Distribusi Hasil Panen", "Korelasi Fitur"])
        with tab1:
            st.plotly_chart(plot_yield_distribution(df), use_container_width=True)
        with tab2:
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)


def display_dataset_page(df):
    st.title("Dataset & Data Cleaning")
    st.write("Preview dataset lokal untuk analisis dan pelatihan.")

    st.markdown("**Ringkasan dataset**")
    st.dataframe(df.head(12), use_container_width=True)
    st.markdown("---")

    missing = df.isna().sum()
    st.write("### Pemeriksaan Missing Value")
    st.write(missing.to_frame("Jumlah Missing").sort_values("Jumlah Missing", ascending=False))

    cleaned = clean_dataset(df)
    st.markdown("---")
    st.subheader("Dataset Setelah Pembersihan")
    st.dataframe(cleaned.head(12), use_container_width=True)
    st.markdown("---")
    st.write("Dataset dilengkapi dengan imputasi sederhana untuk nilai numerik dan mode untuk kolom tanaman.")


def display_prediction_page(df):
    st.title("Prediksi Hasil Panen")
    st.write("Masukkan kondisi tanaman dan dapatkan estimasi panen, waktu, dan rekomendasi AI.")

    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            tanaman = st.selectbox("Jenis Tanaman", ["Cabai", "Tomat", "Terong"])
            umur = st.slider("Umur Tanaman (hari)", 10, 100, 40)
            tinggi = st.number_input("Tinggi Tanaman (cm)", min_value=10, max_value=130, value=45)
            daun = st.number_input("Jumlah Daun", min_value=4, max_value=60, value=20)
        with col2:
            kelembaban = st.slider("Kelembaban (%)", 30, 100, 70)
            suhu = st.slider("Suhu (°C)", 16, 40, 27)
            curah_hujan = st.slider("Curah Hujan (mm)", 0, 40, 16)
            ph = st.slider("pH Tanah", 4.5, 8.0, 6.3, format="%.1f")

        submit = st.form_submit_button("Hitung Prediksi")

    if submit:
        pipeline = build_model_pipeline(df)
        input_payload = {
            "tanaman": tanaman,
            "umur_hari": umur,
            "tinggi_cm": tinggi,
            "jumlah_daun": daun,
            "kelembaban": kelembaban,
            "suhu": suhu,
            "curah_hujan": curah_hujan,
            "ph_tanah": ph,
        }
        predictions = predict_crop_yield(
            pipeline["models"],
            pipeline["feature_columns"],
            input_payload,
        )
        average_prediction = round(np.mean(list(predictions.values())), 2)
        status_color, status_label = get_crop_status(kelembaban, suhu, ph, curah_hujan)
        recommendation = get_recommendation(kelembaban, suhu, curah_hujan, ph)
        health_score = get_health_ratio(kelembaban, suhu, ph, curah_hujan)
        harvest_progress = get_harvest_progress(umur, tinggi, tanaman)
        days_to_harvest = max(3, int({"Cabai": 80, "Tomat": 90, "Terong": 100}[tanaman] - umur))
        status_map = {"Hijau": "green", "Kuning": "yellow", "Merah": "red"}
        status_style = status_map.get(status_color, "green")

        st.markdown(
            "<div class='glass-panel'>",
            unsafe_allow_html=True,
        )
        st.subheader("Hasil Prediksi")
        st.write(f"Jenis Tanaman: **{tanaman}**")
        st.metric("Estimasi Hasil Panen (kg)", f"{average_prediction} kg")
        st.write(f"Prediksi Linear Regression: **{round(predictions['Linear Regression'], 2)} kg**")
        st.write(f"Prediksi Random Forest: **{round(predictions['Random Forest'], 2)} kg**")
        st.write(f"Estimasi waktu panen: **{days_to_harvest} hari lagi**")
        st.markdown(
            f"<div class='status-chip status-{status_style}'>⚡ {status_label}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='ai-insight'>", unsafe_allow_html=True)
        st.write("### AI Insight")
        st.write(
            "Sistem memproses kondisi lingkungan dan rekomendasi perawatan secara bersamaan untuk meningkatkan hasil panen."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        health_col, harvest_col = st.columns(2)
        with health_col:
            st.markdown("<div class='progress-label'>Kesehatan Tanaman</div>", unsafe_allow_html=True)
            st.progress(int(health_score))
            st.write(f"Skor kesehatan: **{health_score}%**")
        with harvest_col:
            st.markdown("<div class='progress-label'>Kemajuan Menuju Panen</div>", unsafe_allow_html=True)
            st.progress(int(harvest_progress))
            st.write(f"Kemajuan: **{harvest_progress}%**")

        st.markdown("---")
        st.write("### Rekomendasi Perawatan")
        st.info(recommendation)


def main():
    menu = render_sidebar()
    df = load_data()

    if df is None or df.empty:
        st.warning("Tidak ada dataset CSV tersedia. Pastikan file Dataset/agroforecast_dataset.csv tersedia.")
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            df = pd.DataFrame()

    df = clean_dataset(df)

    if menu == "Dashboard":
        display_dashboard(df)
    elif menu == "Dataset":
        display_dataset_page(df)
    elif menu == "Prediksi":
        display_prediction_page(df)

    st.markdown("---")
    st.write("**AgroForecast** — Aplikasi AI Smart Farming untuk monitoring, prediksi, dan visualisasi hasil panen.")


if __name__ == "__main__":
    main()
