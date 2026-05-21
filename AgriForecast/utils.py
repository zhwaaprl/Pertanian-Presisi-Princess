import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def load_css(path):
    try:
        with open(path, "r", encoding="utf-8") as css_file:
            css = css_file.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css tidak ditemukan. Pastikan file ada di folder proyek.")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hasil_panen_kg" in df.columns:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    if "tanaman" in df.columns:
        df["tanaman"] = df["tanaman"].fillna(df["tanaman"].mode().iloc[0])
    return df.dropna(how="all")


def get_crop_status(kelembaban, suhu, ph, curah_hujan):
    conditions = 0
    conditions += 1 if 55 <= kelembaban <= 80 else 0
    conditions += 1 if 22 <= suhu <= 31 else 0
    conditions += 1 if 5.8 <= ph <= 6.8 else 0
    conditions += 1 if 8 <= curah_hujan <= 25 else 0

    if conditions >= 3:
        return "Hijau", "Sehat"
    if conditions == 2:
        return "Kuning", "Cukup"
    return "Merah", "Perlu perhatian"


def get_recommendation(kelembaban, suhu, curah_hujan, ph):
    tips = []
    if kelembaban < 55:
        tips.append("Tanaman membutuhkan lebih banyak air")
    elif kelembaban > 80:
        tips.append("Suhu lembab, pastikan drainase bekerja baik")
    if suhu < 24:
        tips.append("Suhu rendah, hindari angin dingin")
    elif suhu > 30:
        tips.append("Suhu tinggi, pertimbangkan naungan parsial")
    if curah_hujan < 10:
        tips.append("Tambahkan penyiraman teratur")
    if ph < 5.8:
        tips.append("pH tanah asam, lakukan penyesuaian kapur")
    if ph > 6.8:
        tips.append("pH tanah basa, tambahkan bahan organik")
    if not tips:
        return "Pertumbuhan optimal, lanjutkan perawatan rutin."
    return "; ".join(tips)


def get_harvest_progress(umur_hari, tinggi_cm, tanaman):
    target_age = {"Cabai": 80, "Tomat": 90, "Terong": 100}
    target_height = {"Cabai": 90, "Tomat": 110, "Terong": 105}
    age_ratio = min(1.0, umur_hari / target_age.get(tanaman, 90))
    height_ratio = min(1.0, tinggi_cm / target_height.get(tanaman, 100))
    progress = round((age_ratio * 0.55 + height_ratio * 0.45) * 100)
    return max(0, min(100, progress))


def get_health_ratio(kelembaban, suhu, ph, curah_hujan):
    score = 0
    score += 1 if 55 <= kelembaban <= 80 else 0
    score += 1 if 22 <= suhu <= 31 else 0
    score += 1 if 5.8 <= ph <= 6.8 else 0
    score += 1 if 8 <= curah_hujan <= 25 else 0
    return int((score / 4) * 100)


def build_data_summary(df: pd.DataFrame) -> dict:
    sample = df.copy()
    return {
        "avg_age": int(sample["umur_hari"].median()),
        "avg_height": int(sample["tinggi_cm"].median()),
        "avg_yield": round(sample["hasil_panen_kg"].mean(), 2),
        "crop_count": sample["tanaman"].value_counts().to_dict(),
    }


def plot_growth_chart(df: pd.DataFrame, crop: str = None):
    sample = df.copy()
    if crop and crop != "Semua":
        sample = sample[sample["tanaman"] == crop]
    sample = sample.sort_values("umur_hari")
    fig = px.line(
        sample,
        x="umur_hari",
        y="tinggi_cm",
        color="tanaman",
        markers=True,
        template="plotly_white",
        title="Grafik Pertumbuhan Tinggi Tanaman"
    )
    fig.update_layout(plot_bgcolor="rgba(255,255,255,0.9)")
    return fig


def plot_yield_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="hasil_panen_kg",
        color="tanaman",
        nbins=18,
        template="plotly_white",
        title="Distribusi Hasil Panen (kg)"
    )
    fig.update_layout(plot_bgcolor="rgba(255,255,255,0.9)")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"])
    correlation = numeric.corr()
    fig = px.imshow(
        correlation,
        text_auto=True,
        color_continuous_scale="Greens",
        title="Heatmap Korelasi Fitur Pertumbuhan",
    )
    fig.update_layout(plot_bgcolor="rgba(255,255,255,0.9)")
    return fig


def feature_distribution(df: pd.DataFrame, column: str):
    fig = px.violin(
        df,
        y=column,
        color="tanaman",
        box=True,
        points="all",
        template="plotly_white",
        title=f"Distribusi {column.replace('_', ' ').title()}"
    )
    fig.update_layout(plot_bgcolor="rgba(255,255,255,0.9)")
    return fig
