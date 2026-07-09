from __future__ import annotations

import json
import os
import zipfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# set konfigurasi
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
RESOURCE_DIR = BASE_DIR / "resource"
OUTPUT_DIR = BASE_DIR / "output"
CONFIG_PATH = BASE_DIR / "config" / "dashboard_config.json"
OUTPUT_DIR.mkdir(exist_ok=True)

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
ECON_COLS = ["Fuel_Price", "CPI", "Unemployment"]
TARGET_COL = "Weekly_Sales"
REQUIRED_TEST_COLS = ["Store", "Dept", "Date", "IsHoliday"]
REQUIRED_FEATURE_COLS = ["Store", "Date", "Temperature", "Fuel_Price", "CPI", "Unemployment", *MARKDOWN_COLS]
REQUIRED_STORE_COLS = ["Store", "Type", "Size"]
REQUIRED_HISTORY_COLS = ["Store", "Dept", "Date", "Weekly_Sales"]
RESIDUAL_CANDIDATES = ["Residual_XGB_Tuning", "Residual_XGB", "Residual", "residual"]
TYPE_MAPPING = {"A": 0, "B": 1, "C": 2}

DEFAULT_XGB_PATH = MODEL_DIR / "hybrid_xgboost_lstm_model_1_xgboost_20260626_040552.pkl"
DEFAULT_LSTM_PATH = MODEL_DIR / "tuning_xgboost_lstm_model_lstm_residual_20260628_072659.keras"
DEFAULT_ARTIFACT_PATH = RESOURCE_DIR / "artefak_tuning_xgboost_lstm_20260628_072659.pkl"


# set tampilan css agar rapi
st.markdown(
    """
    <style>
    :root {
        --primary: #0F172A;
        --secondary: #10B981;
        --tertiary: #3B82F6;
        --neutral: #F8FAFC;
        --surface: #FFFFFF;
        --muted: #64748B;
        --border: #E2E8F0;
        --soft-green: #ECFDF5;
        --soft-blue: #EFF6FF;
        --danger: #EF4444;
        --warning: #F59E0B;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--neutral);
        font-family: Inter, "Segoe UI", sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.3rem;
        padding-bottom: 3rem;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 22px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-logo {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 45%, #10B981 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 900;
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.18);
    }

    .brand-title {
        font-size: 18px;
        font-weight: 900;
        color: var(--primary);
        letter-spacing: -0.4px;
    }

    .brand-subtitle {
        font-size: 12px;
        color: var(--muted);
        margin-top: 2px;
    }

    .top-actions {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--muted);
        font-size: 13px;
    }

    .hero {
        background: linear-gradient(135deg, #0F172A 0%, #14213D 52%, #0B3B36 100%);
        border-radius: 24px;
        padding: 34px 36px;
        color: white;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
        margin-bottom: 20px;
    }

    .hero:after {
        content: "";
        position: absolute;
        right: -120px;
        top: -120px;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: rgba(16, 185, 129, 0.18);
    }

    .hero-label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.16);
        border: 1px solid rgba(16, 185, 129, 0.30);
        color: #BBF7D0;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 38px;
        line-height: 1.08;
        font-weight: 950;
        letter-spacing: -1.1px;
        margin-bottom: 10px;
        max-width: 720px;
    }

    .hero-desc {
        font-size: 14px;
        line-height: 1.65;
        color: #CBD5E1;
        max-width: 790px;
    }

    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 16px 38px rgba(15, 23, 42, 0.055);
    }

    .card-title {
        font-size: 18px;
        font-weight: 900;
        color: var(--primary);
        letter-spacing: -0.3px;
        margin-bottom: 4px;
    }

    .card-subtitle {
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 18px;
        line-height: 1.55;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px 20px 18px 20px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.045);
        min-height: 128px;
    }

    .metric-label {
        font-size: 11px;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 900;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 31px;
        line-height: 1.1;
        color: var(--primary);
        font-weight: 950;
        letter-spacing: -0.8px;
        margin-bottom: 8px;
    }

    .metric-note {
        font-size: 12px;
        color: #047857;
        font-weight: 700;
    }

    .validation-box {
        border-radius: 16px;
        border: 1px solid var(--border);
        padding: 16px 16px;
        background: #FFFFFF;
        min-height: 108px;
    }

    .valid-name {
        color: var(--primary);
        font-weight: 900;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .valid-ok {
        display: inline-flex;
        padding: 6px 9px;
        border-radius: 999px;
        background: var(--soft-green);
        color: #047857;
        font-size: 11px;
        font-weight: 900;
        border: 1px solid #BBF7D0;
        margin-bottom: 8px;
    }

    .valid-bad {
        display: inline-flex;
        padding: 6px 9px;
        border-radius: 999px;
        background: #FEF2F2;
        color: #B91C1C;
        font-size: 11px;
        font-weight: 900;
        border: 1px solid #FECACA;
        margin-bottom: 8px;
    }

    .valid-note {
        font-size: 11px;
        color: var(--muted);
        line-height: 1.45;
    }

    .stFileUploader > div {
        border: 1px dashed #94A3B8 !important;
        border-radius: 18px !important;
        background: #FFFFFF !important;
        padding: 22px !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid #0F172A;
        background: #0F172A;
        color: white;
        padding: 0.85rem 1.1rem;
        font-weight: 900;
        letter-spacing: -0.1px;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.18);
        transition: 0.18s ease;
    }

    .stButton > button:hover {
        background: #111827;
        color: white;
        border-color: #111827;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid #10B981;
        background: #10B981;
        color: white;
        padding: 0.78rem 1rem;
        font-weight: 900;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    .small-note {
        font-size: 12px;
        color: var(--muted);
        line-height: 1.6;
    }

    .divider-space {
        height: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# konfigurasi path 
def safe_json_load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


CONFIG = safe_json_load(CONFIG_PATH)


def resolve_path(config_key: str, default_path: Path) -> Path:
    raw_path = CONFIG.get(config_key)
    if raw_path is None or str(raw_path).strip() == "":
        return default_path

    path = Path(str(raw_path))
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


XGB_PATH = resolve_path("model_xgboost_path", DEFAULT_XGB_PATH)
LSTM_PATH = resolve_path("model_lstm_residual_path", DEFAULT_LSTM_PATH)
ARTIFACT_PATH = resolve_path("artifact_path", DEFAULT_ARTIFACT_PATH)


@st.cache_resource(show_spinner=False)
def load_pickle_resource(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None, f"File tidak ditemukan: {path.name}"
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"File tidak dapat dibaca: {path.name}. Detail: {exc}"


def get_keras_saved_version(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        return str(metadata.get("keras_version", "tidak diketahui"))
    except Exception:
        return "tidak diketahui"


@st.cache_resource(show_spinner=False)
def load_keras_resource(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None, f"File tidak ditemukan: {path.name}"

    saved_version = get_keras_saved_version(path)
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    try:
        import keras  # type: ignore

        try:
            model = keras.saving.load_model(path, compile=False, safe_mode=False)
        except AttributeError:
            model = keras.models.load_model(path, compile=False, safe_mode=False)
        return model, None
    except Exception as exc_keras:
        try:
            from tensorflow import keras as tf_keras  # type: ignore

            model = tf_keras.models.load_model(path, compile=False)
            return model, None
        except Exception as exc_tf:
            message = (
                f"Model LSTM residual tidak dapat dimuat: {path.name}. "
                f"File terdeteksi disimpan dengan Keras {saved_version}. "
                "Gunakan TensorFlow >= 2.16 dan Keras >= 3.0. "
                f"Detail: {exc_keras} | fallback TensorFlow: {exc_tf}"
            )
            return None, message


# mengatur data 
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    rename_map = {
        "Is_Holiday": "IsHoliday",
        "isHoliday": "IsHoliday",
        "isholiday": "IsHoliday",
        "date": "Date",
        "store": "Store",
        "dept": "Dept",
        "weekly_sales": "Weekly_Sales",
    }
    valid_map = {old: new for old, new in rename_map.items() if old in df.columns}
    df = df.rename(columns=valid_map)
    return df


def parse_date(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def missing_columns(df: Optional[pd.DataFrame], required_cols: Iterable[str]) -> list[str]:
    if df is None:
        return list(required_cols)
    return [col for col in required_cols if col not in df.columns]


def read_csv_bytes(content: bytes) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(BytesIO(content))
    except Exception:
        return None


def identify_dataset(filename: str, df: pd.DataFrame) -> Optional[str]:
    df = standardize_columns(df)
    name = filename.lower()
    cols = set(df.columns)

    if "residual" in name:
        return "residual_history"
    if "features" in name:
        return "features"
    if "stores" in name:
        return "stores"
    if "train" in name or "histori" in name or "history" in name:
        return "history"
    if "test" in name:
        return "test"
    if "evaluasi" in name or "evaluation" in name:
        return "evaluasi"

    if {"Store", "Type", "Size"}.issubset(cols):
        return "stores"
    if {"Store", "Date", "Temperature"}.issubset(cols):
        return "features"
    if {"Store", "Dept", "Date", "Weekly_Sales"}.issubset(cols):
        return "history"
    if {"Store", "Dept", "Date", "IsHoliday"}.issubset(cols) and "Weekly_Sales" not in cols:
        return "test"

    return None


def read_uploaded_files(uploaded_files) -> dict[str, Optional[pd.DataFrame]]:
    data: dict[str, Optional[pd.DataFrame]] = {
        "test": None,
        "features": None,
        "stores": None,
        "history": None,
        "residual_history": None,
        "evaluasi": None,
    }

    if uploaded_files is None or len(uploaded_files) == 0:
        return data

    for uploaded in uploaded_files:
        filename = uploaded.name
        raw = uploaded.getvalue()

        if filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(BytesIO(raw)) as zf:
                    for member in zf.namelist():
                        if not member.lower().endswith(".csv"):
                            continue
                        with zf.open(member) as file_obj:
                            df = pd.read_csv(file_obj)
                        key = identify_dataset(member, df)
                        if key is not None:
                            data[key] = standardize_columns(df)
            except Exception as exc:
                st.error(f"ZIP tidak dapat dibaca: {filename}. Detail: {exc}")
        else:
            df = read_csv_bytes(raw)
            if df is not None:
                key = identify_dataset(filename, df)
                if key is not None:
                    data[key] = standardize_columns(df)

    return data


@st.cache_data(show_spinner=False)
def load_csv_from_path(path_str: str) -> Optional[pd.DataFrame]:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return standardize_columns(pd.read_csv(path))
    except Exception:
        return None


def load_data_from_folder() -> dict[str, Optional[pd.DataFrame]]:
    history_df = load_csv_from_path(str(DATA_DIR / "train.csv"))
    if history_df is None:
        history_df = load_csv_from_path(str(DATA_DIR / "data_histori_lag.csv"))

    evaluasi_df = load_csv_from_path(str(DATA_DIR / "evaluasi_model.csv"))
    if evaluasi_df is None:
        evaluasi_df = load_csv_from_path(str(DATA_DIR / "hasil_evaluasi_model.csv"))

    return {
        "test": load_csv_from_path(str(DATA_DIR / "test.csv")),
        "features": load_csv_from_path(str(DATA_DIR / "features.csv")),
        "stores": load_csv_from_path(str(DATA_DIR / "stores.csv")),
        "history": history_df,
        "residual_history": load_csv_from_path(str(DATA_DIR / "residual_history.csv")),
        "evaluasi": evaluasi_df,
    }


def merge_data_sources(uploaded_data: dict[str, Optional[pd.DataFrame]], folder_data: dict[str, Optional[pd.DataFrame]]) -> dict[str, Optional[pd.DataFrame]]:
    merged: dict[str, Optional[pd.DataFrame]] = {}
    for key in ["test", "features", "stores", "history", "residual_history", "evaluasi"]:
        uploaded_df = uploaded_data.get(key)
        folder_df = folder_data.get(key)
        if uploaded_df is not None:
            merged[key] = uploaded_df
        else:
            merged[key] = folder_df
    return merged

# feat-Eng
def encode_type(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    encoded = series.astype(str).str.strip().str.upper().map(TYPE_MAPPING)
    return encoded.fillna(0).astype(int)


def encode_holiday(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "1": 1,
        "yes": 1,
        "y": 1,
        "holiday": 1,
        "false": 0,
        "0": 0,
        "no": 0,
        "n": 0,
    }
    mapped = text.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric).fillna(0).astype(int)


def fill_by_store(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Store" in df.columns:
            df[col] = df.groupby("Store")[col].transform(lambda s: s.ffill().bfill())
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0
        df[col] = df[col].fillna(median_value)
    return df


def integrate_test_data(test_df: pd.DataFrame, features_df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    test_df = parse_date(standardize_columns(test_df))
    features_df = parse_date(standardize_columns(features_df))
    stores_df = standardize_columns(stores_df)

    features_merge = features_df.drop(columns=["IsHoliday", "Is_Holiday"], errors="ignore")
    df = test_df.merge(features_merge, on=["Store", "Date"], how="left")
    df = df.merge(stores_df, on="Store", how="left")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_date(standardize_columns(df))
    df = df.copy()

    df["Store_Asli"] = df["Store"]
    df["Dept_Asli"] = df["Dept"]
    df["Date_Asli"] = df["Date"]
    df["IsHoliday_Asli"] = df["IsHoliday"]
    if "Type" in df.columns:
        df["Type_Asli"] = df["Type"]
    else:
        df["Type_Asli"] = "Tidak diketahui"
        df["Type"] = "A"

    for col in MARKDOWN_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = fill_by_store(df, ["Temperature", "Size", *ECON_COLS])

    df["Total_Markdown"] = df[MARKDOWN_COLS].sum(axis=1)
    df["Log_Total_Markdown"] = np.log1p(df["Total_Markdown"].clip(lower=0))
    df["IsPromo"] = (df["Total_Markdown"] > 0).astype(int)
    df["Markdown_Active_Count"] = (df[MARKDOWN_COLS] > 0).sum(axis=1)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    df["Type"] = encode_type(df["Type"])
    df["IsHoliday"] = encode_holiday(df["IsHoliday"])
    df["Store"] = pd.to_numeric(df["Store"], errors="coerce").fillna(0).astype(int)
    df["Dept"] = pd.to_numeric(df["Dept"], errors="coerce").fillna(0).astype(int)

    return df


# prediksi model
def get_model_feature_columns(xgb_model) -> list[str]:
    if xgb_model is not None and hasattr(xgb_model, "feature_names_in_"):
        return [str(col) for col in list(xgb_model.feature_names_in_)]

    return [
        "Store", "Dept", "Type", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "IsHoliday", "Log_Total_Markdown", "IsPromo", "Markdown_Active_Count",
        "Year", "Month", "WeekOfYear", "Quarter", "Sales_Lag1", "Sales_Lag4",
        "Rolling_Mean4", "Rolling_Mean8",
    ]


def clean_history(history_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if history_df is None:
        return None

    history_df = parse_date(standardize_columns(history_df))
    if TARGET_COL not in history_df.columns:
        return None

    needed = ["Store", "Dept", "Date", TARGET_COL]
    if any(col not in history_df.columns for col in needed):
        return None

    history_df = history_df[needed].copy()
    history_df["Store"] = pd.to_numeric(history_df["Store"], errors="coerce")
    history_df["Dept"] = pd.to_numeric(history_df["Dept"], errors="coerce")
    history_df[TARGET_COL] = pd.to_numeric(history_df[TARGET_COL], errors="coerce")
    history_df = history_df.dropna(subset=["Store", "Dept", "Date", TARGET_COL])
    history_df = history_df[history_df[TARGET_COL] >= 0]
    history_df = history_df.sort_values(["Store", "Dept", "Date"])
    return history_df


def build_sales_history(history_df: Optional[pd.DataFrame]) -> Tuple[Dict[Tuple[int, int], list[float]], float]:
    history_dict: Dict[Tuple[int, int], list[float]] = {}

    cleaned = clean_history(history_df)
    if cleaned is None or cleaned.empty:
        return history_dict, 0.0

    global_median = float(cleaned[TARGET_COL].median())
    for (store, dept), group in cleaned.groupby(["Store", "Dept"]):
        key = (int(store), int(dept))
        values = group.sort_values("Date")[TARGET_COL].astype(float).tolist()
        history_dict[key] = values

    return history_dict, global_median


def calculate_lags(values: list[float], fallback: float) -> Tuple[float, float, float, float]:
    if len(values) > 0:
        lag1 = values[-1]
        rolling4 = float(np.mean(values[-4:]))
        rolling8 = float(np.mean(values[-8:]))
    else:
        lag1 = fallback
        rolling4 = fallback
        rolling8 = fallback

    if len(values) >= 4:
        lag4 = values[-4]
    else:
        lag4 = fallback

    return float(lag1), float(lag4), float(rolling4), float(rolling8)


def prepare_x(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = df.copy()

    missing = [col for col in feature_columns if col not in x.columns]
    if len(missing) > 0:
        raise ValueError(f"Fitur belum tersedia untuk model: {missing}")

    x = x[feature_columns].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        median_value = x[col].median()
        if pd.isna(median_value):
            median_value = 0
        x[col] = x[col].fillna(median_value)

    return x


def predict_xgb_recursive_batched(
    df: pd.DataFrame,
    xgb_model,
    feature_columns: list[str],
    history_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    df = df.copy().sort_values(["Date_Asli", "Store_Asli", "Dept_Asli"]).reset_index(drop=True)
    history_dict, global_median = build_sales_history(history_df)

    predictions = pd.Series(index=df.index, dtype="float64")
    unique_dates = sorted(df["Date_Asli"].dropna().unique())

    progress = st.progress(0, text="Menjalankan prediksi XGBoost...")
    total_dates = max(len(unique_dates), 1)

    for pos, current_date in enumerate(unique_dates, start=1):
        idxs = df.index[df["Date_Asli"] == current_date].tolist()

        for idx in idxs:
            store = int(df.at[idx, "Store_Asli"])
            dept = int(df.at[idx, "Dept_Asli"])
            key = (store, dept)
            sales_history = history_dict.get(key, [])
            lag1, lag4, roll4, roll8 = calculate_lags(sales_history, global_median)
            df.at[idx, "Sales_Lag1"] = lag1
            df.at[idx, "Sales_Lag4"] = lag4
            df.at[idx, "Rolling_Mean4"] = roll4
            df.at[idx, "Rolling_Mean8"] = roll8

        x_batch = prepare_x(df.loc[idxs], feature_columns)
        pred_batch = np.asarray(xgb_model.predict(x_batch)).reshape(-1).astype(float)
        predictions.loc[idxs] = pred_batch

        for idx, pred_value in zip(idxs, pred_batch):
            store = int(df.at[idx, "Store_Asli"])
            dept = int(df.at[idx, "Dept_Asli"])
            key = (store, dept)
            history_dict.setdefault(key, []).append(float(pred_value))

        progress.progress(pos / total_dates, text=f"Prediksi XGBoost berjalan... {pos}/{total_dates} periode")

    progress.empty()
    df["Prediksi_XGBoost"] = predictions.fillna(0).values
    return df


def get_residual_scaler(artifact: Optional[dict]):
    if not isinstance(artifact, dict):
        return None
    if "scaler_residual" in artifact:
        return artifact["scaler_residual"]
    if "scaler_residual_xgb_lstm" in artifact:
        return artifact["scaler_residual_xgb_lstm"]
    return None


def get_window_size(artifact: Optional[dict], default: int = 12) -> int:
    if not isinstance(artifact, dict):
        return default

    for key in ["window_size", "window_size_lstm", "window_size_residual"]:
        if key in artifact:
            try:
                return int(artifact[key])
            except Exception:
                pass

    params = artifact.get("best_params_lstm_residual")
    if isinstance(params, dict) and "window_size" in params:
        try:
            return int(params["window_size"])
        except Exception:
            pass

    return default


def detect_residual_column(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None:
        return None
    df = standardize_columns(df)
    for col in RESIDUAL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def build_residual_history(residual_df: Optional[pd.DataFrame], residual_col: Optional[str]) -> Dict[Tuple[int, int], list[float]]:
    history: Dict[Tuple[int, int], list[float]] = {}

    if residual_df is None or residual_col is None:
        return history

    residual_df = parse_date(standardize_columns(residual_df))
    needed = ["Store", "Dept", "Date", residual_col]
    if any(col not in residual_df.columns for col in needed):
        return history

    residual_df = residual_df[needed].copy()
    residual_df["Store"] = pd.to_numeric(residual_df["Store"], errors="coerce")
    residual_df["Dept"] = pd.to_numeric(residual_df["Dept"], errors="coerce")
    residual_df[residual_col] = pd.to_numeric(residual_df[residual_col], errors="coerce")
    residual_df = residual_df.dropna(subset=["Store", "Dept", "Date", residual_col])

    for (store, dept), group in residual_df.groupby(["Store", "Dept"]):
        key = (int(store), int(dept))
        history[key] = group.sort_values("Date")[residual_col].astype(float).tolist()

    return history


def make_residual_sequence(values: list[float], scaler, window_size: int) -> np.ndarray:
    if len(values) >= window_size:
        seq = values[-window_size:]
    else:
        seq = [0.0] * (window_size - len(values)) + values

    arr = np.asarray(seq, dtype="float32").reshape(-1, 1)
    if scaler is not None:
        arr = scaler.transform(arr)
    return arr.reshape(window_size, 1)


def inverse_residual_batch(values_scaled: np.ndarray, scaler) -> np.ndarray:
    values_scaled = np.asarray(values_scaled, dtype="float32").reshape(-1, 1)
    if scaler is not None:
        return scaler.inverse_transform(values_scaled).reshape(-1)
    return values_scaled.reshape(-1)


def apply_lstm_residual_batched(
    df: pd.DataFrame,
    lstm_model,
    scaler_residual,
    residual_df: Optional[pd.DataFrame],
    residual_col: Optional[str],
    window_size: int,
) -> pd.DataFrame:
    df = df.copy().sort_values(["Date_Asli", "Store_Asli", "Dept_Asli"]).reset_index(drop=True)
    residual_history = build_residual_history(residual_df, residual_col)
    pred_residuals = pd.Series(index=df.index, dtype="float64")

    unique_dates = sorted(df["Date_Asli"].dropna().unique())
    progress = st.progress(0, text="Menjalankan koreksi residual LSTM...")
    total_dates = max(len(unique_dates), 1)

    for pos, current_date in enumerate(unique_dates, start=1):
        idxs = df.index[df["Date_Asli"] == current_date].tolist()
        seq_list = []
        key_list = []

        for idx in idxs:
            store = int(df.at[idx, "Store_Asli"])
            dept = int(df.at[idx, "Dept_Asli"])
            key = (store, dept)
            values = residual_history.get(key, [])
            seq_list.append(make_residual_sequence(values, scaler_residual, window_size))
            key_list.append(key)

        x_seq = np.asarray(seq_list, dtype="float32")
        pred_scaled = np.asarray(lstm_model.predict(x_seq, verbose=0)).reshape(-1)
        pred_residual = inverse_residual_batch(pred_scaled, scaler_residual)
        pred_residuals.loc[idxs] = pred_residual

        for key, value in zip(key_list, pred_residual):
            residual_history.setdefault(key, []).append(float(value))

        progress.progress(pos / total_dates, text=f"Koreksi residual LSTM berjalan... {pos}/{total_dates} periode")

    progress.empty()
    df["Prediksi_Residual_LSTM"] = pred_residuals.fillna(0).values
    df["Predicted_Weekly_Sales"] = df["Prediksi_XGBoost"] + df["Prediksi_Residual_LSTM"]
    df["Predicted_Weekly_Sales"] = df["Predicted_Weekly_Sales"].clip(lower=0)
    return df


def add_prediction_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["Kategori_Prediksi"] = []
        return df

    q1 = df["Predicted_Weekly_Sales"].quantile(0.33)
    q2 = df["Predicted_Weekly_Sales"].quantile(0.66)

    def category(value: float) -> str:
        if value <= q1:
            return "Rendah"
        if value <= q2:
            return "Sedang"
        return "Tinggi"

    df["Kategori_Prediksi"] = df["Predicted_Weekly_Sales"].apply(category)
    return df


def final_display_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "Store_Asli", "Dept_Asli", "Date_Asli", "Type_Asli", "Size", "IsHoliday_Asli",
        "Temperature", "Fuel_Price", "CPI", "Unemployment", "Total_Markdown",
        "Log_Total_Markdown", "IsPromo", "Sales_Lag1", "Sales_Lag4", "Rolling_Mean4",
        "Rolling_Mean8", "Prediksi_XGBoost", "Prediksi_Residual_LSTM",
        "Predicted_Weekly_Sales", "Kategori_Prediksi",
    ]
    return [col for col in candidates if col in df.columns]


def build_submission(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["Store_Asli", "Dept_Asli", "Date_Asli", "Predicted_Weekly_Sales"]].copy()
    result = result.rename(
        columns={
            "Store_Asli": "Store",
            "Dept_Asli": "Dept",
            "Date_Asli": "Date",
            "Predicted_Weekly_Sales": "Weekly_Sales",
        }
    )
    result["Date"] = pd.to_datetime(result["Date"]).dt.strftime("%Y-%m-%d")
    result["Id"] = result["Store"].astype(str) + "_" + result["Dept"].astype(str) + "_" + result["Date"]
    return result[["Id", "Weekly_Sales"]]


# Setting UI 
def format_short(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:.1f}k"
    return f"${value:,.0f}"


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validation_box(name: str, df: Optional[pd.DataFrame], required_cols: list[str], optional: bool = False) -> None:
    missing = missing_columns(df, required_cols)
    ready = df is not None and len(missing) == 0

    if ready:
        status_html = '<div class="valid-ok">SIAP</div>'
        rows = len(df)
        cols = len(df.columns)
        note = f"{rows:,} baris · {cols:,} kolom"
    else:
        status_html = '<div class="valid-bad">OPSIONAL</div>' if optional else '<div class="valid-bad">BELUM SIAP</div>'

    st.markdown(
        f"""
        <div class="validation-box">
            <div class="valid-name">{name}</div>
            {status_html}
            <div class="valid-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_train_actual_line(history_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Menyiapkan garis aktual dari train.csv.
    Data diagregasi per tanggal agar menjadi satu kurva historis.
    """
    if history_df is None:
        return None

    df = parse_date(standardize_columns(history_df))
    if "Date" not in df.columns or TARGET_COL not in df.columns:
        return None

    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=["Date", TARGET_COL])
    df = df[df[TARGET_COL] >= 0]

    if df.empty:
        return None

    train_line = (
        df.groupby("Date", as_index=False)[TARGET_COL]
        .sum()
        .sort_values("Date")
        .rename(columns={"Date": "Tanggal", TARGET_COL: "Aktual_Train"})
    )
    return train_line


def prepare_test_prediction_line(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Menyiapkan garis prediksi dari test.csv tanpa target.
    Hasil prediksi diagregasi per tanggal agar terlihat sebagai prediksi ke depan.
    """
    df = result_df.copy()
    df.columns = df.columns.astype(str).str.strip()

    if "Date_Asli" not in df.columns:
        if "Date" in df.columns:
            df["Date_Asli"] = df["Date"]
        else:
            raise ValueError("Kolom tanggal tidak ditemukan. Pastikan hasil prediksi memiliki Date_Asli atau Date.")

    if "Predicted_Weekly_Sales" not in df.columns:
        raise ValueError("Kolom Predicted_Weekly_Sales tidak ditemukan pada hasil prediksi.")

    df["Date_Asli"] = pd.to_datetime(df["Date_Asli"], errors="coerce")
    df["Predicted_Weekly_Sales"] = pd.to_numeric(df["Predicted_Weekly_Sales"], errors="coerce")
    df = df.dropna(subset=["Date_Asli", "Predicted_Weekly_Sales"])

    test_line = (
        df.groupby("Date_Asli", as_index=False)["Predicted_Weekly_Sales"]
        .sum()
        .sort_values("Date_Asli")
        .rename(columns={"Date_Asli": "Tanggal", "Predicted_Weekly_Sales": "Prediksi_Test"})
    )
    return test_line


def plot_train_test_forecast_line(
    history_df: Optional[pd.DataFrame],
    result_df: pd.DataFrame,
) -> None:
    """
    Membuat satu visualisasi line gabungan:
    - train.csv sebagai data aktual historis;
    - test.csv sebagai hasil prediksi ke depan.
    """
    train_line = prepare_train_actual_line(history_df)
    test_line = prepare_test_prediction_line(result_df)

    fig = go.Figure()

    if train_line is not None and not train_line.empty:
        fig.add_trace(
            go.Scatter(
                x=train_line["Tanggal"],
                y=train_line["Aktual_Train"],
                mode="lines",
                name="Aktual Train",
                line=dict(color="#0F172A", width=3),
                hovertemplate=(
                    "<b>Aktual Train</b><br>"
                    "Tanggal: %{x|%d %b %Y}<br>"
                    "Weekly Sales: %{y:,.2f}<extra></extra>"
                ),
            )
        )

        last_train_date = train_line["Tanggal"].max()
        fig.add_vline(
            x=last_train_date,
            line_width=2,
            line_dash="dash",
            line_color="#94A3B8",
        )
        fig.add_annotation(
            x=last_train_date,
            y=1.06,
            yref="paper",
            text="Akhir data train",
            showarrow=False,
            xanchor="left",
            font=dict(size=12, color="#64748B"),
        )

    if test_line is not None and not test_line.empty:
        fig.add_trace(
            go.Scatter(
                x=test_line["Tanggal"],
                y=test_line["Prediksi_Test"],
                mode="lines+markers",
                name="Prediksi Test",
                line=dict(color="#10B981", width=3, dash="dash"),
                marker=dict(size=7, color="#10B981", line=dict(width=1, color="#FFFFFF")),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.10)",
                hovertemplate=(
                    "<b>Prediksi Test</b><br>"
                    "Tanggal: %{x|%d %b %Y}<br>"
                    "Predicted Weekly Sales: %{y:,.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=38, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#0F172A"),
        ),
        xaxis=dict(
            title="Tanggal",
            showgrid=False,
            tickfont=dict(color="#475569", size=11),
        ),
        yaxis=dict(
            title="Weekly Sales",
            showgrid=True,
            gridcolor="#E2E8F0",
            tickfont=dict(color="#475569", size=11),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def build_train_test_timeline(history_df: Optional[pd.DataFrame], result_df: pd.DataFrame) -> pd.DataFrame:
    """Membuat tabel transisi aktual train menuju prediksi test."""
    train_line = prepare_train_actual_line(history_df)
    test_line = prepare_test_prediction_line(result_df)

    parts = []
    if train_line is not None and not train_line.empty:
        train_part = train_line.copy()
        train_part["Prediksi_Test"] = np.nan
        train_part["Jenis_Data"] = "Aktual Train"
        parts.append(train_part)

    if test_line is not None and not test_line.empty:
        test_part = test_line.copy()
        test_part["Aktual_Train"] = np.nan
        test_part["Jenis_Data"] = "Prediksi Test"
        parts.append(test_part)

    if not parts:
        return pd.DataFrame()

    timeline = pd.concat(parts, ignore_index=True)
    timeline = timeline[["Tanggal", "Jenis_Data", "Aktual_Train", "Prediksi_Test"]]
    return timeline.sort_values("Tanggal").reset_index(drop=True)


def dataset_summary(df: Optional[pd.DataFrame], date_col: str = "Date") -> dict:
    """Ringkasan sederhana untuk kartu data utama."""
    if df is None or df.empty:
        return {"rows": 0, "cols": 0, "start": "-", "end": "-"}

    temp = standardize_columns(df)
    rows = len(temp)
    cols = len(temp.columns)
    start = "-"
    end = "-"
    if date_col in temp.columns:
        dates = pd.to_datetime(temp[date_col], errors="coerce").dropna()
        if not dates.empty:
            start = dates.min().strftime("%Y-%m-%d")
            end = dates.max().strftime("%Y-%m-%d")
    return {"rows": rows, "cols": cols, "start": start, "end": end}


def run_prediction_pipeline(
    data: dict[str, Optional[pd.DataFrame]],
    status_box=None,
) -> Optional[pd.DataFrame]:
    """
    Menjalankan pipeline deployment prediksi.
    Log proses ditulis ke status_box agar user tahu model sedang melakukan apa.
    """
    def log(message: str) -> None:
        if status_box is not None:
            status_box.write(message)

    test_df = data.get("test")
    features_df = data.get("features")
    stores_df = data.get("stores")
    history_df = data.get("history")
    residual_history_df = data.get("residual_history")

    log("1. Memvalidasi ketersediaan data utama dan data test.")
    if test_df is None:
        st.error("File `test.csv` belum tersedia. Upload file test tanpa target terlebih dahulu.")
        return None
    if features_df is None:
        st.error("File `features.csv` belum tersedia di folder data/.")
        return None
    if stores_df is None:
        st.error("File `stores.csv` belum tersedia di folder data/.")
        return None
    if history_df is None:
        st.error("File `train.csv` belum tersedia di folder data/. Data train dibutuhkan untuk fitur lag, rolling, dan grafik historis.")
        return None

    missing_test = missing_columns(test_df, REQUIRED_TEST_COLS)
    missing_features = missing_columns(features_df, REQUIRED_FEATURE_COLS)
    missing_stores = missing_columns(stores_df, REQUIRED_STORE_COLS)
    missing_history = missing_columns(history_df, REQUIRED_HISTORY_COLS)

    if len(missing_test) > 0:
        st.error(f"Kolom pada test.csv belum lengkap: {missing_test}")
        return None
    if len(missing_features) > 0:
        st.error(f"Kolom pada features.csv belum lengkap: {missing_features}")
        return None
    if len(missing_stores) > 0:
        st.error(f"Kolom pada stores.csv belum lengkap: {missing_stores}")
        return None
    if len(missing_history) > 0:
        st.error(f"Kolom pada train.csv belum lengkap: {missing_history}")
        return None

    log("2. Memuat model XGBoost dari folder model/.")
    xgb_model, xgb_error = load_pickle_resource(str(XGB_PATH))
    if xgb_model is None:
        st.error(xgb_error if xgb_error is not None else "Model XGBoost tidak dapat dimuat.")
        return None

    log("3. Memuat model LSTM residual dan artefak pendukung.")
    lstm_model, lstm_error = load_keras_resource(str(LSTM_PATH))
    artifact, artifact_error = load_pickle_resource(str(ARTIFACT_PATH))

    if lstm_model is None:
        st.warning(
            "Model LSTM residual tidak digunakan karena tidak dapat dimuat. "
            "Prediksi tetap dijalankan menggunakan XGBoost saja. "
            f"Detail: {lstm_error}"
        )

    if not isinstance(artifact, dict):
        st.warning(
            "Artefak residual tidak terbaca. Koreksi LSTM akan berjalan tanpa artefak tertentu bila memungkinkan. "
            f"Detail: {artifact_error}"
        )

    log("4. Mengintegrasikan test.csv dengan features.csv dan stores.csv.")
    integrated = integrate_test_data(test_df, features_df, stores_df)

    log("5. Membentuk fitur promosi, waktu, kategori, dan variabel eksternal.")
    prepared = feature_engineering(integrated)
    feature_columns = get_model_feature_columns(xgb_model)

    log("6. Menjalankan prediksi awal XGBoost secara batch per tanggal agar lebih cepat.")
    xgb_result = predict_xgb_recursive_batched(
        df=prepared,
        xgb_model=xgb_model,
        feature_columns=feature_columns,
        history_df=history_df,
    )

    if lstm_model is not None:
        scaler_residual = get_residual_scaler(artifact)
        window_size = get_window_size(artifact)
        residual_col = detect_residual_column(residual_history_df)

        log("7. Menjalankan koreksi residual LSTM secara batch per tanggal.")
        final_result = apply_lstm_residual_batched(
            df=xgb_result,
            lstm_model=lstm_model,
            scaler_residual=scaler_residual,
            residual_df=residual_history_df,
            residual_col=residual_col,
            window_size=window_size,
        )
    else:
        log("7. Koreksi LSTM dilewati karena model residual tidak tersedia.")
        final_result = xgb_result.copy()
        final_result["Prediksi_Residual_LSTM"] = 0.0
        final_result["Predicted_Weekly_Sales"] = final_result["Prediksi_XGBoost"].clip(lower=0)

    log("8. Menggabungkan hasil prediksi, memberi kategori, dan menyimpan output.")
    final_result = add_prediction_category(final_result)
    final_result.to_csv(OUTPUT_DIR / "hasil_prediksi_test.csv", index=False)
    build_submission(final_result).to_csv(OUTPUT_DIR / "submission.csv", index=False)
    st.session_state["hasil_prediksi"] = final_result
    return final_result


# rendering 
st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-logo">RF</div>
            <div>
                <div class="brand-title">Retail Forecast Dashboard</div>
                <div class="brand-subtitle"> Data utama tersedia · Upload test · Run prediction</div>
            </div>
        </div>
        <div class="top-actions">Hybrid XGBoost–LSTM · Deployment Forecast</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">● Deployment Prediction</div>
        <div class="hero-title">Sales Forecasting with Hybrid Modeling</div>
        <div class="hero-desc">
            Dashboard ini menggunakan <b>train.csv</b>, <b>features.csv</b>, dan <b>stores.csv</b> sebagai data utama. 
            User hanya perlu mengupload <b>test.csv</b> tanpa target, lalu model hybrid <i>XGBoost–LSTM</i> akan menghasilkan prediksi <i>Weekly_Sales</i> untuk periode ke depan.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Data utama selalu dibaca dari folder data/
folder_data = load_data_from_folder()

# Upload difokuskan untuk test.csv, tetapi ZIP/CSV tetap dapat dikenali
uploaded_files = st.file_uploader(
    "Upload test.csv atau ZIP berisi test.csv",
    type=["csv", "zip"],
    accept_multiple_files=True,
    help="Upload test.csv tanpa kolom Weekly_Sales. train.csv, features.csv, dan stores.csv dibaca dari folder data/.",
)
uploaded_data = read_uploaded_files(uploaded_files)

# Sesuai konsep deployment: train/features/stores dari folder, test dari upload
active_data: dict[str, Optional[pd.DataFrame]] = {
    "history": folder_data.get("history"),
    "features": folder_data.get("features"),
    "stores": folder_data.get("stores"),
    "residual_history": folder_data.get("residual_history"),
    "evaluasi": folder_data.get("evaluasi"),
    "test": uploaded_data.get("test"),
}

st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

# main data
with st.container():
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Data Utama dari Folder data</div>
            <div class="card-subtitle">
                Data train, features, dan stores menjadi sumber utama dashboard. Data ini digunakan untuk konteks historis, integrasi fitur, serta pembentukan lag dan rolling.
            </div>
        """,
        unsafe_allow_html=True,
    )

    v1, v2, v3 = st.columns(3)
    with v1:
        validation_box("train.csv", active_data.get("history"), REQUIRED_HISTORY_COLS, optional=False)
    with v2:
        validation_box("features.csv", active_data.get("features"), REQUIRED_FEATURE_COLS, optional=False)
    with v3:
        validation_box("stores.csv", active_data.get("stores"), REQUIRED_STORE_COLS, optional=False)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

# Ringkasan data utama
s_train = dataset_summary(active_data.get("history"))
s_features = dataset_summary(active_data.get("features"))
s_stores = dataset_summary(active_data.get("stores"), date_col="Date")

m1, m2, m3 = st.columns(3)
with m1:
    metric_card("Train Rows", f"{s_train['rows']:,}", f"Periode {s_train['start']} s.d. {s_train['end']}")
with m2:
    metric_card("Features Rows", f"{s_features['rows']:,}", f"Periode {s_features['start']} s.d. {s_features['end']}")
with m3:
    metric_card("Stores", f"{s_stores['rows']:,}", "Data karakteristik toko")

st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

with st.expander("Lihat preview data utama", expanded=False):
    tab1, tab2, tab3 = st.tabs(["train.csv", "features.csv", "stores.csv"])
    with tab1:
        if active_data.get("history") is not None:
            st.dataframe(active_data["history"].head(50), use_container_width=True, hide_index=True)
        else:
            st.warning("train.csv belum ditemukan di folder data/.")
    with tab2:
        if active_data.get("features") is not None:
            st.dataframe(active_data["features"].head(50), use_container_width=True, hide_index=True)
        else:
            st.warning("features.csv belum ditemukan di folder data/.")
    with tab3:
        if active_data.get("stores") is not None:
            st.dataframe(active_data["stores"].head(50), use_container_width=True, hide_index=True)
        else:
            st.warning("stores.csv belum ditemukan di folder data/.")

# upload data
st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

with st.container():
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Upload Data Test untuk Prediksi</div>
            <div class="card-subtitle">
                Data test digunakan sebagai periode masa depan yang belum memiliki target Weekly_Sales. File ini wajib diupload sebelum tombol Run Prediction dijalankan.
            </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2 = st.columns([0.35, 0.65], vertical_alignment="center")
    with t1:
        validation_box("test.csv upload", active_data.get("test"), REQUIRED_TEST_COLS, optional=False)
    with t2:
        test_df = active_data.get("test")
        if test_df is not None:
            test_summary = dataset_summary(test_df)
            st.markdown(
                f"""
                <div class="small-note">
                <b>Test siap diprediksi.</b><br>
                Jumlah baris: <b>{test_summary['rows']:,}</b> · Jumlah kolom: <b>{test_summary['cols']:,}</b><br>
                Periode test: <b>{test_summary['start']}</b> s.d. <b>{test_summary['end']}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="small-note">
                Upload file test tanpa kolom <b>Weekly_Sales</b> untuk melihat prediksi ke depan.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# button run data
st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

run_col, note_col = st.columns([0.24, 0.76], vertical_alignment="center")
with run_col:
    run_clicked = st.button("🚀 Run Prediction")

if run_clicked:
    with st.status("Menjalankan pipeline prediksi...", expanded=True) as status:
        result = run_prediction_pipeline(active_data, status_box=status)
        if result is not None:
            status.update(label="Prediksi selesai dan output berhasil disimpan.", state="complete", expanded=False)
            st.success("Prediksi berhasil dibuat.")
        else:
            status.update(label="Prediksi gagal. Periksa data dan model.", state="error", expanded=True)

result_df = st.session_state.get("hasil_prediksi")

# hasil
if result_df is not None:
    result_df = result_df.copy()
    result_df["Date_Asli"] = pd.to_datetime(result_df["Date_Asli"], errors="coerce")

    total_pred = float(result_df["Predicted_Weekly_Sales"].sum())
    avg_pred = float(result_df["Predicted_Weekly_Sales"].mean())
    total_rows = len(result_df)
    total_store = result_df["Store_Asli"].nunique() if "Store_Asli" in result_df.columns else 0
    total_dept = result_df["Dept_Asli"].nunique() if "Dept_Asli" in result_df.columns else 0

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        metric_card("Total Predicted", format_short(total_pred), "Akumulasi seluruh prediksi test")
    with r2:
        metric_card("Average Prediction", format_short(avg_pred), "Rata-rata per baris test")
    with r3:
        metric_card("Rows Processed", f"{total_rows:,}", "Jumlah baris test diproses")
    with r4:
        metric_card("Store / Dept", f"{total_store} / {total_dept}", "Entitas pada test")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Aktual Train ke Prediksi Test</div>
                <div class="card-subtitle">
                    Garis gelap menunjukkan data aktual historis dari train.csv, sedangkan garis hijau putus-putus menunjukkan hasil prediksi pada test.csv tanpa target.
                </div>
            """,
            unsafe_allow_html=True,
        )
        plot_train_test_forecast_line(history_df=active_data.get("history"), result_df=result_df)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    with st.expander("Lihat tabel timeline train ke test", expanded=False):
        timeline_df = build_train_test_timeline(active_data.get("history"), result_df)
        if not timeline_df.empty:
            st.dataframe(timeline_df.tail(150), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Timeline Train-Test",
                data=timeline_df.to_csv(index=False).encode("utf-8"),
                file_name="timeline_train_test_forecast.csv",
                mime="text/csv",
            )
        else:
            st.warning("Timeline belum dapat dibuat.")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Preview Hasil Prediksi</div>
                <div class="card-subtitle">Tabel berikut menampilkan sebagian hasil prediksi Weekly_Sales pada data test.</div>
            """,
            unsafe_allow_html=True,
        )

        display_cols = final_display_columns(result_df)
        preview_df = result_df[display_cols].copy()
        st.dataframe(preview_df.head(100), use_container_width=True, hide_index=True)

        csv_full = preview_df.to_csv(index=False).encode("utf-8")
        csv_submission = build_submission(result_df).to_csv(index=False).encode("utf-8")

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download Hasil Prediksi Lengkap",
                data=csv_full,
                file_name="hasil_prediksi_test.csv",
                mime="text/csv",
            )
        with d2:
            st.download_button(
                "Download Format Submission",
                data=csv_submission,
                file_name="submission.csv",
                mime="text/csv",
            )

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Prediction Results</div>
                <div class="card-subtitle">Hasil prediksi akan muncul setelah test.csv diupload dan tombol Run Prediction dijalankan.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
