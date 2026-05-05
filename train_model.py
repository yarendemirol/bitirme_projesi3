# train_model.py
# Stacking Ensemble model egitim scripti
# XGBoost + CatBoost + LightGBM + RandomForest -> Ridge meta-learner

import os
import warnings
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, make_scorer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
import lightgbm as lgb
import joblib

try:
    from catboost import CatBoostRegressor, Pool
except Exception as e:
    raise RuntimeError(
        "CatBoost import edilemedi. Kurulum için: pip install catboost\n"
        f"Detay: {e}"
    )

warnings.filterwarnings("ignore")

# =========================================================
# AYARLAR / YAPILANDIRMA
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MERGE_CSV = False
DATA_DIR = Path(".")

DEDUP_COL = "url"
ILAN_OZ_COL = "Ilan_Ozellikleri"

# CSV'den alinacak sutunlar
RAW_KEEP_COLS = [
    "Fiyat_TL", "Fiyat",
    "Lokasyon", "Baslik", "Açıklama", "Aciklama",
    "Net Metrekare", "Brüt Metrekare", "Oda Sayısı", "Banyo Sayısı",
    "Binanın Yaşı", "Binanın Kat Sayısı", "Bulunduğu Kat",
    "Krediye Uygunluk", "Site İçerisinde",
    "Ilce", "Mahalle",
    "Tipi", "Türü", "Isıtma Tipi", "Tapu Durumu",
    "Ilan_Ozellikleri", "İlan Özellikleri", "Ilan Ozellikleri", "İlan_Ozellikleri", "Ilan Özellikleri",
    # demografik veriler
    "Demo_Toplam_Nufus", "Demo_Egitim_Durumu", "Demo_Ortalama_Yas",
    "Demo_Medeni_Hal", "Demo_Sosyo_Ekonomik_Statu",
]

# Target encoding uygulanacak kategorik sutunlar
TARGET_ENC_COLS = ["Il", "Ilce", "Mahalle", "Tipi", "Türü", "Isıtma Tipi", "Tapu Durumu", "Il_Ilce", "Il_Ilce_Mahalle"]
TARGET_ENC_ALPHA = 20


# =========================================================
# CSV OKUMA / KOLON TEMİZLEME
# =========================================================
def read_csv_robust(fp: Path) -> pd.DataFrame:
    """CSV dosyasini farkli encoding'lerle okur, basarisiz olursa skip modunda dener."""
    encodings = ["utf-8-sig", "utf-8", "cp1254", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(fp, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
            try:
                return pd.read_csv(fp, encoding=enc, engine="python", on_bad_lines="skip", low_memory=False)
            except Exception as e2:
                last_err = e2
    raise RuntimeError(f"Dosya okunamadı: {fp} | Son hata: {last_err}")


def _clean_and_dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sutun isimlerini temizle, tekrarlilari kaldir."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def _keep_only_needed_raw_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Sadece RAW_KEEP_COLS'taki sutunlari tut."""
    df = _clean_and_dedupe_columns(df)
    keep = [c for c in RAW_KEEP_COLS if c in df.columns]
    return df.loc[:, keep].copy()


def _find_ilan_oz_col(df: pd.DataFrame):
    """Ilan_Ozellikleri sutununu bul."""
    candidates = [
        "Ilan_Ozellikleri", "İlan Özellikleri", "Ilan Ozellikleri", "İlan_Ozellikleri", "Ilan Özellikleri"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================================================
# MERGE
# =========================================================
def merge_city_csv(pattern: str, output: str):
    """Sehir CSV dosyalarini birlestir, tekrarlilari at."""
    files = sorted(DATA_DIR.glob(pattern))
    out_path = DATA_DIR / output
    files = [f for f in files if f.name != out_path.name]
    if not files:
        raise FileNotFoundError(f"Hiç dosya bulunamadı: {DATA_DIR.resolve()}/{pattern}")

    dfs = []
    for fp in files:
        df = read_csv_robust(fp)
        df = _clean_and_dedupe_columns(df)
        df["__source_file"] = fp.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True, sort=False)

    if DEDUP_COL in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=[DEDUP_COL], keep="first")
        after = len(merged)
        print(f"[INFO] Dedupe ({DEDUP_COL}): {before} -> {after}")

    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Birleştirildi: {out_path.resolve()} | Dosya: {len(files)} | Satır: {len(merged)}")


def merge_all_if_needed():
    merge_city_csv("istanbul*.csv", "istanbul_merged_all.csv")
    merge_city_csv("ankara*.csv", "ankara_merged_all.csv")
    merge_city_csv("izmir*.csv", "izmir_merged_all.csv")


# =========================================================
# PARSE / CLEAN
# =========================================================
def clean_price_to_int(val):
    if pd.isna(val):
        return np.nan
    s_val = str(val)
    # Breadcrumb/başlık karışmış ilanları tamamen at (eğitime alınmasın)
    if ">" in s_val or "anasayfa" in s_val.lower() or "satılık" in s_val.lower():
        return np.nan
    digits = re.sub(r"[^0-9]", "", s_val)
    if digits == "":
        return np.nan
    return int(digits)


def extract_number(val):
    if pd.isna(val):
        return np.nan
    nums = re.findall(r"\d+", str(val))
    return float(nums[0]) if nums else np.nan


def parse_oda_sayisi(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if "stüdyo" in s or "studio" in s:
        return 1.0
    # "2.5+1", "3,5+1" gibi değerleri doğru parse et
    s_norm = s.replace(",", ".")
    if "+" in s_norm:
        parts = s_norm.split("+")
        total = 0.0
        for p in parts:
            p = p.strip()
            if p:
                try:
                    total += float(p)
                except ValueError:
                    nums = re.findall(r"\d+\.?\d*", p)
                    if nums:
                        total += float(nums[0])
        return total if total > 0 else np.nan
    nums = re.findall(r"\d+\.?\d*", s_norm)
    if not nums:
        return np.nan
    return float(nums[0])


def parse_banyo_sayisi(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if "yok" in s:
        return 0.0
    nums = re.findall(r"\d+", s)
    return float(nums[0]) if nums else np.nan


def parse_bina_yasi(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return np.nan
    if "-" in s and len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2
    if "ve üzeri" in s.lower() or "ve uzeri" in s.lower():
        return float(int(nums[0]) + 5)
    return float(nums[0])


def parse_bina_kat_sayisi(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return np.nan
    if "ve üzeri" in s.lower() or "ve uzeri" in s.lower():
        return float(int(nums[0]) + 2)
    return float(nums[0])


def parse_location(loc):
    if pd.isna(loc):
        return None, None, None
    s = str(loc).replace(" - ", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    il = ilce = mahalle = None
    if len(parts) >= 3:
        il, ilce, mahalle = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        il, ilce = parts[0], parts[1]
    elif len(parts) == 1:
        il = parts[0]
    if mahalle is not None:
        mahalle = (
            mahalle.replace("Mahallesi", "")
                   .replace("Mah.", "")
                   .replace("Mah", "")
                   .strip()
        )
    return il, ilce, mahalle


def flag_from_yes_no(val, yes_word="Evet"):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s == yes_word.lower():
        return 1.0
    if "hayır" in s or "hayir" in s:
        return 0.0
    return np.nan


def parse_bulundugu_kat_ordinal(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    s = (
        s.replace("ı", "i").replace("ç", "c").replace("ğ", "g")
         .replace("ş", "s").replace("ö", "o").replace("ü", "u")
    )

    if "zemin" in s or "giris" in s or "bahce" in s:
        return 0.0

    if "kot" in s:
        nums = re.findall(r"-?\d+", s)
        if nums:
            try:
                n = int(nums[0])
                if n > 0 and "-" not in s:
                    return float(-n)
                return float(n)
            except:
                return np.nan
        return np.nan

    if "bodrum" in s:
        nums = re.findall(r"\d+", s)
        if nums:
            try:
                return float(-int(nums[0]))
            except:
                return -1.0
        return -1.0

    nums = re.findall(r"\d+", s)
    if nums:
        try:
            return float(int(nums[0]))
        except:
            return np.nan

    if "en ust" in s or "cati" in s or "teras" in s:
        return 10.0

    return np.nan


# ---- İlan Özellikleri gürültü temizleme (JSON/POI kırpma + token sadeleştirme)
_TURKISH_TRANS = str.maketrans({
    "\u00e7": "c", "\u00c7": "c",
    "\u011f": "g", "\u011e": "g",
    "\u0131": "i", "\u0130": "i",
    "\u00f6": "o", "\u00d6": "o",
    "\u015f": "s", "\u015e": "s",
    "\u00fc": "u", "\u00dc": "u"
})

def _tr_ascii(s: str) -> str:
    s = s.translate(_TURKISH_TRANS)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def clean_ilan_ozellikleri(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val)

    # JSON / POI karışmışsa agresif kırp
    low = s.lower()
    if ("distance_m" in low) or ('"name"' in low) or ("egitim kurumlari" in low) or ("{" in s) or ("}" in s):
        # ilk { gördüğü yerden itibaren at
        cut = s.find("{")
        if cut != -1:
            s = s[:cut]

    s = s.replace(";", "|").replace(",", "|").replace("/", "|")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.strip()

    s = _tr_ascii(s.lower())

    parts = [p.strip() for p in s.split("|") if p.strip()]

    cleaned = []
    seen = set()
    for p in parts:
        # sadece harf/boşluk bırak
        p = re.sub(r"[^a-z\s]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < 2 or len(p) > 35:
            continue
        if p in seen:
            continue
        seen.add(p)
        cleaned.append(p)

    return "|".join(cleaned)


def normalize_object_cols(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].str.replace(r"\s+", " ", regex=True)
            df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan})
    return df


# =========================================================
# DEMOGRAFİK VERİ PARSE
# =========================================================
def parse_demo_numeric(val: str) -> float:
    """Demografik string'den ilk sayisal degeri cikarir. Ör: '3.988 - Normal' -> 3.988"""
    if pd.isna(val):
        return np.nan
    s = str(val)
    s = s.replace("%", "").replace(",", ".")
    nums = re.findall(r"[0-9]+\.?[0-9]*", s)
    if not nums:
        return np.nan
    try:
        return float(nums[0])
    except:
        return np.nan


# =========================================================
# KEYWORD FEATURES (başlık+açıklama)
# =========================================================
def add_text_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_series = None
    if "Baslik" in df.columns:
        text_series = df["Baslik"].astype(str)
    if "Açıklama" in df.columns:
        text_series = (text_series + " " + df["Açıklama"].astype(str)) if text_series is not None else df["Açıklama"].astype(str)
    if "Aciklama" in df.columns:
        text_series = (text_series + " " + df["Aciklama"].astype(str)) if text_series is not None else df["Aciklama"].astype(str)
    if text_series is None:
        return df

    text = text_series.fillna("").str.lower()

    df["kw_deniz_manzarasi"] = text.str.contains(r"deniz\s*manzar|deniz\s*gorunum|sea\s*view", na=False).astype(int)
    df["kw_denize_yakin"] = text.str.contains(
        r"denize\s*yak[ıi]n|deniz[e]?\s*yak[ıi]n|sahile\s*yak[ıi]n|sahil[e]?\s*yak[ıi]n|k[ıi]y[ıi]ya\s*yak[ıi]n",
        na=False
    ).astype(int)
    df["kw_metroya_yakin"] = text.str.contains(r"metroya\s*yakin|metro\s*yakin|metro\s*yakini|metro", na=False).astype(int)
    df["kw_guney_cephe"]   = text.str.contains(r"guney\s*cephe|güne[yı]\s*cephe", na=False).astype(int)
    df["kw_luks"]          = text.str.contains(r"luks|lüks", na=False).astype(int)
    df["kw_rezidans"]      = text.str.contains(r"rezidans|residence", na=False).astype(int)
    return df


# =========================================================
# İLAN ÖZELLİKLERİ -> SADECE SAYISAL BAYRAKLAR (gürültüsüz)
# =========================================================
def add_ilan_ozellikleri_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if ILAN_OZ_COL not in X.columns:
        X[ILAN_OZ_COL] = ""

    s = X[ILAN_OZ_COL].fillna("").astype(str)

    # token sayısı (| sayısına göre)
    pipe_cnt = s.str.count(r"\|")
    X["ozellik_sayisi"] = np.where(s.str.len() == 0, 0, pipe_cnt + 1)

    low = s.str.lower()

    def has(pat):  # regex
        return low.str.contains(pat, regex=True, na=False).astype(int)

    # daha geniş ama stabil bayrak seti
    X["oz_wifi"]        = has(r"wi\s*-?\s*fi")
    X["oz_fiber"]       = has(r"fiber")
    X["oz_asansor"]     = has(r"asans")
    X["oz_otopark"]     = has(r"otopark")
    X["oz_guvenlik"]    = has(r"guvenlik|kamera sistemi|alarm")
    X["oz_havuz"]       = has(r"havuz")
    X["oz_spor"]        = has(r"spor|fitness|gym")
    X["oz_sauna"]       = has(r"sauna")
    X["oz_metro"]       = has(r"metro")
    X["oz_deniz"]       = has(r"deniz")
    X["oz_manzara"]     = has(r"manzar")
    X["oz_luks"]        = has(r"luks|lüks")
    X["oz_rezidans"]    = has(r"rezidans|residence")
    X["oz_klima"]       = has(r"klima")
    X["oz_balkon"]      = has(r"balkon|cam balkon")
    X["oz_teras"]       = has(r"teras")
    X["oz_giyinme"]     = has(r"giyinme odasi")
    X["oz_ebeveyn"]     = has(r"ebeveyn banyo|ebeveyn")
    X["oz_gomme"]       = has(r"gomme dolap")
    X["oz_yerden_isit"] = has(r"yerden isit")
    X["oz_kombi"]       = has(r"kombi")
    X["oz_merkezi"]     = has(r"merkezi isit")
    X["oz_isicam"]      = has(r"isicam")
    X["oz_celik_kapi"]  = has(r"celik kapi")
    X["oz_pvc"]         = has(r"pvc dograma")
    X["oz_yangin"]      = has(r"yangin merdiven")

    return X


def add_advanced_features(X: pd.DataFrame, y_train=None):
    X = X.copy()

    # ilan özellik bayrakları
    X = add_ilan_ozellikleri_flags(X)

    if "Net_m2" in X.columns and "Oda_sayi" in X.columns:
        X["m2_per_oda"] = X["Net_m2"] / X["Oda_sayi"].replace(0, np.nan)

    if "Brut_m2" in X.columns and "Net_m2" in X.columns:
        X["brut_net_ratio"] = X["Brut_m2"] / X["Net_m2"].replace(0, np.nan)
        X["brut_minus_net"] = X["Brut_m2"] - X["Net_m2"]

    if "Net_m2" in X.columns:
        X["log_net_m2"] = np.log1p(pd.to_numeric(X["Net_m2"], errors="coerce"))

    if "Brut_m2" in X.columns:
        X["log_brut_m2"] = np.log1p(pd.to_numeric(X["Brut_m2"], errors="coerce"))

    if "Bina_yas_yil" in X.columns:
        X["Bina_yas_kategori"] = pd.cut(
            pd.to_numeric(X["Bina_yas_yil"], errors="coerce"),
            bins=[0, 5, 10, 20, 100],
            labels=["Yeni", "Orta", "Eski", "Cok Eski"]
        ).astype("category")

    if "Net_m2" in X.columns:
        X["m2_kategori"] = pd.cut(
            pd.to_numeric(X["Net_m2"], errors="coerce"),
            bins=[0, 75, 100, 150, 200, 1000],
            labels=["Kucuk", "Orta", "Buyuk", "Cok Buyuk", "Luks"]
        ).astype("category")

    if "Il" in X.columns and "Ilce" in X.columns:
        X["Il_Ilce"] = X["Il"].astype(str) + "_" + X["Ilce"].astype(str)

    if "Kat_ordinal" in X.columns:
        ko = pd.to_numeric(X["Kat_ordinal"], errors="coerce")
        X["Zemin_kat"] = (ko == 0).astype(int)
        X["Yuksek_kat"] = (ko > 5).astype(int)

    # --- Etkileşim özellikleri (MAE düşürücü) ---
    if "Kat_ordinal" in X.columns and "Bina_kat_sayisi" in X.columns:
        bks = pd.to_numeric(X["Bina_kat_sayisi"], errors="coerce").replace(0, np.nan)
        X["Kat_ratio"] = pd.to_numeric(X["Kat_ordinal"], errors="coerce") / bks

    if "Net_m2" in X.columns and "Fiyat_m2" in X.columns:
        _nm2 = pd.to_numeric(X["Net_m2"], errors="coerce")
        _fm2 = pd.to_numeric(X["Fiyat_m2"], errors="coerce")
        X["log_m2_x_fiyat_m2"] = np.log1p((_nm2 * _fm2).clip(lower=0))

    if "Net_m2" in X.columns and "Banyo_sayi" in X.columns:
        X["m2_per_banyo"] = pd.to_numeric(X["Net_m2"], errors="coerce") / pd.to_numeric(X["Banyo_sayi"], errors="coerce").replace(0, np.nan)

    # --- Yeni etkileşim özellikleri (Phase 2) ---
    if "Oda_sayi" in X.columns and "Net_m2" in X.columns:
        X["Oda_x_m2"] = pd.to_numeric(X["Oda_sayi"], errors="coerce") * pd.to_numeric(X["Net_m2"], errors="coerce")

    if "Fiyat_m2" in X.columns:
        X["log_fiyat_m2"] = np.log1p(pd.to_numeric(X["Fiyat_m2"], errors="coerce").clip(lower=0))

    if "Bina_yas_yil" in X.columns and "Net_m2" in X.columns:
        X["Bina_yas_x_m2"] = pd.to_numeric(X["Bina_yas_yil"], errors="coerce") * pd.to_numeric(X["Net_m2"], errors="coerce")

    if "Il" in X.columns and "Ilce" in X.columns and "Mahalle" in X.columns:
        X["Il_Ilce_Mahalle"] = X["Il"].astype(str) + "_" + X["Ilce"].astype(str) + "_" + X["Mahalle"].astype(str)

    return X


# =========================================================
# LOAD ALL CITIES
# =========================================================
def load_all_cities():
    """3 sehrin CSV dosyalarini yukle ve birlestir."""
    files = [
        ("İstanbul", BASE_DIR / "istanbul_merged_all.csv"),
        ("Ankara",   BASE_DIR / "ankara_merged_all.csv"),
        ("İzmir",    BASE_DIR / "izmir_merged_all.csv"),
    ]

    dfs = []
    for city, path in files:
        df = read_csv_robust(path)
        df = _keep_only_needed_raw_cols(df)
        df["Il"] = city
        df = _clean_and_dedupe_columns(df)
        dfs.append(df)
        print(f"[INFO] {city} yüklendi. Satır sayısı: {len(df)} | {path}")

    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    df_all = _clean_and_dedupe_columns(df_all)
    print("[INFO] Tüm şehirler birleştirildi. Toplam satır:", len(df_all))
    return df_all


# =========================================================
# BUILD X / y
# =========================================================
def build_X_y(df: pd.DataFrame):
    """Ham veriyi parse edip X (ozellik matrisi) ve y (fiyat TL) olarak dondurur."""
    df = df.copy()
    df = _clean_and_dedupe_columns(df)

    # ilan özellikleri canonical + temiz
    oz_col = _find_ilan_oz_col(df)
    if oz_col is None:
        df[ILAN_OZ_COL] = ""
    else:
        df[ILAN_OZ_COL] = df[oz_col]

    df[ILAN_OZ_COL] = df[ILAN_OZ_COL].apply(clean_ilan_ozellikleri)

    # fiyat
    if "Fiyat_TL" not in df.columns:
        if "Fiyat" not in df.columns:
            raise ValueError("Ne 'Fiyat_TL' ne 'Fiyat' kolonu bulunamadı!")
        df["Fiyat_TL"] = df["Fiyat"].apply(clean_price_to_int)

    df["Fiyat_TL"] = pd.to_numeric(df["Fiyat_TL"], errors="coerce")

    # lokasyon -> ilce/mahalle
    if "Lokasyon" in df.columns:
        il_ilce_mah = df["Lokasyon"].apply(lambda x: pd.Series(parse_location(x)))
        il_ilce_mah.columns = ["Il_loc", "Ilce_loc", "Mahalle_loc"]
        df = pd.concat([df, il_ilce_mah], axis=1)

        if "Ilce" in df.columns:
            df["Ilce"] = df["Ilce"].where(df["Ilce"].notna(), df["Ilce_loc"])
        else:
            df["Ilce"] = df["Ilce_loc"]

        if "Mahalle" in df.columns:
            df["Mahalle"] = df["Mahalle"].where(df["Mahalle"].notna(), df["Mahalle_loc"])
        else:
            df["Mahalle"] = df["Mahalle_loc"]

        df["Mahalle"] = (
            df["Mahalle"].astype(str)
            .str.replace("Mahallesi", "", regex=False)
            .str.replace("Mah.", "", regex=False)
            .str.replace("Mah", "", regex=False)
            .str.strip()
        )

    # temel numeric parse
    df["Net_m2"] = df["Net Metrekare"].apply(extract_number) if "Net Metrekare" in df.columns else np.nan
    df["Brut_m2"] = df["Brüt Metrekare"].apply(extract_number) if "Brüt Metrekare" in df.columns else np.nan
    df["Oda_sayi"] = df["Oda Sayısı"].apply(parse_oda_sayisi) if "Oda Sayısı" in df.columns else np.nan
    df["Banyo_sayi"] = df["Banyo Sayısı"].apply(parse_banyo_sayisi) if "Banyo Sayısı" in df.columns else np.nan
    df["Bina_yas_yil"] = df["Binanın Yaşı"].apply(parse_bina_yasi) if "Binanın Yaşı" in df.columns else np.nan
    df["Bina_kat_sayisi"] = df["Binanın Kat Sayısı"].apply(parse_bina_kat_sayisi) if "Binanın Kat Sayısı" in df.columns else np.nan

    # flags
    if "Krediye Uygunluk" in df.columns:
        df["Kredi_uygun_flag"] = df["Krediye Uygunluk"].apply(
            lambda x: 0.0 if isinstance(x, str) and "Uygun Değil" in x
            else (1.0 if isinstance(x, str) and "Krediye Uygun" in x else np.nan)
        )
    else:
        df["Kredi_uygun_flag"] = np.nan

    if "Site İçerisinde" in df.columns:
        df["Site_icinde_flag"] = df["Site İçerisinde"].apply(lambda x: flag_from_yes_no(x, yes_word="Evet"))
    else:
        df["Site_icinde_flag"] = np.nan

    if "Bulunduğu Kat" in df.columns:
        df["Kat_ordinal"] = df["Bulunduğu Kat"].apply(parse_bulundugu_kat_ordinal)
    else:
        df["Kat_ordinal"] = np.nan

    # object kolon normalize (cardinality gürültüsü azalır)
    df = normalize_object_cols(df, ["Il", "Ilce", "Mahalle", "Tipi", "Türü", "Isıtma Tipi", "Tapu Durumu"])

    # keyword flags
    df = add_text_keyword_features(df)

    # demografik veriler: string -> sayisal deger
    if "Demo_Toplam_Nufus" in df.columns:
        df["Demo_Toplam_Nufus_num"] = df["Demo_Toplam_Nufus"].apply(parse_demo_numeric)
    if "Demo_Egitim_Durumu" in df.columns:
        df["Demo_Egitim_Durumu_pct"] = df["Demo_Egitim_Durumu"].apply(parse_demo_numeric)
    if "Demo_Ortalama_Yas" in df.columns:
        df["Demo_Ortalama_Yas_num"] = df["Demo_Ortalama_Yas"].apply(parse_demo_numeric)
    if "Demo_Medeni_Hal" in df.columns:
        df["Demo_Medeni_Hal_pct"] = df["Demo_Medeni_Hal"].apply(parse_demo_numeric)
    if "Demo_Sosyo_Ekonomik_Statu" in df.columns:
        df["Demo_Sosyo_Ekonomik_Statu"] = df["Demo_Sosyo_Ekonomik_Statu"].astype(str)

    # filtre (aynı)
    df["_tmp_fiyat_m2"] = df["Fiyat_TL"] / df["Net_m2"]
    df = df[df["Fiyat_TL"] >= 400_000]
    df = df[df["_tmp_fiyat_m2"] >= 6_000]

    # Tipi == "Bina" olanları kırp (çok heterojen, standart konut MAE'sini bozar)
    if "Tipi" in df.columns:
        n_bina = (df["Tipi"].astype(str).str.lower() == "bina").sum()
        df = df[df["Tipi"].astype(str).str.lower() != "bina"]
        if n_bina > 0:
            print(f"[INFO] Tipi='Bina' olan {n_bina} ilan kırpıldı.")

    # Hisseli Tapu olanları kırp (yüzde kaç hisse satıldığı bilinmiyor, fiyat ilişkisi farklı)
    if "Tapu Durumu" in df.columns:
        tapu_lower = df["Tapu Durumu"].astype(str).str.lower()
        hisseli_mask = tapu_lower.str.contains("hisseli", na=False)
        n_hisseli = hisseli_mask.sum()
        df = df[~hisseli_mask]
        if n_hisseli > 0:
            print(f"[INFO] Tapu Durumu='Hisseli Tapu' olan {n_hisseli} ilan kırpıldı.")

    # Net_m2 > Brut_m2 veya brut_net_ratio < 0.35 olan ilanları kırp
    if "Net_m2" in df.columns and "Brut_m2" in df.columns:
        _net = pd.to_numeric(df["Net_m2"], errors="coerce")
        _brut = pd.to_numeric(df["Brut_m2"], errors="coerce")
        _ratio = _net / _brut
        bad_m2 = (_net > _brut) | (_ratio < 0.35)
        bad_m2 = bad_m2.fillna(False)
        n_bad = bad_m2.sum()
        if n_bad > 0:
            print(f"[INFO] Net_m2 > Brut_m2 veya brut_net_ratio < 0.35 olan {n_bad} ilan kırpıldı.")
        df = df[~bad_m2]

    before = len(df)
    df = df.dropna(subset=["Fiyat_TL", "Net_m2"])
    df = df[df["Fiyat_TL"] > 0]
    df = df[df["Net_m2"] > 0]

    # Sabit eşik değerleri kullan (quantile hesabı tüm veriyi kullanıp
    # test verisinden train verisine bilgi sızdırıyordu - leakage fix)
    # Eski quantile değerlerine yakın sabit eşikler:
    PRICE_LOWER = 1_500_000   # ≈ eski q1 (alt %1)
    PRICE_UPPER = 55_000_000  # ≈ eski q99 (üst %1)
    df = df[(df["Fiyat_TL"] >= PRICE_LOWER) & (df["Fiyat_TL"] <= PRICE_UPPER)]

    # Fiyat_m2 üst sınır: ≈ eski fm2_q99
    FM2_UPPER = 270_000   # ≈ eski fiyat_m2 q99
    if df["_tmp_fiyat_m2"].notna().sum() > 0:
        df = df[df["_tmp_fiyat_m2"] <= FM2_UPPER]

    after = len(df)
    print(f"[INFO] Outlier / eksik temizliği ile atılan ilan sayısı: {before - after}")

    y_tl = df["Fiyat_TL"]
    X = df.drop(columns=["Fiyat_TL"])

    # ham text sütunları modelde kullanmayacağız (sadece bayrak üreteceğiz)
    drop_useless = [
        "url", "Fiyat", "Lokasyon", "Baslik", "Açıklama", "Aciklama",
        "İlan Numarası", "İlan Oluşturma Tarihi", "İlan Güncelleme Tarihi",
        "Net Metrekare", "Brüt Metrekare", "Oda Sayısı", "Banyo Sayısı",
        "Binanın Yaşı", "Binanın Kat Sayısı", "Krediye Uygunluk", "Site İçerisinde",
        "Fiyat Durumu", "Il_loc", "Ilce_loc", "Mahalle_loc",
        "Slug", "_tmp_fiyat_m2", "__source_file", "ParseStatus", "Error",
        "Bulunduğu Kat",
        "Demo_Toplam_Nufus", "Demo_Egitim_Durumu", "Demo_Ortalama_Yas", "Demo_Medeni_Hal",
    ]
    drop_existing = [c for c in drop_useless if c in X.columns]
    if drop_existing:
        print("[INFO] Atılan gereksiz sütunlar:", drop_existing)
        X = X.drop(columns=drop_existing)

    # numeric coercions
    for col in [
        "Net_m2","Brut_m2","Oda_sayi","Banyo_sayi","Bina_yas_yil","Bina_kat_sayisi",
        "Kredi_uygun_flag","Site_icinde_flag","Kat_ordinal",
        "kw_deniz_manzarasi","kw_denize_yakin","kw_metroya_yakin","kw_guney_cephe","kw_luks","kw_rezidans",
        "Fiyat_m2",
        "Demo_Toplam_Nufus_num", "Demo_Egitim_Durumu_pct", "Demo_Ortalama_Yas_num", "Demo_Medeni_Hal_pct"
    ]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = _clean_and_dedupe_columns(X)

    print("[INFO] Özellik matrisi X shape:", X.shape)
    print("[INFO] Hedef y (TL) shape:", y_tl.shape)
    return X, y_tl


# =========================================================
# PREPROCESSOR (XGB)
# =========================================================
def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64", "Int64", "Float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_features = [c for c in categorical_features if c != ILAN_OZ_COL]

    print("[INFO] Sayısal sütunlar:", numeric_features)
    print("[INFO] Kategorik sütunlar:", categorical_features)
    if ILAN_OZ_COL in X.columns:
        print("[INFO] İlan özellikleri ham text -> modele VERİLMEYECEK (sadece oz_* bayrakları kullanılacak)")
    HIGH_CARD = {"Ilce", "Mahalle", "Il_Ilce", "Il_Ilce_Mahalle"}
    categorical_features = [c for c in categorical_features if c not in HIGH_CARD and c != ILAN_OZ_COL]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop"
    )


# =========================================================
# METRİKLER
# =========================================================
def evaluate_model(y_test_tl, y_pred_tl, model_name: str):
    y_pred_tl = np.maximum(y_pred_tl, 0)
    mae = mean_absolute_error(y_test_tl, y_pred_tl)
    rmse = np.sqrt(mean_squared_error(y_test_tl, y_pred_tl))
    r2 = r2_score(y_test_tl, y_pred_tl)
    med_ae = median_absolute_error(y_test_tl, y_pred_tl)
    mean_price = float(np.mean(y_test_tl))
    mae_ratio = mae / mean_price if mean_price else np.nan

    print(f"\n========== {model_name.upper()} MODEL PERFORMANSI ==========")
    print(f"MAE   : {mae:,.2f} TL")
    print(f"Median AE : {med_ae:,.2f} TL")
    print(f"RMSE  : {rmse:,.2f} TL")
    print(f"R²    : {r2:.4f}")
    print(f"Ortalama Fiyat: {mean_price:,.2f} TL | MAE%: {mae_ratio*100:.2f}%")


def evaluate_mae_by_city(y_test_tl, y_pred_tl, il_test, title="ŞEHİR BAZLI TEST MAE"):
    df_eval = pd.DataFrame({
        "Il": il_test.astype(str).values,
        "y_true": np.array(y_test_tl),
        "y_pred": np.array(y_pred_tl),
    })
    df_eval["abs_err"] = (df_eval["y_true"] - df_eval["y_pred"]).abs()

    print(f"\n=========== {title} ===========")
    for city, g in df_eval.groupby("Il"):
        mae = g["abs_err"].mean()
        mean_price = g["y_true"].mean()
        ratio = (mae / mean_price * 100) if mean_price else np.nan
        print(f"{city:10s} | n={len(g):4d} | MAE={mae:,.0f} TL | Ortalama={mean_price:,.0f} TL | MAE%={ratio:.2f}")


# =========================================================
# Leakage-free Fiyat_m2 (TL üzerinden)
# =========================================================
def _safe_series(df: pd.DataFrame, colname: str):
    x = df[colname]
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def add_leakage_free_fiyat_m2(X_train: pd.DataFrame, y_train_tl: pd.Series, X_target: pd.DataFrame, n_splits=5):
    X_train = _clean_and_dedupe_columns(X_train.copy())
    X_target = _clean_and_dedupe_columns(X_target.copy())

    def _get(df, name):
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return _safe_series(df, name)

    net_tr = pd.to_numeric(_get(X_train, "Net_m2"), errors="coerce")
    net_ta = pd.to_numeric(_get(X_target, "Net_m2"), errors="coerce")

    if net_tr.isna().all():
        X_train["Fiyat_m2"] = np.nan
        X_target["Fiyat_m2"] = np.nan
        return X_train, X_target

    pm2_tr = (y_train_tl.astype(float) / net_tr).replace([np.inf, -np.inf], np.nan)
    global_mean = pm2_tr.mean()

    keys = ["Il", "Ilce", "Mahalle"]
    for k in keys:
        if k not in X_train.columns:
            X_train[k] = np.nan
        if k not in X_target.columns:
            X_target[k] = np.nan

    k_train = X_train[keys].fillna("__MISSING__").astype(str)
    k_target = X_target[keys].fillna("__MISSING__").astype(str)

    oof = pd.Series(index=X_train.index, dtype="float64")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(X_train):
        tmp = k_train.iloc[tr_idx].copy()
        tmp["pm2"] = pm2_tr.iloc[tr_idx].values

        map3 = tmp.groupby(keys)["pm2"].mean()
        map2 = tmp.groupby(keys[:2])["pm2"].mean()
        map1 = tmp.groupby(keys[:1])["pm2"].mean()

        k3 = pd.MultiIndex.from_frame(k_train.iloc[val_idx])
        k2 = pd.MultiIndex.from_frame(k_train.iloc[val_idx][keys[:2]])
        k1 = pd.MultiIndex.from_frame(k_train.iloc[val_idx][keys[:1]])

        out = pd.Series(k3.map(map3), index=k_train.iloc[val_idx].index, dtype="float64")
        out = out.fillna(pd.Series(k2.map(map2), index=k_train.iloc[val_idx].index, dtype="float64"))
        out = out.fillna(pd.Series(k1.map(map1), index=k_train.iloc[val_idx].index, dtype="float64"))
        out = out.fillna(global_mean)
        oof.iloc[val_idx] = out.values

    tmp_all = k_train.copy()
    tmp_all["pm2"] = pm2_tr.values
    map3_all = tmp_all.groupby(keys)["pm2"].mean()
    map2_all = tmp_all.groupby(keys[:2])["pm2"].mean()
    map1_all = tmp_all.groupby(keys[:1])["pm2"].mean()

    k3t = pd.MultiIndex.from_frame(k_target)
    k2t = pd.MultiIndex.from_frame(k_target[keys[:2]])
    k1t = pd.MultiIndex.from_frame(k_target[keys[:1]])

    out_t = pd.Series(k3t.map(map3_all), index=k_target.index, dtype="float64")
    out_t = out_t.fillna(pd.Series(k2t.map(map2_all), index=k_target.index, dtype="float64"))
    out_t = out_t.fillna(pd.Series(k1t.map(map1_all), index=k_target.index, dtype="float64"))
    out_t = out_t.fillna(global_mean)

    X_train["Fiyat_m2"] = oof.values
    X_target["Fiyat_m2"] = out_t.values
    return X_train, X_target


def add_target_encoding_oof(X_train, y_train_log, X_target, col, n_splits=5, random_state=42, alpha=20):
    if col not in X_train.columns:
        return None, None

    tr_col = X_train[col].fillna("UNKNOWN").astype(str)
    ta_col = X_target[col].fillna("UNKNOWN").astype(str)
    global_mean = float(np.mean(y_train_log))

    oof = pd.Series(index=X_train.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for tr_idx, val_idx in kf.split(X_train):
        vals_tr = tr_col.iloc[tr_idx]
        y_tr = y_train_log.iloc[tr_idx]
        stats = y_tr.groupby(vals_tr).agg(['mean', 'count'])
        enc = (stats['mean'] * stats['count'] + global_mean * alpha) / (stats['count'] + alpha)
        oof.iloc[val_idx] = tr_col.iloc[val_idx].map(enc)

    oof = oof.fillna(global_mean)

    full_stats = y_train_log.groupby(tr_col).agg(['mean', 'count'])
    enc_full = (full_stats['mean'] * full_stats['count'] + global_mean * alpha) / (full_stats['count'] + alpha)
    target = ta_col.map(enc_full).fillna(global_mean)

    return oof, target


# =========================================================
# XGB TUNE (corr hedef) ama CV scoring TL-MAE
# =========================================================
def tl_mae_scorer(estimator, X_raw, y_true_log):
    pred_log = estimator.predict(X_raw)
    pred_tl = np.expm1(pred_log)
    true_tl = np.expm1(y_true_log)
    return -mean_absolute_error(true_tl, pred_tl)

def tune_xgboost(X_train, y_train_log, preprocessor):
    print("\n[INFO] XGBoost hiperparametre araması (log-target, TL-MAE scorer) başlıyor...")

    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb_base)])


    param_distributions = {
        "model__n_estimators": [1000, 1500, 2000],
        "model__max_depth": [5, 6, 7, 8],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.04],
        "model__subsample": [0.75, 0.85, 0.95],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__min_child_weight": [2, 3, 5],
        "model__reg_lambda": [3.0, 6.0, 10.0],
        "model__reg_alpha": [0.0, 0.3, 0.8],
        "model__gamma": [0, 0.1, 0.3],
    }

    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring=tl_mae_scorer,
        n_jobs=3,
        verbose=1,
        random_state=42,
        refit=True,
        pre_dispatch=3
    )
    rs.fit(X_train, y_train_log)

    print("\n[INFO] XGBoost en iyi parametreler:")
    print(rs.best_params_)
    print(f"[INFO] XGBoost CV best (neg TL-MAE): {rs.best_score_:.4f}")

    return rs.best_estimator_


# =========================================================
# CATBOOST (log hedef)
# =========================================================
def prepare_catboost_frame(X: pd.DataFrame, feature_cols=None, cat_cols=None, num_medians=None):
    X2 = X.copy()
    if feature_cols is not None:
        X2 = X2.reindex(columns=feature_cols)

    if cat_cols is None:
        cat_cols = X2.select_dtypes(include=["object", "category"]).columns.tolist()

    # ham ilan text'i catboost'a sokma
    if ILAN_OZ_COL in X2.columns:
        X2 = X2.drop(columns=[ILAN_OZ_COL])
    if ILAN_OZ_COL in cat_cols:
        cat_cols.remove(ILAN_OZ_COL)

    num_cols = [c for c in X2.columns if c not in cat_cols]

    for c in cat_cols:
        X2[c] = X2[c].astype(str)
        X2.loc[X2[c].isin(["nan", "NaN", "None", "<NA>"]), c] = "MISSING"
        X2[c] = X2[c].fillna("MISSING")

    for c in num_cols:
        X2[c] = pd.to_numeric(X2[c], errors="coerce")
        X2[c] = X2[c].replace([np.inf, -np.inf], np.nan)

    if num_medians is None:
        num_medians = X2[num_cols].median(numeric_only=True)

    for c in num_cols:
        X2[c] = X2[c].fillna(num_medians.get(c, np.nan))

    cat_feature_indices = [X2.columns.get_loc(c) for c in cat_cols if c in X2.columns]
    return X2, cat_cols, cat_feature_indices, num_medians


def train_catboost_safe(X_tr: pd.DataFrame, y_tr_log: pd.Series, X_va: pd.DataFrame, y_va_log: pd.Series):
    feature_cols = X_tr.columns.tolist()
    X_tr_cb, cat_cols, cat_idx, num_medians = prepare_catboost_frame(X_tr, feature_cols=feature_cols)
    X_va_cb, _, _, _ = prepare_catboost_frame(X_va, feature_cols=X_tr_cb.columns.tolist(), cat_cols=cat_cols, num_medians=num_medians)

    train_pool = Pool(X_tr_cb, label=y_tr_log.values, cat_features=cat_idx)
    val_pool = Pool(X_va_cb, label=y_va_log.values, cat_features=cat_idx)

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=5000,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=4.0,
        random_seed=42,
        allow_writing_files=False,
        verbose=250,
        od_type="Iter",
        od_wait=300,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, X_tr_cb.columns.tolist(), cat_cols, num_medians


def find_best_weight(y_val_tl, pred_xgb_tl, pred_cat_tl):
    best_w = 0.5
    best_mae = float("inf")
    for w in np.linspace(0, 1, 101):
        p = w * pred_cat_tl + (1 - w) * pred_xgb_tl
        mae = mean_absolute_error(y_val_tl, p)
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)
    return best_w, best_mae


def find_best_weights_3(y_val_tl, pred_xgb_tl, pred_cat_tl, pred_lgb_tl, step=0.05):
    """3 model icin agirlik araması (w_xgb + w_cat + w_lgb = 1)."""
    best_w = (1/3, 1/3, 1/3)
    best_mae = float("inf")
    steps = np.arange(0, 1.01, step)
    for w_xgb in steps:
        for w_cat in steps:
            w_lgb = 1.0 - w_xgb - w_cat
            if w_lgb < -0.001 or w_lgb > 1.001:
                continue
            w_lgb = max(0.0, min(1.0, w_lgb))
            p = w_xgb * pred_xgb_tl + w_cat * pred_cat_tl + w_lgb * pred_lgb_tl
            mae = mean_absolute_error(y_val_tl, p)
            if mae < best_mae:
                best_mae = mae
                best_w = (float(w_xgb), float(w_cat), float(w_lgb))
    return best_w, best_mae


# =========================================================
# RANDOM FOREST (log hedef)
# =========================================================
def train_random_forest(X_tr: pd.DataFrame, y_tr_log: pd.Series,
                       X_va: pd.DataFrame, y_va_log: pd.Series,
                       preprocessor):
    """Random Forest modelini egit."""
    print("\n[INFO] Random Forest eğitimi başlıyor...")

    X_tr_proc = preprocessor.transform(X_tr)
    X_va_proc = preprocessor.transform(X_va)

    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    rf_model.fit(X_tr_proc, y_tr_log)
    pred_val_log = rf_model.predict(X_va_proc)
    pred_val_tl = np.expm1(pred_val_log)
    y_va_tl_arr = np.expm1(y_va_log)
    mae = mean_absolute_error(y_va_tl_arr, pred_val_tl)
    print(f"[INFO] Random Forest Val MAE (TL): {mae:,.0f} TL")
    return rf_model


# =========================================================
# OUTLIER REMOVAL
# =========================================================
def remove_outliers(X: pd.DataFrame, y_tl: pd.Series, max_price=55_000_000, max_net_m2=1000):
    """Sabit esik degerlerle asiri fiyat/m2 olan ilanlari at (leakage-free)."""
    n_before = len(X)
    net_m2 = pd.to_numeric(X.get("Net_m2", pd.Series(dtype=float)), errors="coerce")

    mask_price = y_tl <= max_price
    mask_area = (net_m2 <= max_net_m2) | net_m2.isna()
    keep = mask_price & mask_area

    X_clean = X.loc[keep].copy()
    y_clean = y_tl.loc[keep].copy()
    n_after = len(X_clean)

    print(f"[INFO] Outlier temizliği: {n_before} -> {n_after} satır ({n_before - n_after} satır çıkarıldı)")
    print(f"  - Fiyat üst sınır: {max_price:,.0f} TL")
    print(f"  - Maksimum Net m²: {max_net_m2} m²")
    return X_clean, y_clean


# =========================================================
# LIGHTGBM (log hedef)
# =========================================================
def train_lightgbm(X_tr: pd.DataFrame, y_tr_log: pd.Series,
                   X_va: pd.DataFrame, y_va_log: pd.Series,
                   preprocessor):
    """LightGBM modelini early stopping ile egit."""
    print("\n[INFO] LightGBM eğitimi başlıyor...")

    X_tr_proc = preprocessor.transform(X_tr)
    X_va_proc = preprocessor.transform(X_va)

    lgb_model = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=127,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.3,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    lgb_model.fit(
        X_tr_proc, y_tr_log,
        eval_set=[(X_va_proc, y_va_log)],
        callbacks=[
            lgb.early_stopping(300, verbose=True),
            lgb.log_evaluation(250),
        ],
    )

    print(f"[INFO] LightGBM best iteration: {lgb_model.best_iteration_}")
    return lgb_model


# =========================================================
# PARITY FIX WRAPPERS (log model -> TL output)
# =========================================================
class InferenceFeatureBuilder:
    def __init__(self, te_cols=None, te_alpha=20, pm2_keys=("Il","Ilce","Mahalle")):
        self.te_cols = te_cols or []
        self.te_alpha = float(te_alpha)
        self.pm2_keys = list(pm2_keys)

    def fit(self, X: pd.DataFrame, y_tl: pd.Series):
        X = X.copy()
        y_tl = pd.Series(y_tl).astype(float)

        y_log = np.log1p(y_tl)
        self.te_global_mean_ = float(y_log.mean())
        self.te_maps_ = {}

        # Il_Ilce ve Il_Ilce_Mahalle fit sırasında da mevcut olmalı (TE mapping oluşsun)
        if "Il" in X.columns and "Ilce" in X.columns:
            X["Il_Ilce"] = X["Il"].astype(str) + "_" + X["Ilce"].astype(str)
        if "Il" in X.columns and "Ilce" in X.columns and "Mahalle" in X.columns:
            X["Il_Ilce_Mahalle"] = X["Il"].astype(str) + "_" + X["Ilce"].astype(str) + "_" + X["Mahalle"].astype(str)

        for col in self.te_cols:
            if col not in X.columns:
                continue
            s = X[col].fillna("UNKNOWN").astype(str)
            stats = y_log.groupby(s).agg(["mean", "count"])
            enc = (stats["mean"] * stats["count"] + self.te_global_mean_ * self.te_alpha) / (stats["count"] + self.te_alpha)
            self.te_maps_[col] = enc

        net = pd.to_numeric(X.get("Net_m2", np.nan), errors="coerce")
        pm2 = (y_tl / net).replace([np.inf, -np.inf], np.nan)
        self.pm2_global_ = float(pm2.mean())

        K = X.reindex(columns=self.pm2_keys).copy()
        for k in self.pm2_keys:
            if k not in K.columns:
                K[k] = "__MISSING__"
        K = K.fillna("__MISSING__").astype(str)

        tmp = K.copy()
        tmp["pm2"] = pm2.values
        self.pm2_map3_ = tmp.groupby(self.pm2_keys)["pm2"].mean()
        self.pm2_map2_ = tmp.groupby(self.pm2_keys[:2])["pm2"].mean()
        self.pm2_map1_ = tmp.groupby(self.pm2_keys[:1])["pm2"].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X2 = X.copy()

        # ilan özellikleri canonical + temiz
        if ILAN_OZ_COL in X2.columns:
            X2[ILAN_OZ_COL] = X2[ILAN_OZ_COL].apply(clean_ilan_ozellikleri)
        else:
            X2[ILAN_OZ_COL] = ""

        for col, enc_series in self.te_maps_.items():
            te_name = f"{col}_te"
            s = X2[col].fillna("UNKNOWN").astype(str) if col in X2.columns else pd.Series("UNKNOWN", index=X2.index)
            X2[te_name] = s.map(enc_series).fillna(self.te_global_mean_)

        keys = self.pm2_keys
        K = X2.reindex(columns=keys).copy()
        for k in keys:
            if k not in K.columns:
                K[k] = "__MISSING__"
        K = K.fillna("__MISSING__").astype(str)

        mi3 = pd.MultiIndex.from_frame(K[keys])
        mi2 = pd.MultiIndex.from_frame(K[keys[:2]])
        mi1 = pd.MultiIndex.from_frame(K[keys[:1]])

        out = pd.Series(mi3.map(self.pm2_map3_), index=X2.index, dtype="float64")
        out = out.fillna(pd.Series(mi2.map(self.pm2_map2_), index=X2.index, dtype="float64"))
        out = out.fillna(pd.Series(mi1.map(self.pm2_map1_), index=X2.index, dtype="float64"))
        out = out.fillna(self.pm2_global_)
        X2["Fiyat_m2"] = out.values

        X2 = add_advanced_features(X2, y_train=None)

        # Il_Ilce ve Il_Ilce_Mahalle, add_advanced_features() tarafından üretiliyor
        # Bu yüzden TE map'lerini burada (üretildikten sonra) uygula
        for col in ["Il_Ilce", "Il_Ilce_Mahalle"]:
            if col in self.te_maps_ and col in X2.columns:
                te_name = f"{col}_te"
                s = X2[col].fillna("UNKNOWN").astype(str)
                X2[te_name] = s.map(self.te_maps_[col]).fillna(self.te_global_mean_)

        return X2


def _ensure_columns(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    df2 = df.copy()
    for c in required_cols:
        if c not in df2.columns:
            df2[c] = np.nan
    if ILAN_OZ_COL in required_cols and ILAN_OZ_COL not in df2.columns:
        df2[ILAN_OZ_COL] = ""
    return df2


class XGBModelWithFeatures:
    def __init__(self, feature_builder: InferenceFeatureBuilder, xgb_pipeline, required_cols: list):
        self.feature_builder = feature_builder
        self.xgb = xgb_pipeline
        self.required_cols = list(required_cols)

    def predict(self, X: pd.DataFrame):
        X_trans = self.feature_builder.transform(X)
        X_trans = _ensure_columns(X_trans, self.required_cols)
        X_trans = X_trans[self.required_cols]
        
        pred_log = self.xgb.predict(X_trans)
        return np.expm1(pred_log)

    def __getattr__(self, name):
        return getattr(self.xgb, name)


class EnsembleWithFeatures:
    def __init__(self, feature_builder: InferenceFeatureBuilder, xgb_pipeline, xgb_required_cols: list,
                 cat_model, cat_feature_cols: list, cat_cols: list, num_medians,
                 lgb_model=None, lgb_preprocessor=None,
                 rf_model=None, rf_preprocessor=None,
                 meta_model=None, weights=(0.25, 0.25, 0.25, 0.25),
                 bias_ratio=1.0):
        self.feature_builder = feature_builder
        self.xgb = xgb_pipeline
        self.xgb_required_cols = list(xgb_required_cols)

        self.cat = cat_model
        self.cat_feature_cols = list(cat_feature_cols)
        self.cat_cols = list(cat_cols)
        self.num_medians = num_medians

        self.lgb = lgb_model
        self.lgb_preprocessor = lgb_preprocessor
        self.rf = rf_model
        self.rf_preprocessor = rf_preprocessor
        self.meta_model = meta_model  # Ridge stacking meta-learner
        self.weights = weights  # fallback weights
        self.bias_ratio = bias_ratio  # log-target bias correction

    def _prep_cat(self, X_feat: pd.DataFrame):
        X_cb, _, cat_idx, _ = prepare_catboost_frame(
            X_feat, feature_cols=self.cat_feature_cols, cat_cols=self.cat_cols, num_medians=self.num_medians
        )
        return Pool(X_cb, cat_features=cat_idx)

    def _get_base_preds_log(self, Xf):
        """4 base modelden log tahminleri al."""
        Xf_xgb = _ensure_columns(Xf, self.xgb_required_cols)
        pred_xgb_log = self.xgb.predict(Xf_xgb)
        pred_cat_log = self.cat.predict(self._prep_cat(Xf))
        if self.lgb is not None and self.lgb_preprocessor is not None:
            Xf_lgb = _ensure_columns(Xf, self.xgb_required_cols)
            Xf_lgb_proc = self.lgb_preprocessor.transform(Xf_lgb)
            pred_lgb_log = self.lgb.predict(Xf_lgb_proc)
        else:
            pred_lgb_log = pred_xgb_log  # fallback
        if self.rf is not None and self.rf_preprocessor is not None:
            Xf_rf = _ensure_columns(Xf, self.xgb_required_cols)
            Xf_rf_proc = self.rf_preprocessor.transform(Xf_rf)
            pred_rf_log = self.rf.predict(Xf_rf_proc)
        else:
            pred_rf_log = pred_xgb_log  # fallback
        return pred_xgb_log, pred_cat_log, pred_lgb_log, pred_rf_log

    def predict(self, X: pd.DataFrame):
        Xf = self.feature_builder.transform(X)
        pred_xgb_log, pred_cat_log, pred_lgb_log, pred_rf_log = self._get_base_preds_log(Xf)
        
        if self.meta_model is not None:
            # Stacking: meta-learner combines log predictions
            stack_features = np.column_stack([pred_xgb_log, pred_cat_log, pred_lgb_log, pred_rf_log])
            pred_log_meta = self.meta_model.predict(stack_features)
            return np.expm1(pred_log_meta) * self.bias_ratio
        else:
            # Fallback: weighted blending in TL space
            w_xgb, w_cat, w_lgb, w_rf = self.weights
            pred_xgb_tl = np.expm1(pred_xgb_log)
            pred_cat_tl = np.expm1(pred_cat_log)
            pred_lgb_tl = np.expm1(pred_lgb_log)
            pred_rf_tl = np.expm1(pred_rf_log)
            return w_xgb * pred_xgb_tl + w_cat * pred_cat_tl + w_lgb * pred_lgb_tl + w_rf * pred_rf_tl


# =========================================================
# TRAIN
# =========================================================
def main_train():
    # Load data
    df = load_all_cities()
    X, y_tl = build_X_y(df)

    # ---- OUTLIER REMOVAL ----
    X, y_tl = remove_outliers(X, y_tl, max_price=55_000_000, max_net_m2=1000)
    X = X.reset_index(drop=True)
    y_tl = y_tl.reset_index(drop=True)

    # split
    il_series = X["Il"].astype(str)
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=il_series)

    X_train_full = X.iloc[train_idx].copy()
    y_train_full_tl = y_tl.iloc[train_idx].copy()

    X_test = X.iloc[test_idx].copy()
    y_test_tl = y_tl.iloc[test_idx].copy()
    il_test = X_test["Il"].copy()

    il_train_full = X_train_full["Il"].astype(str)
    tr_idx, va_idx = train_test_split(
        np.arange(len(X_train_full)),
        test_size=0.15,
        random_state=42,
        stratify=il_train_full
    )

    X_tr = X_train_full.iloc[tr_idx].copy()
    y_tr_tl = y_train_full_tl.iloc[tr_idx].copy()
    y_tr_log = np.log1p(y_tr_tl)

    X_va = X_train_full.iloc[va_idx].copy()
    y_va_tl = y_train_full_tl.iloc[va_idx].copy()
    y_va_log = np.log1p(y_va_tl)

    print("[INFO] Train size:", X_tr.shape[0], "| Val size:", X_va.shape[0], "| Test size:", X_test.shape[0])

    # leakage-free pm2 (TL)
    X_tr, X_va = add_leakage_free_fiyat_m2(X_tr, y_tr_tl, X_va)
    X_tr = add_advanced_features(X_tr, y_train=y_tr_tl)
    X_va = add_advanced_features(X_va, y_train=None)

    # TE (log hedef)
    for col in TARGET_ENC_COLS:
        oof_tr, oof_val = add_target_encoding_oof(
            X_tr, y_tr_log, X_va, col, n_splits=5, random_state=42, alpha=TARGET_ENC_ALPHA
        )
        if oof_tr is not None:
            te_name = f"{col}_te"
            X_tr[te_name] = oof_tr
            X_va[te_name] = oof_val

    # XGB
    preprocessor = build_preprocessor(X_tr)
    best_xgb = tune_xgboost(X_tr, y_tr_log, preprocessor)

    pred_xgb_val_log = best_xgb.predict(X_va)
    pred_xgb_val_tl = np.expm1(pred_xgb_val_log)
    print(f"[INFO] XGB Val MAE (TL): {mean_absolute_error(y_va_tl, pred_xgb_val_tl):,.0f} TL")

    # CatBoost
    cat_model, cat_feat_cols, cat_cols, num_medians = train_catboost_safe(X_tr, y_tr_log, X_va, y_va_log)
    X_va_cb, _, cat_idx, _ = prepare_catboost_frame(X_va, feature_cols=cat_feat_cols, cat_cols=cat_cols, num_medians=num_medians)
    pred_cat_val_log = cat_model.predict(Pool(X_va_cb, cat_features=cat_idx))
    pred_cat_val_tl = np.expm1(pred_cat_val_log)
    print(f"[INFO] CatBoost Val MAE (TL): {mean_absolute_error(y_va_tl, pred_cat_val_tl):,.0f} TL")

    # LightGBM
    xgb_preprocessor_fitted = best_xgb.named_steps["preprocessor"]
    lgb_model_val = train_lightgbm(X_tr, y_tr_log, X_va, y_va_log, xgb_preprocessor_fitted)
    X_va_lgb_proc = xgb_preprocessor_fitted.transform(X_va)
    pred_lgb_val_log = lgb_model_val.predict(X_va_lgb_proc)
    pred_lgb_val_tl = np.expm1(pred_lgb_val_log)
    print(f"[INFO] LightGBM Val MAE (TL): {mean_absolute_error(y_va_tl, pred_lgb_val_tl):,.0f} TL")

    # Random Forest
    rf_model_val = train_random_forest(X_tr, y_tr_log, X_va, y_va_log, xgb_preprocessor_fitted)
    pred_rf_val_log = rf_model_val.predict(xgb_preprocessor_fitted.transform(X_va))
    pred_rf_val_tl = np.expm1(pred_rf_val_log)
    print(f"[INFO] Random Forest Val MAE (TL): {mean_absolute_error(y_va_tl, pred_rf_val_tl):,.0f} TL")

    # =====================================================
    # STACKING: Ridge meta-learner on log predictions (4 models)
    # Leakage-free: CV ile alpha seçimi (validation üzerinde fit+eval aynı set olmamalı)
    # =====================================================
    print("\n[INFO] Stacking (Ridge meta-learner, 4 model) eğitimi başlıyor...")
    stack_val = np.column_stack([pred_xgb_val_log, pred_cat_val_log, pred_lgb_val_log, pred_rf_val_log])

    # Leakage-free alpha selection: KFold CV within validation set
    best_meta_mae = float("inf")
    best_meta_alpha = 1.0
    for alpha_try in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        cv_maes = []
        kf_alpha = KFold(n_splits=3, shuffle=True, random_state=42)
        for cv_tr, cv_va in kf_alpha.split(stack_val):
            meta_try = Ridge(alpha=alpha_try)
            meta_try.fit(stack_val[cv_tr], y_va_log.iloc[cv_tr])
            cv_pred_log = meta_try.predict(stack_val[cv_va])
            cv_pred_tl = np.expm1(cv_pred_log)
            cv_true_tl = np.expm1(y_va_log.iloc[cv_va])
            cv_maes.append(mean_absolute_error(cv_true_tl, cv_pred_tl))
        avg_mae = np.mean(cv_maes)
        if avg_mae < best_meta_mae:
            best_meta_mae = avg_mae
            best_meta_alpha = alpha_try

    meta_model = Ridge(alpha=best_meta_alpha)
    meta_model.fit(stack_val, y_va_log)
    pred_stack_val_log = meta_model.predict(stack_val)
    pred_stack_val_tl = np.expm1(pred_stack_val_log)
    print(f"[INFO] Stacking Val MAE (TL, CV-selected alpha): {best_meta_mae:,.0f} TL (alpha={best_meta_alpha})")
    print(f"[INFO] Stacking Val MAE (TL, refit on full val): {mean_absolute_error(y_va_tl, pred_stack_val_tl):,.0f} TL")
    print(f"[INFO] Ridge coefficients: xgb={meta_model.coef_[0]:.4f}, cat={meta_model.coef_[1]:.4f}, lgb={meta_model.coef_[2]:.4f}, rf={meta_model.coef_[3]:.4f}, intercept={meta_model.intercept_:.4f}")

    # Also compute blending for comparison
    best_weights, best_mae_val = find_best_weights_3(
        y_va_tl, pred_xgb_val_tl, pred_cat_val_tl, pred_lgb_val_tl, step=0.05
    )
    print(f"[INFO] Blending weights (xgb={best_weights[0]:.2f}, cat={best_weights[1]:.2f}, lgb={best_weights[2]:.2f}) | Val MAE = {best_mae_val:,.0f} TL")
    import sys; sys.stdout.flush()

    try:
        # full train/test features
        print("\n[INFO] Full retrain başlıyor...")
        sys.stdout.flush()
        y_train_full_log = np.log1p(y_train_full_tl)

        X_train_full2, X_test2 = add_leakage_free_fiyat_m2(X_train_full, y_train_full_tl, X_test)
        X_train_full2 = add_advanced_features(X_train_full2, y_train=y_train_full_tl)
        X_test2 = add_advanced_features(X_test2, y_train=None)

        for col in TARGET_ENC_COLS:
            tr_enc, test_enc = add_target_encoding_oof(
                X_train_full2, y_train_full_log, X_test2, col, n_splits=5, random_state=42, alpha=TARGET_ENC_ALPHA
            )
            if tr_enc is not None:
                te_name = f"{col}_te"
                X_train_full2[te_name] = tr_enc
                X_test2[te_name] = test_enc

        # XGBoost full retrain
        print("[INFO] XGBoost full retrain...")
        sys.stdout.flush()
        best_xgb.fit(X_train_full2, y_train_full_log)
        pred_xgb_test_log = best_xgb.predict(X_test2)
        pred_xgb_test_tl = np.expm1(pred_xgb_test_log)

        # CatBoost full retrain
        print("[INFO] CatBoost full retrain...")
        sys.stdout.flush()
        X_train_cb, cat_cols2, cat_idx2, num_medians2 = prepare_catboost_frame(X_train_full2)
        X_test_cb, _, _, _ = prepare_catboost_frame(X_test2, feature_cols=X_train_cb.columns.tolist(), cat_cols=cat_cols2, num_medians=num_medians2)

        train_pool_full = Pool(X_train_cb, label=y_train_full_log.values, cat_features=cat_idx2)
        test_pool = Pool(X_test_cb, cat_features=cat_idx2)

        cat_model_full = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=5000,
            learning_rate=0.03,
            depth=10,
            l2_leaf_reg=4.0,
            random_seed=42,
            allow_writing_files=False,
            verbose=250,
        )
        cat_model_full.fit(train_pool_full)
        pred_cat_test_log = cat_model_full.predict(test_pool)
        pred_cat_test_tl = np.expm1(pred_cat_test_log)

        # LightGBM full retrain
        print("[INFO] LightGBM full retrain...")
        sys.stdout.flush()
        full_preprocessor = build_preprocessor(X_train_full2)
        full_preprocessor.fit(X_train_full2)
        X_train_lgb_proc = full_preprocessor.transform(X_train_full2)
        X_test_lgb_proc = full_preprocessor.transform(X_test2)

        lgb_model_full = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            n_estimators=lgb_model_val.best_iteration_ if lgb_model_val.best_iteration_ > 0 else 3000,
            learning_rate=0.02,
            max_depth=8,
            num_leaves=127,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.3,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model_full.fit(X_train_lgb_proc, y_train_full_log)
        pred_lgb_test_log = lgb_model_full.predict(X_test_lgb_proc)
        pred_lgb_test_tl = np.expm1(pred_lgb_test_log)

        # Random Forest full retrain
        print("[INFO] Random Forest full retrain...")
        sys.stdout.flush()
        rf_model_full = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features=0.7,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        rf_model_full.fit(X_train_lgb_proc, y_train_full_log)
        pred_rf_test_log = rf_model_full.predict(X_test_lgb_proc)
        pred_rf_test_tl = np.expm1(pred_rf_test_log)

        # =====================================================
        # Stacking meta-learner on full retrained models (4 models)
        # =====================================================
        print("\n[INFO] Stacking meta-learner full OOF eğitimi (4 model)...")
        sys.stdout.flush()
        kf_meta = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_xgb_log = np.zeros(len(X_train_full2))
        oof_cat_log = np.zeros(len(X_train_full2))
        oof_lgb_log = np.zeros(len(X_train_full2))
        oof_rf_log  = np.zeros(len(X_train_full2))

        for fold_i, (tr_i, va_i) in enumerate(kf_meta.split(X_train_full2)):
            Xf_tr = X_train_full2.iloc[tr_i]
            yf_tr = y_train_full_log.iloc[tr_i]
            Xf_va = X_train_full2.iloc[va_i]

            # XGB fold
            from sklearn.base import clone as sk_clone
            xgb_fold = sk_clone(best_xgb)
            xgb_fold.fit(Xf_tr, yf_tr)
            oof_xgb_log[va_i] = xgb_fold.predict(Xf_va)

            # CatBoost fold
            Xf_tr_cb, f_cat_cols, f_cat_idx, f_num_med = prepare_catboost_frame(Xf_tr)
            Xf_va_cb, _, _, _ = prepare_catboost_frame(Xf_va, feature_cols=Xf_tr_cb.columns.tolist(), cat_cols=f_cat_cols, num_medians=f_num_med)
            cat_fold = CatBoostRegressor(
                loss_function="MAE", iterations=4000, learning_rate=0.03, depth=10,
                l2_leaf_reg=4.0, random_seed=42, allow_writing_files=False, verbose=0
            )
            cat_fold.fit(Pool(Xf_tr_cb, label=yf_tr.values, cat_features=f_cat_idx))
            oof_cat_log[va_i] = cat_fold.predict(Pool(Xf_va_cb, cat_features=f_cat_idx))

            # LightGBM fold
            pp_fold = build_preprocessor(Xf_tr)
            pp_fold.fit(Xf_tr)
            lgb_fold = lgb.LGBMRegressor(
                objective="regression_l1", n_estimators=3000, learning_rate=0.02,
                max_depth=8, num_leaves=127, subsample=0.85, colsample_bytree=0.8,
                min_child_samples=10, reg_alpha=0.3, reg_lambda=5.0, random_state=42, n_jobs=-1, verbose=-1
            )
            lgb_fold.fit(pp_fold.transform(Xf_tr), yf_tr)
            oof_lgb_log[va_i] = lgb_fold.predict(pp_fold.transform(Xf_va))

            # Random Forest fold
            rf_fold = RandomForestRegressor(
                n_estimators=500, max_depth=20, min_samples_split=5,
                min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=-1, verbose=0
            )
            rf_fold.fit(pp_fold.transform(Xf_tr), yf_tr)
            oof_rf_log[va_i] = rf_fold.predict(pp_fold.transform(Xf_va))

            print(f"  Fold {fold_i+1}/5 tamamlandı.")
            sys.stdout.flush()

        # Train final meta-learner on OOF predictions (4 models)
        stack_train = np.column_stack([oof_xgb_log, oof_cat_log, oof_lgb_log, oof_rf_log])
        meta_model_full = Ridge(alpha=best_meta_alpha)
        meta_model_full.fit(stack_train, y_train_full_log)

        oof_stack_log = meta_model_full.predict(stack_train)
        oof_stack_tl = np.expm1(oof_stack_log)
        print(f"[INFO] Stacking OOF MAE (TL): {mean_absolute_error(y_train_full_tl, oof_stack_tl):,.0f} TL")
        print(f"[INFO] Meta-learner coefs: xgb={meta_model_full.coef_[0]:.4f}, cat={meta_model_full.coef_[1]:.4f}, lgb={meta_model_full.coef_[2]:.4f}, rf={meta_model_full.coef_[3]:.4f}")

        # Log-target bias correction
        bias_ratio = np.median(y_train_full_tl.values / np.maximum(oof_stack_tl, 1.0))
        print(f"[INFO] Log-target bias correction ratio: {bias_ratio:.6f}")

        # Final test predictions via stacking (4 models)
        stack_test = np.column_stack([pred_xgb_test_log, pred_cat_test_log, pred_lgb_test_log, pred_rf_test_log])
        pred_stack_test_tl = np.expm1(meta_model_full.predict(stack_test)) * bias_ratio

        # Also compute blending test for comparison (3 model only)
        w_xgb, w_cat, w_lgb = best_weights
        pred_blend_test_tl = w_xgb * pred_xgb_test_tl + w_cat * pred_cat_test_tl + w_lgb * pred_lgb_test_tl

        print("\n[INFO] ===== TEST SONUÇLARI =====")
        evaluate_model(y_test_tl, pred_xgb_test_tl, "XGBoost (Test)")
        evaluate_model(y_test_tl, pred_cat_test_tl, "CatBoost (Test)")
        evaluate_model(y_test_tl, pred_lgb_test_tl, "LightGBM (Test)")
        evaluate_model(y_test_tl, pred_rf_test_tl, "Random Forest (Test)")
        evaluate_model(y_test_tl, pred_blend_test_tl, f"Blending (xgb={w_xgb:.2f},cat={w_cat:.2f},lgb={w_lgb:.2f})")
        evaluate_model(y_test_tl, pred_stack_test_tl, "STACKING 4-MODEL (Ridge Meta-Learner)")
        evaluate_mae_by_city(y_test_tl, pred_stack_test_tl, il_test, title="ŞEHİR BAZLI TEST MAE (STACKING 4-MODEL)")

        # save models
        fb = InferenceFeatureBuilder(te_cols=TARGET_ENC_COLS, te_alpha=TARGET_ENC_ALPHA)
        fb.fit(X_train_full, y_train_full_tl)

        xgb_required_cols = X_train_full2.columns.tolist()
        cat_feature_cols_final = X_train_cb.columns.tolist()

        xgb_wrapped = XGBModelWithFeatures(feature_builder=fb, xgb_pipeline=best_xgb, required_cols=xgb_required_cols)
        ens_wrapped = EnsembleWithFeatures(
            feature_builder=fb,
            xgb_pipeline=best_xgb,
            xgb_required_cols=xgb_required_cols,
            cat_model=cat_model_full,
            cat_feature_cols=cat_feature_cols_final,
            cat_cols=cat_cols2,
            num_medians=num_medians2,
            lgb_model=lgb_model_full,
            lgb_preprocessor=full_preprocessor,
            rf_model=rf_model_full,
            rf_preprocessor=full_preprocessor,
            meta_model=meta_model_full,
            weights=(0.25, 0.25, 0.25, 0.25),
            bias_ratio=bias_ratio,
        )

        out_xgb = str(BASE_DIR / "model_xgb_pipeline.pkl")
        out_cat = str(BASE_DIR / "model_catboost.cbm")
        out_ens = str(BASE_DIR / "emlak_fiyat_model_ensemble_xgb_cat_3sehir.pkl")

        joblib.dump(xgb_wrapped, out_xgb)
        cat_model_full.save_model(out_cat)
        joblib.dump(ens_wrapped, out_ens)

        print(f"\n[OK] XGB (feature-aware) kaydedildi: {out_xgb}")
        print(f"[OK] CatBoost kaydedildi: {out_cat}")
        print(f"[OK] 4-Model Stacking Ensemble kaydedildi: {out_ens}")

    except Exception as e:
        import traceback
        print(f"\n[HATA] Full retrain aşamasında hata oluştu:")
        traceback.print_exc()
        sys.stdout.flush()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        print("[INFO] Evaluate mode...")
        df = load_all_cities()
        X, y_tl = build_X_y(df)
        
        il_series = X["Il"].astype(str)
        idx = np.arange(len(X))
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=il_series)
        
        X_test = X.iloc[test_idx].copy()
        y_test_tl = y_tl.iloc[test_idx].copy()
        
        import joblib
        ens = joblib.load('emlak_fiyat_model_ensemble_xgb_cat_3sehir.pkl')
        preds = ens.predict(X_test)
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_test_tl, preds)
        
        print("\n\n" + "="*50)
        print(f"Test MAE (Stacking Ensemble with Baseline Corr): {mae:,.0f} TL")
        print("="*50 + "\n\n")
    else:
        if MERGE_CSV:
            merge_all_if_needed()
        main_train()