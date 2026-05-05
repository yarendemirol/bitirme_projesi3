

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import re
import difflib
import sys
import base64

# Yeni modeldeki özel sınıfların modeli yüklerken (joblib.load) 
# tanınabilmesi için eğitim dosyasından içe aktarılması
try:
    from train_model import InferenceFeatureBuilder, XGBModelWithFeatures, EnsembleWithFeatures, parse_bulundugu_kat_ordinal
except ImportError:
    pass

from sklearn.preprocessing import OneHotEncoder

st.markdown(
"""
<style>

/* =========================
   SADECE SAYFA ARKA PLAN
========================= */

.stApp {
    background: linear-gradient(135deg, #f5f8ff, #eef4ff) !important;
}

/* =========================
   SADECE YAZILAR (KONTROLLÜ)
========================= */

h1, h2, h3, h4, h5, p, label, span {
    color: #000000 !important;
}

/* Streamlit default text */
.stMarkdown, .stText, .stCaption {
    color: #000000 !important;
}

/* =========================
   KUTULAR -> BEYAZ
========================= */

/* SELECTBOX */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #ddd !important;
}

/* select iç yazı */
div[data-baseweb="select"] * {
    color: #000000 !important;
}

/* INPUT */
input, textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* NUMBER INPUT */
div[data-testid="stNumberInput"] {
    background-color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #ddd !important;
}

div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* DROPDOWN */
div[data-baseweb="popover"] {
    background-color: #ffffff !important;
}

div[role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

div[role="option"]:hover {
    background-color: #f2f2f2 !important;
}

/* =========================
   BUTTON
========================= */

.stButton>button {
    background-color: #111111 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

</style>
""",
unsafe_allow_html=True
)
# ================== DOSYA YOLLARI ==================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "emlak_fiyat_model_ensemble_xgb_cat_3sehir.pkl"

DATA_PATHS = {
    "İstanbul": BASE_DIR / "istanbul_merged_all.csv",
    "Ankara":   BASE_DIR / "ankara_merged_all.csv",
    "İzmir":    BASE_DIR / "izmir_merged_all.csv",
}



# ================== HATA ORANI ==================
ERROR_RATE = 0.1636



# ================== ŞEHİR / İLÇE (SABİT - fallback) ==================
CITY_OPTIONS = list(DATA_PATHS.keys())

# Bunlar artık sadece FALLBACK; esas ilçe listesi CSV'den gelecek.
DISTRICTS_BY_CITY = {
    "İstanbul": [
        "Adalar","Arnavutköy","Ataşehir","Avcılar","Bağcılar","Bahçelievler","Bakırköy","Başakşehir","Bayrampaşa",
        "Beşiktaş","Beykoz","Beylikdüzü","Beyoğlu","Büyükçekmece","Çatalca","Çekmeköy","Esenler","Esenyurt",
        "Eyüpsultan","Fatih","Gaziosmanpaşa","Güngören","Kadıköy","Kağıthane","Kartal","Küçükçekmece","Maltepe",
        "Pendik","Sancaktepe","Sarıyer","Silivri","Sultanbeyli","Sultangazi","Şile","Şişli","Tuzla","Ümraniye",
        "Üsküdar","Zeytinburnu"
    ],
    "Ankara": ["Çankaya","Keçiören","Yenimahalle","Mamak","Etimesgut","Sincan","Altındağ","Gölbaşı","Pursaklar","Polatlı"],
    "İzmir":  ["Konak","Karşıyaka","Bornova","Buca","Bayraklı","Balçova","Narlıdere","Gaziemir","Çiğli","Menemen"]
}


# ================== NORMALIZE ==================
def norm_tr(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace("İ", "I").replace("ı", "i")
    s = s.lower()
    s = (s.replace("ş", "s").replace("ğ", "g").replace("ü", "u")
           .replace("ö", "o").replace("ç", "c"))
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_mahalle_suffix(s: str) -> str:
    if s is None:
        return ""
    x = str(s)
    x = re.sub(r"\bmahallesi\b", "", x, flags=re.IGNORECASE)
    x = re.sub(r"\bmahalle\b", "", x, flags=re.IGNORECASE)
    x = re.sub(r"\bmh\.?\b", "", x, flags=re.IGNORECASE)
    x = re.sub(r"\bmah\.?\b", "", x, flags=re.IGNORECASE)
    x = re.sub(r"[()\-_/|,]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def find_col(df: pd.DataFrame, keywords):
    if df is None or df.empty:
        return None
    col_norm = {c: norm_tr(c) for c in df.columns}

    for kw in keywords:
        k = norm_tr(kw)
        for c, n in col_norm.items():
            if n == k:
                return c

    for kw in keywords:
        k = norm_tr(kw)
        for c, n in col_norm.items():
            if k in n:
                return c
    return None


def looks_like_mahalle_or_address(x: str) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    if not s or s == "nan":
        return False
    bad_keys = ["mahallesi", "mahalle", "mh", "mah.", "sokak", "sk", "cadde", "cd", "bulvari", "site", "apt", "apartman"]
    return any(k in s for k in bad_keys)


# ================== PARSE HELPERS ==================
def parse_price_series(sr: pd.Series) -> pd.Series:
    if sr is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(sr):
        return pd.to_numeric(sr, errors="coerce")
    s = sr.astype(str)
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace(r"[^\d,\.]", "", regex=True)
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def extract_number_series(sr: pd.Series) -> pd.Series:
    if sr is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(sr):
        return pd.to_numeric(sr, errors="coerce")
    s = sr.astype(str)
    nums = s.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(nums, errors="coerce")


def oda_total(oda_sayi_str):
    s = str(oda_sayi_str).strip().lower()
    if "stüdyo" in s or "studio" in s:
        return 1.0
    nums = re.findall(r"\d+", s)
    if not nums:
        return np.nan
    if "+" in s:
        return float(sum(int(n) for n in nums))
    return float(nums[0])


def parse_loc_to_ilce_mahalle(loc_val):
    """
    'İzmir - Aliağa - Yeni Mahallesi' -> ('Aliağa','Yeni')
    """
    if loc_val is None or (isinstance(loc_val, float) and np.isnan(loc_val)):
        return (np.nan, np.nan)
    s = str(loc_val)
    parts = re.split(r"\s*-\s*|/|,|\||>", s)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) >= 3:
        ilce = parts[1]
        mah = parts[2]
    elif len(parts) == 2:
        ilce = parts[1]
        mah = np.nan
    else:
        return (np.nan, np.nan)

    mah = strip_mahalle_suffix(mah) if isinstance(mah, str) else mah
    return (ilce, mah)


def safe_unique(df, col, fallback_list):
    if df is None or df.empty or col not in df.columns:
        return fallback_list
    vals = df[col].dropna().astype(str).tolist()
    vals = [v.strip() for v in vals if v and v.strip() and v.strip().lower() != "nan"]
    vals = list(dict.fromkeys(vals))
    return vals if vals else fallback_list


# ================== MODEL/DATA LOAD ==================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def _read_csv_smart(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] == 1:
        df2 = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        if df2.shape[1] > 1:
            df = df2
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _prepare_df(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    ilce_col = find_col(df, ["ilce", "ilçe", "district"])
    mahalle_col = find_col(df, ["mahalle", "semt", "neighborhood"])
    il_col = find_col(df, ["il", "şehir", "sehir", "city"])

    if ilce_col and ilce_col != "Ilce":
        df = df.rename(columns={ilce_col: "Ilce"})
    if mahalle_col and mahalle_col != "Mahalle":
        df = df.rename(columns={mahalle_col: "Mahalle"})
    if il_col and il_col != "Il":
        df = df.rename(columns={il_col: "Il"})

    df["Il"] = city_name  # şehir garanti

    loc_col = find_col(df, ["lokasyon", "konum", "location", "adres", "address"])
    if loc_col:
        if "Ilce" not in df.columns:
            df["Ilce"] = np.nan
        if "Mahalle" not in df.columns:
            df["Mahalle"] = np.nan

        parsed = df[loc_col].apply(parse_loc_to_ilce_mahalle)
        parsed = pd.DataFrame(parsed.tolist(), columns=["_ilce_p", "_mah_p"], index=df.index)

        df["Ilce"] = df["Ilce"].fillna(parsed["_ilce_p"])
        df["Mahalle"] = df["Mahalle"].fillna(parsed["_mah_p"])

    # Fiyat_TL / Net_m2 / Fiyat_m2
    if "Fiyat_TL" not in df.columns:
        fiyat_col = find_col(df, ["fiyat", "price"])
        df["Fiyat_TL"] = parse_price_series(df[fiyat_col]) if fiyat_col else np.nan
    else:
        df["Fiyat_TL"] = pd.to_numeric(df["Fiyat_TL"], errors="coerce")

    if "Net_m2" not in df.columns:
        net_col = find_col(df, ["net metrekare", "net m2", "net_m2", "net"])
        df["Net_m2"] = extract_number_series(df[net_col]) if net_col else np.nan
    else:
        df["Net_m2"] = pd.to_numeric(df["Net_m2"], errors="coerce")

    if "Fiyat_m2" not in df.columns and ("Fiyat_TL" in df.columns) and ("Net_m2" in df.columns):
        df["Fiyat_m2"] = df["Fiyat_TL"] / df["Net_m2"]
    df["Fiyat_m2"] = pd.to_numeric(df.get("Fiyat_m2"), errors="coerce")

    # string temizliği
    for c in ["Ilce", "Mahalle", "Il", "Isıtma Tipi", "Isitma Tipi", "Tapu Durumu", "Kullanım Durumu", "Kullanim Durumu",
              "Bulunduğu Kat", "Bulundugu Kat", "Takas", "Türü", "Turu", "Kategorisi", "Tipi", "ParseStatus"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    df["Il_norm"] = df["Il"].astype("string").fillna("").apply(norm_tr) if "Il" in df.columns else ""
    df["Ilce_norm"] = df["Ilce"].astype("string").fillna("").apply(norm_tr) if "Ilce" in df.columns else ""

    if "Mahalle" in df.columns:
        df["Mahalle_clean"] = df["Mahalle"].astype("string").fillna("").apply(strip_mahalle_suffix)
        df["Mahalle_norm"] = df["Mahalle_clean"].astype("string").fillna("").apply(norm_tr)
    else:
        df["Mahalle_clean"] = ""
        df["Mahalle_norm"] = ""

    return df


@st.cache_data
def load_all_data():
    dfs = []
    for city, path in DATA_PATHS.items():
        try:
            df = _read_csv_smart(path)
            df = _prepare_df(df, city)
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


model = load_model()
df_all = load_all_data()


# ================== MODELİN BEKLEDİĞİ FEATURE KOLONLARI ==================
if hasattr(model, "xgb") and hasattr(model.xgb, "named_steps"):
    pre = model.xgb.named_steps.get("preprocessor")
elif hasattr(model, "named_steps"):
    pre = model.named_steps.get("preprocessor")
else:
    pre = model

expected_cols = []
if hasattr(pre, "transformers_"):
    for _, _, cols in pre.transformers_:
        if cols is None:
            continue
        expected_cols.extend(list(cols))
X_columns = list(dict.fromkeys(expected_cols))

# Yeni model pipeline'ında eksilen veya OneHotEncoding dışında kalan
# mecburi ham özelliklerin tahmin matrisinde kesin var olmasını sağlıyoruz:
raw_features = ["Ilce", "Mahalle", "Il_Ilce", "Il_Ilce_Mahalle", "Net_m2", "Brut_m2", "Oda_sayi", 
                "Banyo_sayi", "Bina_yas_yil", "Bina_kat_sayisi", "Kat_ordinal", "Fiyat_m2",
                "Demo_Toplam_Nufus_num", "Demo_Egitim_Durumu_pct", "Demo_Ortalama_Yas_num",
                "Demo_Medeni_Hal_pct", "Demo_Sosyo_Ekonomik_Statu"]
for rc in raw_features:
    if rc not in X_columns:
        X_columns.append(rc)


def resolve_col(x_cols, keywords):
    if not x_cols:
        return None
    norms = {c: norm_tr(c) for c in x_cols}
    key_norms = [norm_tr(k) for k in keywords]

    for c, n in norms.items():
        if n in key_norms:
            return c

    for c, n in norms.items():
        for kn in key_norms:
            if kn and kn in n:
                return c
    return None


COL_IL      = resolve_col(X_columns, ["il", "şehir", "sehir", "city"])
COL_ILCE    = resolve_col(X_columns, ["ilce", "ilçe", "district"])
COL_MAHALLE = resolve_col(X_columns, ["mahalle", "semt", "neighborhood"])

COL_TIPI    = resolve_col(X_columns, ["tipi"])
COL_TURU    = resolve_col(X_columns, ["türü", "turu"])
COL_KAT     = resolve_col(X_columns, ["bulundugu kat", "bulunduğu kat"])
COL_TAPU    = resolve_col(X_columns, ["tapu durumu"])
COL_ISITMA  = resolve_col(X_columns, ["isitma tipi", "ısıtma tipi"])
COL_KULLAN  = resolve_col(X_columns, ["kullanim durumu", "kullanım durumu"])
COL_TAKAS   = resolve_col(X_columns, ["takas"])

COL_NETM2   = resolve_col(X_columns, ["net_m2", "net m2", "net"])
COL_BRUTM2  = resolve_col(X_columns, ["brut_m2", "brüt_m2", "brut"])
COL_ODA     = resolve_col(X_columns, ["oda_sayi", "oda"])
COL_BANYO   = resolve_col(X_columns, ["banyo_sayi", "banyo"])
COL_YAS     = resolve_col(X_columns, ["bina_yas_yil", "yas"])
COL_KATSAY  = resolve_col(X_columns, ["bina_kat_sayisi", "kat sayisi", "kat sayısı"])
COL_SITEF   = resolve_col(X_columns, ["site_icinde_flag", "site icinde"])
COL_KREDI   = resolve_col(X_columns, ["kredi_uygun_flag", "krediye uygunluk", "kredi"])
COL_FIYATM2 = resolve_col(X_columns, ["fiyat_m2", "fiyat m2"])

COL_PARSE   = resolve_col(X_columns, ["parsestatus"])
COL_SOURCE  = resolve_col(X_columns, ["__source_file"])
COL_ERROR   = resolve_col(X_columns, ["error"])


# ================== OHE KATEGORİLERİNİ SAĞLAM ÇEK ==================
def _extract_ohe_from_pipeline(trans):
    if hasattr(trans, "named_steps"):
        for _, step in trans.named_steps.items():
            if isinstance(step, OneHotEncoder) or hasattr(step, "categories_"):
                if hasattr(step, "categories_"):
                    return step
    if isinstance(trans, OneHotEncoder) and hasattr(trans, "categories_"):
        return trans
    return None


def get_cat_cols_and_ohe(preprocessor):
    if not hasattr(preprocessor, "transformers_"):
        return [], None

    for _, trans, cols in preprocessor.transformers_:
        ohe = _extract_ohe_from_pipeline(trans)
        if ohe is not None and hasattr(ohe, "categories_"):
            cols_list = list(cols) if cols is not None and not isinstance(cols, slice) else []
            return cols_list, ohe

    return [], None


CAT_COLS, OHE = get_cat_cols_and_ohe(pre)

OHE_CATS = {}
if OHE is not None and CAT_COLS:
    n = min(len(CAT_COLS), len(OHE.categories_))
    for i in range(n):
        c = str(CAT_COLS[i])
        OHE_CATS[c] = [str(x) for x in OHE.categories_[i]]


def build_canon_map(colname: str, suffix_clean=False):
    cats = OHE_CATS.get(colname, [])
    canon = {}
    for v in cats:
        canon[norm_tr(v)] = v
        if suffix_clean:
            vv = strip_mahalle_suffix(v)
            if vv:
                canon[norm_tr(vv)] = v
    return canon


IL_CANON      = build_canon_map(str(COL_IL)) if COL_IL else {}
ILCE_CANON    = build_canon_map(str(COL_ILCE)) if COL_ILCE else {}
MAHALLE_CANON = build_canon_map(str(COL_MAHALLE), suffix_clean=True) if COL_MAHALLE else {}

TAPU_CANON   = build_canon_map(str(COL_TAPU)) if COL_TAPU else {}
ISITMA_CANON = build_canon_map(str(COL_ISITMA)) if COL_ISITMA else {}
KULLAN_CANON = build_canon_map(str(COL_KULLAN)) if COL_KULLAN else {}
KAT_CANON    = build_canon_map(str(COL_KAT)) if COL_KAT else {}
TAKAS_CANON  = build_canon_map(str(COL_TAKAS)) if COL_TAKAS else {}
TIPI_CANON   = build_canon_map(str(COL_TIPI)) if COL_TIPI else {}
TURU_CANON   = build_canon_map(str(COL_TURU)) if COL_TURU else {}


def canonicalize(val, canon_map, cutoff=0.90, allow_none=True, mahalle_mode=False):
    if val is None or str(val).strip() in ["", "Seçilmedi"]:
        return (np.nan if allow_none else ""), False

    v = str(val).strip()
    if mahalle_mode:
        v = strip_mahalle_suffix(v)

    if not canon_map:
        return v, False

    key = norm_tr(v)
    if key in canon_map:
        return canon_map[key], True

    best = difflib.get_close_matches(key, list(canon_map.keys()), n=1, cutoff=cutoff)
    if best:
        return canon_map[best[0]], True

    return v, False


# ================== Fiyat_m2 MEDYAN ==================
@st.cache_data
def build_fiyat_m2_maps(df: pd.DataFrame):
    if df is None or df.empty or "Fiyat_m2" not in df.columns:
        return {}, {}, {}, np.nan

    dfx = df.copy()
    dfx["Fiyat_m2"] = pd.to_numeric(dfx["Fiyat_m2"], errors="coerce")
    dfx = dfx[dfx["Fiyat_m2"].notna()]

    global_med = float(dfx["Fiyat_m2"].median()) if not dfx.empty else np.nan

    city_med = {}
    ilce_med = {}
    mah_med = {}

    if "Il_norm" in dfx.columns:
        city_med = dfx.groupby("Il_norm")["Fiyat_m2"].median().to_dict()

    if "Il_norm" in dfx.columns and "Ilce_norm" in dfx.columns:
        ilce_med = dfx.groupby(["Il_norm", "Ilce_norm"])["Fiyat_m2"].median().to_dict()

    if "Il_norm" in dfx.columns and "Ilce_norm" in dfx.columns and "Mahalle_norm" in dfx.columns:
        mah_med = dfx.groupby(["Il_norm", "Ilce_norm", "Mahalle_norm"])["Fiyat_m2"].median().to_dict()

    return city_med, ilce_med, mah_med, global_med


CITY_FM2, ILCE_FM2, MAH_FM2, GLOBAL_FM2 = build_fiyat_m2_maps(df_all)


# ================== DEMOGRAFİK VERİ LOOKUP ==================
def _parse_demo_numeric(val) -> float:
    """Demografik string'den ilk sayıyı çeker. Ör: '17.235 - Normal' -> 17.235, '%49 Evli' -> 49"""
    if pd.isna(val):
        return np.nan
    s = str(val).replace("%", "").replace(",", ".")
    nums = re.findall(r"[0-9]+\.?[0-9]*", s)
    if not nums:
        return np.nan
    try:
        return float(nums[0])
    except Exception:
        return np.nan


@st.cache_data
def build_demo_lookup(df: pd.DataFrame):
    """
    CSV verisinden Il+Ilce+Mahalle bazlı demografik lookup tabloları oluşturur.
    Fallback zinciri: Il+Ilce+Mahalle -> Il+Ilce -> Il -> global
    """
    DEMO_RAW = ["Demo_Toplam_Nufus", "Demo_Ortalama_Yas", "Demo_Medeni_Hal",
                "Demo_Sosyo_Ekonomik_Statu", "Demo_Egitim_Durumu"]
    DEMO_OUT = ["Demo_Toplam_Nufus_num", "Demo_Ortalama_Yas_num", "Demo_Medeni_Hal_pct",
                "Demo_Sosyo_Ekonomik_Statu", "Demo_Egitim_Durumu_pct"]
    # Sadece sayısal olan sütunlar (median hesaplanabilir)
    DEMO_NUMERIC = ["Demo_Toplam_Nufus_num", "Demo_Ortalama_Yas_num", "Demo_Medeni_Hal_pct",
                    "Demo_Egitim_Durumu_pct"]

    empty = {c: np.nan for c in DEMO_OUT}
    if df is None or df.empty:
        return {}, {}, {}, empty

    # Gerekli sütunlar var mı kontrol
    missing = [c for c in DEMO_RAW if c not in df.columns]
    if missing:
        return {}, {}, {}, empty

    dfx = df.dropna(subset=DEMO_RAW[:1]).copy()  # en az Nufus doluysa al
    if dfx.empty:
        return {}, {}, {}, empty

    # Parsed sayısal sütunlar oluştur
    dfx["Demo_Toplam_Nufus_num"] = dfx["Demo_Toplam_Nufus"].apply(_parse_demo_numeric)
    dfx["Demo_Egitim_Durumu_pct"] = dfx["Demo_Egitim_Durumu"].apply(_parse_demo_numeric)
    dfx["Demo_Ortalama_Yas_num"] = dfx["Demo_Ortalama_Yas"].apply(_parse_demo_numeric)
    dfx["Demo_Medeni_Hal_pct"] = dfx["Demo_Medeni_Hal"].apply(_parse_demo_numeric)
    dfx["Demo_Sosyo_Ekonomik_Statu"] = dfx["Demo_Sosyo_Ekonomik_Statu"].astype(str)

    # norm key oluştur
    dfx["_il_n"] = dfx["Il"].astype(str).fillna("").apply(norm_tr)
    dfx["_ilce_n"] = dfx["Ilce"].astype(str).fillna("").apply(norm_tr)
    dfx["_mah_n"] = dfx["Mahalle"].astype(str).fillna("").apply(strip_mahalle_suffix).apply(norm_tr)

    # --- Mahalle düzeyi (Il+Ilce+Mahalle) ---
    mah_grp = dfx.groupby(["_il_n", "_ilce_n", "_mah_n"])[DEMO_OUT].first()
    mah_map = {idx: row.to_dict() for idx, row in mah_grp.iterrows()}

    # --- İlçe düzeyi (Il+Ilce) ---
    ilce_grp = dfx.groupby(["_il_n", "_ilce_n"])[DEMO_NUMERIC].median()  # sayısal medyan
    ilce_ses = dfx.groupby(["_il_n", "_ilce_n"])["Demo_Sosyo_Ekonomik_Statu"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "C")
    ilce_map = {}
    for idx in ilce_grp.index:
        d = ilce_grp.loc[idx].to_dict()
        d["Demo_Sosyo_Ekonomik_Statu"] = ilce_ses.get(idx, "C")
        ilce_map[idx] = d

    # --- Şehir düzeyi (Il) ---
    il_grp = dfx.groupby("_il_n")[DEMO_NUMERIC].median()
    il_ses = dfx.groupby("_il_n")["Demo_Sosyo_Ekonomik_Statu"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "C")
    il_map = {}
    for idx in il_grp.index:
        d = il_grp.loc[idx].to_dict()
        d["Demo_Sosyo_Ekonomik_Statu"] = il_ses.get(idx, "C")
        il_map[idx] = d

    # --- Global fallback ---
    global_vals = {}
    for c in DEMO_NUMERIC:
        global_vals[c] = float(dfx[c].median()) if dfx[c].notna().any() else np.nan
    ses_mode = dfx["Demo_Sosyo_Ekonomik_Statu"].mode()
    global_vals["Demo_Sosyo_Ekonomik_Statu"] = ses_mode.iloc[0] if len(ses_mode) > 0 else "C"

    return mah_map, ilce_map, il_map, global_vals


def get_demo_values(il, ilce, mahalle):
    """
    Fallback zinciri ile demografik verileri çeker:
    Il+Ilce+Mahalle -> Il+Ilce -> Il -> global
    """
    il_n = norm_tr(il) if il else ""
    ilce_n = norm_tr(ilce) if ilce else ""
    mah_n = norm_tr(strip_mahalle_suffix(mahalle)) if mahalle and str(mahalle).strip() not in ["", "Seçilmedi"] else ""

    DEMO_OUT = ["Demo_Toplam_Nufus_num", "Demo_Ortalama_Yas_num", "Demo_Medeni_Hal_pct",
                "Demo_Sosyo_Ekonomik_Statu", "Demo_Egitim_Durumu_pct"]

    # 1) Mahalle düzeyi
    if mah_n:
        vals = DEMO_MAH_MAP.get((il_n, ilce_n, mah_n))
        if vals:
            return vals

    # 2) İlçe düzeyi
    if ilce_n:
        vals = DEMO_ILCE_MAP.get((il_n, ilce_n))
        if vals:
            return vals

    # 3) Şehir düzeyi
    if il_n:
        vals = DEMO_IL_MAP.get(il_n)
        if vals:
            return vals

    # 4) Global fallback
    return DEMO_GLOBAL


DEMO_MAH_MAP, DEMO_ILCE_MAP, DEMO_IL_MAP, DEMO_GLOBAL = build_demo_lookup(df_all)


# ================== İLÇE LİSTESİ (CSV'den) ==================
def clean_ilce_value(v: str) -> str | None:
    """
    Ilce hücresinde bazen 'İzmir - Aliağa - Yeni Mahallesi' gibi string olabiliyor.
    Bu fonksiyon bunu 'Aliağa'ya indirger ve mahalle/adres gibi şeyleri elemek için filtre uygular.
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None

    # Eğer komple lokasyon gibi geldiyse split ile ilçeyi çek
    if " - " in s:
        parts = [p.strip() for p in s.split(" - ") if p.strip()]
        if len(parts) >= 2:
            s = parts[1]

    # İlçe kelimesi/sonek temizliği
    s = re.sub(r"\b(ilcesi|ilçesi)\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+", " ", s).strip()

    # Mahalle/adres gibi görünüyorsa at
    if looks_like_mahalle_or_address(s):
        return None

    # Çok uzun ve lokasyon gibi ise at (ekstra güvenlik)
    if len(s) > 40:
        return None

    return s


def get_ilce_options_from_csv(city: str, df_city: pd.DataFrame) -> list[str]:
    # fallback sabit liste
    base = DISTRICTS_BY_CITY.get(city, []).copy()

    csv_list = []
    if df_city is not None and (not df_city.empty) and ("Ilce" in df_city.columns):
        raw = df_city["Ilce"].dropna().astype(str).tolist()
        for v in raw:
            c = clean_ilce_value(v)
            if c:
                csv_list.append(c)

    # unique (norm key ile)
    merged = {}
    for x in base:
        merged[norm_tr(x)] = x
    for x in csv_list:
        merged[norm_tr(x)] = x  # csv öncelikli

    out = list(merged.values())
    out = [x for x in out if x and x.strip()]
    out = sorted(out, key=lambda z: norm_tr(z))
    return out if out else base


# ================== INPUT ROW ==================
def build_input_row(
    sehir, ilce, mahalle,
    net_m2, brut_m2, oda_sayi_str, banyo_sayi,
    bina_yas, bina_kat_sayisi,
    tapu_durumu,
    bulundugu_kat, isitma_tipi,
    kredi_uygun, site_icinde,
    turu, kategorisi, tipi,
    ozellikler=None
):
    row = {col: np.nan for col in X_columns}

    il_val, il_known = canonicalize(sehir, IL_CANON, cutoff=0.92, allow_none=False)
    ilce_val, ilce_known = canonicalize(ilce, ILCE_CANON, cutoff=0.92, allow_none=False)
    mah_val, mah_known = canonicalize(mahalle, MAHALLE_CANON, cutoff=0.78, allow_none=True, mahalle_mode=True)

    tapu_val, tapu_known = canonicalize(tapu_durumu, TAPU_CANON, cutoff=0.88, allow_none=False)
    isitma_val, isitma_known = canonicalize(isitma_tipi, ISITMA_CANON, cutoff=0.82, allow_none=False)
    kat_val, kat_known = canonicalize(bulundugu_kat, KAT_CANON, cutoff=0.82, allow_none=False)
    tipi_val, tipi_known = canonicalize(tipi, TIPI_CANON, cutoff=0.82, allow_none=False)
    turu_val, turu_known = canonicalize(turu, TURU_CANON, cutoff=0.82, allow_none=False)

    if COL_IL in row:
        row[COL_IL] = il_val
    if COL_ILCE in row:
        row[COL_ILCE] = ilce_val
    if COL_MAHALLE in row:
        row[COL_MAHALLE] = (np.nan if str(mahalle).strip() in ["", "Seçilmedi"] else mah_val)

    if COL_NETM2 in row:
        row[COL_NETM2] = float(net_m2)
    if COL_BRUTM2 in row:
        row[COL_BRUTM2] = float(brut_m2)
    if COL_ODA in row:
        row[COL_ODA] = oda_total(oda_sayi_str)
    if COL_BANYO in row:
        row[COL_BANYO] = int(banyo_sayi)
    if COL_YAS in row:
        row[COL_YAS] = int(bina_yas)
    if COL_KATSAY in row:
        row[COL_KATSAY] = int(bina_kat_sayisi)

    if "Kat_ordinal" in row and "parse_bulundugu_kat_ordinal" in globals():
        row["Kat_ordinal"] = parse_bulundugu_kat_ordinal(bulundugu_kat)
        
    if COL_KAT in row:
        row[COL_KAT] = kat_val
    if COL_ISITMA in row:
        row[COL_ISITMA] = isitma_val
    if COL_TAPU in row:
        row[COL_TAPU] = tapu_val
    if COL_TIPI in row:
        row[COL_TIPI] = tipi_val
    if COL_TURU in row:
        row[COL_TURU] = turu_val

    if COL_KREDI in row:
        row[COL_KREDI] = 1 if kredi_uygun == "Uygun" else (0 if kredi_uygun == "Uygun Değil" else np.nan)

    if COL_SITEF in row:
        row[COL_SITEF] = 1 if site_icinde == "Evet" else 0

    # ---- Demografik verileri otomatik doldur (Il+Ilce+Mahalle lookup) ----
    demo_vals = get_demo_values(il_val, ilce_val, mahalle)
    for demo_col, demo_val in demo_vals.items():
        if demo_col in row:
            row[demo_col] = demo_val

    # ---- Lüks / Ek Özellik flagları (Metin olarak yansıtma) ----
    ilan_oz = []
    aciklama_kw = []
    if ozellikler:
        if ozellikler.get("oz_asansor"): ilan_oz.append("asansor")
        if ozellikler.get("oz_otopark"): ilan_oz.append("otopark")
        if ozellikler.get("oz_guvenlik"): ilan_oz.append("guvenlik")
        if ozellikler.get("oz_havuz"): ilan_oz.append("havuz")
        if ozellikler.get("oz_balkon"): ilan_oz.append("balkon")
        if ozellikler.get("oz_teras"): ilan_oz.append("teras")
        if ozellikler.get("oz_manzara"): ilan_oz.append("manzara")
        if ozellikler.get("oz_deniz"): ilan_oz.append("deniz")
        if ozellikler.get("kw_deniz_manzarasi"): aciklama_kw.append("deniz manzarasi")
        if ozellikler.get("oz_ebeveyn"): ilan_oz.append("ebeveyn banyo")

    # Modeli ezmemesi için ham metin sütunlarını dolduruyoruz
    if "Ilan_Ozellikleri" in row:
        row["Ilan_Ozellikleri"] = "|".join(ilan_oz)
    if "Aciklama" in row:
        row["Aciklama"] = " ".join(aciklama_kw)
    if "Baslik" in row:
        row["Baslik"] = " ".join(aciklama_kw)

    if COL_FIYATM2 in row:
        il_n = norm_tr(il_val)
        ilce_n = norm_tr(ilce_val)
        mah_n = norm_tr(strip_mahalle_suffix(mah_val)) if isinstance(mah_val, str) else norm_tr(mahalle)

        v = np.nan
        if str(mahalle).strip() not in ["", "Seçilmedi"]:
            v = MAH_FM2.get((il_n, ilce_n, mah_n), np.nan)
        if pd.isna(v):
            v = ILCE_FM2.get((il_n, ilce_n), np.nan)
        if pd.isna(v):
            v = CITY_FM2.get(il_n, GLOBAL_FM2)

        row[COL_FIYATM2] = v

    if COL_PARSE in row:
        row[COL_PARSE] = "OK"
    if COL_SOURCE in row:
        row[COL_SOURCE] = "streamlit"
    if COL_ERROR in row:
        row[COL_ERROR] = 0.0

    dbg = {
        "col_il": COL_IL, "col_ilce": COL_ILCE, "col_mahalle": COL_MAHALLE,
        "col_isitma": COL_ISITMA, "col_tapu": COL_TAPU,

        "il_ui": sehir, "il_sent": il_val, "il_known": il_known, "il_cat_count": len(IL_CANON),
        "ilce_ui": ilce, "ilce_sent": ilce_val, "ilce_known": ilce_known, "ilce_cat_count": len(ILCE_CANON),
        "mah_ui": mahalle, "mah_sent": mah_val, "mah_known": mah_known, "mah_cat_count": len(MAHALLE_CANON),
    }

    return pd.DataFrame([row]), dbg


# ================== HATA KUTUSU + ÇUBUK ==================
def render_error_box_and_bar(pred: float, rate: float):
    margin = max(0.0, pred * rate)
    low = max(0.0, pred - margin)
    high = pred + margin

    st.markdown(
        f"""
        <div style="border:1px solid #ff4b4b; padding:14px; border-radius:14px;
                    background: rgba(255,75,75,0.08); margin-top:10px;">
          <div style="font-size:1.00rem; color:#ff4b4b; font-weight:800;">Tahmini Hata Payı (±)</div>
          <div style="font-size:1.25rem; color:#ffffff; margin-top:8px;">
            {low:,.0f} TL — {high:,.0f} TL
          </div>
          <div style="font-size:0.90rem; color:#b0b0b0; margin-top:8px;">
            ±{margin:,.0f} TL (yaklaşık ±{rate*100:.2f}%)
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    scale_min = max(0.0, pred - 2 * margin)
    scale_max = pred + 2 * margin
    denom = (scale_max - scale_min) if (scale_max > scale_min) else 1.0

    low_pct = (low - scale_min) / denom * 100
    high_pct = (high - scale_min) / denom * 100
    pred_pct = (pred - scale_min) / denom * 100

    st.markdown(
        f"""
        <div style="margin-top:12px;">
          <div style="position:relative; height:12px; background:rgba(255,255,255,0.10); border-radius:999px;">
            <div style="position:absolute; left:{low_pct:.2f}%; width:{(high_pct-low_pct):.2f}%;
                        height:12px; background:rgba(255,75,75,0.70); border-radius:999px;"></div>
            <div style="position:absolute; left:calc({pred_pct:.2f}% - 1px); width:2px; height:18px; top:-3px;
                        background:#ffffff;"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#b0b0b0; margin-top:6px;">
            <span>{scale_min:,.0f} TL</span>
            <span>{scale_max:,.0f} TL</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ================== UI ==================
st.title("🏠 Satılık Konut Fiyat Tahmin Sistemi")
st.markdown("## 🏙️ Şehir Seçimi")
st.write("İl/İlçe, metrekare ve temel özelliklere göre **tahmini satış fiyatını** hesaplayan bir sistem.")
st.markdown("---")

import base64

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


col1, col2, col3 = st.columns(3)

with col1:
    img = get_image_base64("assets/istanbul.jpg")
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/jpg;base64,{img}" style="width:100%; border-radius:15px;">
            <h3>İstanbul</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    img = get_image_base64("assets/ankara.jpg")
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/jpg;base64,{img}" style="width:100%; border-radius:15px;">
            <h3>Ankara</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    img = get_image_base64("assets/izmir.jpg")
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/jpg;base64,{img}" style="width:100%; border-radius:15px;">
            <h3>İzmir</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
sehir = st.selectbox("Şehir", options=CITY_OPTIONS, index=0)

df_city = df_all
if df_all is not None and (not df_all.empty) and ("Il_norm" in df_all.columns):
    df_city = df_all.loc[df_all["Il_norm"] == norm_tr(sehir)].copy()

# ilce secenekleri CSV'den cekilir
ilce_list = get_ilce_options_from_csv(sehir, df_city)
if not ilce_list:
    ilce_list = ["Seçilmedi"]

ilce = st.selectbox("İlçe", options=ilce_list, index=0)

# Mahalle listesi: şehir+ilçe filtreli veriden
mahalle_opts = []
if (
    df_city is not None and (not df_city.empty)
    and ("Ilce_norm" in df_city.columns)
    and ("Mahalle_clean" in df_city.columns)
):
    ilce_key = norm_tr(ilce)
    sub = df_city.loc[df_city["Ilce_norm"] == ilce_key, "Mahalle_clean"]
    mahalle_opts = sub.dropna().astype(str).map(str.strip).tolist()
    mahalle_opts = [m for m in mahalle_opts if m and m.strip() and m.strip().lower() != "nan"]
    mahalle_opts = list(dict.fromkeys(mahalle_opts))
    mahalle_opts = sorted(mahalle_opts, key=lambda z: norm_tr(z))

# Eğer ilçede mahalle yoksa şehir geneline düş
if len(mahalle_opts) == 0 and df_city is not None and (not df_city.empty) and ("Mahalle_clean" in df_city.columns):
    all_city = df_city["Mahalle_clean"].dropna().astype(str).map(str.strip).tolist()
    all_city = [m for m in all_city if m and m.strip() and m.strip().lower() != "nan"]
    mahalle_opts = sorted(list(dict.fromkeys(all_city)), key=lambda z: norm_tr(z))

only_known = st.checkbox("Sadece modelin tanıdığı mahalleleri göster", value=False)

if only_known and MAHALLE_CANON and len(mahalle_opts) > 0:
    filtered = []
    for m in mahalle_opts:
        _, known = canonicalize(m, MAHALLE_CANON, cutoff=0.78, allow_none=True, mahalle_mode=True)
        if known:
            filtered.append(m)
    if filtered:
        mahalle_opts = filtered

if len(mahalle_opts) == 0:
    mahalle = st.selectbox("Mahalle (opsiyonel)", options=["Seçilmedi"], index=0)
else:
    mahalle = st.selectbox("Mahalle (opsiyonel)", options=["Seçilmedi"] + mahalle_opts, index=0)

# Seçenek listeleri (genel)
tipi_list = safe_unique(df_all, "Tipi", ["Daire"])
bulundugu_kat_list = safe_unique(df_all, "Bulunduğu Kat", safe_unique(df_all, "Bulundugu Kat", ["Zemin Kat", "1.Kat", "2.Kat", "3.Kat"]))
isitma_list = safe_unique(df_all, "Isıtma Tipi", safe_unique(df_all, "Isitma Tipi", ["Kombi (Doğalgaz)", "Merkezi", "Klima", "Soba"]))
tapu_list = safe_unique(df_all, "Tapu Durumu", ["Kat Mülkiyeti", "Kat İrtifakı"])

col1, col2 = st.columns(2)

with col1:
    turu = "Konut"
    kategorisi = "Satılık"
    tipi = st.selectbox("Tipi", options=tipi_list, index=0)

    net_m2 = st.number_input("Net Metrekare (m²)", min_value=20, max_value=2000, value=100, step=5)

    brut_default = float(net_m2) * 1.15
    brut_m2 = st.number_input(
        "Brüt Metrekare (m²)",
        min_value=20.0, max_value=2500.0,
        value=float(brut_default), step=5.0
    )

    bina_yas = st.number_input("Bina Yaşı (yıl)", min_value=0, max_value=120, value=5, step=1)
    bina_kat_sayisi = st.number_input("Binanın Kat Sayısı", min_value=1, max_value=80, value=5, step=1)

    tapu_durumu = st.selectbox("Tapu Durumu", options=tapu_list, index=0)

with col2:
    oda_sayi_str = st.selectbox("Oda Sayısı", options=["1+0", "1+1", "2+1", "3+1", "4+1", "5+1", "6+1", "7+1"], index=2)
    banyo_sayi = st.number_input("Banyo Sayısı", min_value=0, max_value=10, value=1, step=1)

    bulundugu_kat = st.selectbox("Bulunduğu Kat", options=bulundugu_kat_list, index=0)
    isitma_tipi = st.selectbox("Isıtma Tipi", options=isitma_list, index=0)

    kredi_uygun = st.selectbox("Krediye Uygunluk", options=["Uygun", "Uygun Değil", "Bilinmiyor"], index=0)
    site_icinde = st.selectbox("Site İçerisinde mi?", options=["Evet", "Hayır"], index=0)

st.markdown("---")

# ================== EK ÖZELLİKLER (LÜKS FAKTÖRLER) ==================
with st.expander("✨ Ek Özellikler (Lüks Faktörler)", expanded=False):
    oz_col1, oz_col2, oz_col3 = st.columns(3)
    with oz_col1:
        oz_asansor = st.checkbox("🛗 Asansör", value=False)
        oz_otopark = st.checkbox("🅿️ Otopark", value=False)
        oz_guvenlik = st.checkbox("🔒 Site Güvenliği", value=False)
    with oz_col2:
        oz_havuz = st.checkbox("🏊 Havuz", value=False)
        oz_balkon = st.checkbox("🌅 Balkon", value=False)
        oz_teras = st.checkbox("☀️ Teras", value=False)
    with oz_col3:
        oz_manzara = st.checkbox("🌊 Deniz/Doğa Manzarası", value=False)
        oz_ebeveyn = st.checkbox("🚿 Ebeveyn Banyosu", value=False)

# Seçilen özellikleri dict olarak topla
ozellikler_dict = {
    "oz_asansor": oz_asansor,
    "oz_otopark": oz_otopark,
    "oz_guvenlik": oz_guvenlik,
    "oz_havuz": oz_havuz,
    "oz_balkon": oz_balkon,
    "oz_teras": oz_teras,
    "oz_manzara": oz_manzara,
    "oz_deniz": oz_manzara,       # deniz manzarası = oz_deniz flagı da açılır
    "kw_deniz_manzarasi": oz_manzara,  # keyword flagı
    "oz_ebeveyn": oz_ebeveyn,
}

st.markdown("---")
show_debug = st.checkbox("Debug göster", value=False)

if st.button("💰 Tahmini Fiyatı Hesapla"):
    X_new, dbg = build_input_row(
        sehir=sehir,
        ilce=ilce,
        mahalle=mahalle,
        net_m2=net_m2,
        brut_m2=brut_m2,
        oda_sayi_str=oda_sayi_str,
        banyo_sayi=banyo_sayi,
        bina_yas=bina_yas,
        bina_kat_sayisi=bina_kat_sayisi,
        tapu_durumu=tapu_durumu,
        bulundugu_kat=bulundugu_kat,
        isitma_tipi=isitma_tipi,
        kredi_uygun=kredi_uygun,
        site_icinde=site_icinde,
        turu=turu,
        kategorisi=kategorisi,
        tipi=tipi,
        ozellikler=ozellikler_dict,
    )

    try:
        y_pred = model.predict(X_new)[0]
        tahmini_fiyat = float(y_pred)

        st.subheader("📌 Tahmini Satış Fiyatı")
        st.success(f"Yaklaşık **{tahmini_fiyat:,.0f} TL**")

        render_error_box_and_bar(tahmini_fiyat, ERROR_RATE)

        st.caption("Not: Bu değer, ilan verileri ve eğitilen makine öğrenmesi modeline göre yaklaşık bir tahmindir.")
        st.warning("Uyarı: Kullanılan ilan verileri 11-12 Şubat tarihinde çekilmiştir.")

        if show_debug:
            with st.expander("🔎 Mahalle/İlçe/Şehir Debug"):
                st.write("Modelde şehir kolonu:", dbg["col_il"])
                st.write("Modelde ilçe kolonu:", dbg["col_ilce"])
                st.write("Modelde mahalle kolonu:", dbg["col_mahalle"])

                st.write("Şehir (UI):", dbg["il_ui"])
                st.write("Şehir (modele giden):", dbg["il_sent"])
                st.write("Model şehir tanıyor mu?:", dbg["il_known"])
                st.write("Model şehir kategori sayısı:", dbg["il_cat_count"])

                st.write("İlçe (UI):", dbg["ilce_ui"])
                st.write("İlçe (modele giden):", dbg["ilce_sent"])
                st.write("Model ilçe tanıyor mu?:", dbg["ilce_known"])
                st.write("Model ilçe kategori sayısı:", dbg["ilce_cat_count"])

                st.write("Mahalle (UI):", dbg["mah_ui"])
                st.write("Mahalle (modele giden):", dbg["mah_sent"])
                st.write("Model mahalle tanıyor mu?:", dbg["mah_known"])
                st.write("Model mahalle kategori sayısı:", dbg["mah_cat_count"])

                # İlçe sayısı hızlı kontrol
                st.write("Bu şehir için UI ilçe sayısı:", len(ilce_list))

    except Exception as e:
        st.error(f"Tahmin yapılırken hata oluştu: {e}")
        st.write("Modelin beklediği feature formatı ile girdiler uyuşmuyor olabilir.")





# Arayüzü çalıştırmak için terminale şu komutu girin 
# python -m streamlit run app.py
