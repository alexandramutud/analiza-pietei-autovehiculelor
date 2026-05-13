import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def treat_missing_values(df, numeric_col=None, method='median', group_col=None, date_col=None):
    """
    Funcție optimizată pentru tratarea valorilor lipsă și curățare.
    """
    df_copy = df.copy()
    
    # 0. Eliminare coloană index reziduală (Unnamed: 0)
    unnamed_cols = [col for col in df_copy.columns if 'Unnamed' in col]
    if unnamed_cols:
        df_copy = df_copy.drop(columns=unnamed_cols)
    
    # 1. Imputare (numerică sau categorică)
    if numeric_col and numeric_col in df_copy.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df_copy[numeric_col])
        
        if group_col and group_col in df_copy.columns:
            if not is_numeric:
                # AVANSAT TEXT: Fallback pentru coloane ne-numerice pe grup (folosim Modulul grupului)
                df_copy[numeric_col] = df_copy.groupby(group_col)[numeric_col].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else df_copy[numeric_col].mode()[0])
                )
            else:
                # AVANSAT NUMERIC: Imputare pe grupuri cu fallback la mediana globală
                global_val = df_copy[numeric_col].median() if method == 'median' else df_copy[numeric_col].mean()
                df_copy[numeric_col] = df_copy.groupby(group_col)[numeric_col].transform(
                    lambda x: x.fillna(x.median() if method == 'median' else x.mean())
                ).fillna(global_val)
        else:
            # Imputare simplă (globală)
            if is_numeric and method == 'median':
                df_copy[numeric_col] = df_copy[numeric_col].fillna(df_copy[numeric_col].median())
            elif is_numeric and method == 'mean':
                df_copy[numeric_col] = df_copy[numeric_col].fillna(df_copy[numeric_col].mean())
            else:
                # Modul pentru orice tip de date (fallback sau cerere explicită)
                mode_val = df_copy[numeric_col].mode()
                if not mode_val.empty:
                    df_copy[numeric_col] = df_copy[numeric_col].fillna(mode_val[0])
    
    # 2. Conversie dată (dacă este specificată)
    if date_col and date_col in df_copy.columns:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        # 3. Creare coloană binară
        df_copy['has_review'] = df_copy[date_col].apply(lambda x: 0 if pd.isnull(x) else 1)
        
    return df_copy

def encode_categorical_columns(df, columns, method='label', normalize_frequency=True):
    """
    Aplică encodare pe coloanele categoriale selectate.
    Returnează DataFrame-ul encodat și mapările folosite (acolo unde este cazul).
    """
    df_copy = df.copy()
    mappings = {}

    if method == 'label':
        for col in columns:
            categories = sorted(df_copy[col].dropna().unique().tolist(), key=lambda x: str(x))
            mapping = {category: idx for idx, category in enumerate(categories)}
            df_copy[col] = df_copy[col].map(mapping)
            mappings[col] = mapping

    elif method == 'frequency':
        for col in columns:
            freq_map = df_copy[col].value_counts(normalize=normalize_frequency, dropna=True).to_dict()
            df_copy[col] = df_copy[col].map(freq_map)
            mappings[col] = freq_map

    return df_copy, mappings

def scale_data(df, columns, method='standard'):
    """
    Scalează coloanele numerice selectate folosind Standardizare (Z-score) sau Normalizare (Min-Max).
    """
    df_copy = df.copy()
    for col in columns:
        if method == 'standard':
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            if std_val != 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
        elif method == 'minmax':
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            if max_val != min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
    return df_copy

st.title("Proiect PSW - Pachete Software")
st.markdown(
    """
    <style>
    .custom-title {
        color: #F39C12;
        font-size: 40px;
        text-align: center;
        color: red !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="custom-title">Analiza Specificațiilor Auto 2023</h1>', unsafe_allow_html=True)

# Încărcare seturi de date folosind data caching
@st.cache_data
def func_incarcare_date_2023():
    base = Path(__file__).parent
    data_file = base / "Set de date" / "Set masini 2023.csv"
    return pd.read_csv(data_file, low_memory=False)

# Bara laterală pentru navigare între secțiuni
section = st.sidebar.radio("Navigați la:",
                           ["Introducere", "Setul de date", "Informații și Previzualizare"])

# Încărcare set de date folosind data caching pentru aplicații performante
@st.cache_data
def func_incarcare_date():
    return pd.read_csv(r"c:\Users\YAN\Desktop\PSW\Proiect PSW\Set de date\Specificatii masini 2023.csv")

# Preluăm setul de date
df = func_incarcare_date()

# ---------------------------
# Secțiunea: Introducere
# ---------------------------
if section == "Introducere":
    st.header("Despre Proiect")
    st.markdown("""
        Acest proiect este o aplicație web interactivă dezvoltată în **Python cu Streamlit**, 
        ce urmărește analiza pieței autovehiculelor, în anul 2023, pentru viitoare predicții,
        informări sau grafice. Este destinată atât persoaneor fizice, care urmează
        să achiziționeze un nou autovehicul, cât și firmelor de tip parc auto sau închirieri.
        
        
        ### Obiectivele Principale
        ...
        
        Navigați folosind meniul din stânga pentru a explora mai departe!
        """)

# ---------------------------
# Secțiunea: Setul de date
# ---------------------------
elif section == "Setul de date":
    st.header("Contextul Datelor")
    st.write("""
    Acest proiect utilizează un set de date complex denumit **Specificatii masini 2023.csv**.
    El conține detalii tehnice pentru diferite autovehicule disponibile pe piață, precum:
    - `Company`, `Model`, `Segment`, `Fuel`
    - Specificații engine (`Power(HP)`, `Torque(Nm)`, `Displacement`)
    - Dimensiuni (`Length`, `Width`, `Height`, `Wheelbase`)
    - Informații de performanță și eficiență.
    """)
    
    

# ---------------------------
# Secțiunea: Informații și Previzualizare
# ---------------------------
elif section == "Informații și Previzualizare":
    st.header("Explorarea Datelor (EDA)")
    
    st.write("Dimensiunile setului de date:")
    st.write(df.shape)

    st.subheader("Filtrare Multiplă a Înregistrărilor")
    
    st.write("Folosește filtrele de mai jos pentru a rafina datele afișate:")
    
    # Folosim coloane pentru a aranja filtrele de tip text/dropdown frumos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lista_marci = ["Toate"] + sorted(df['Company'].dropna().unique())
        marca = st.selectbox("Marcă auto (Company):", lista_marci)
        
    with col2:
        lista_segmente = ["Toate"] + sorted(df['Segment'].dropna().unique())
        segment = st.selectbox("Segment:", lista_segmente)
        
    with col3:
        lista_combustibil = ["Toate"] + sorted(df['Fuel'].dropna().unique())
        combustibil = st.selectbox("Combustibil (Fuel):", lista_combustibil)
        
    # Slider pentru Cai Putere (încercăm să luăm minimul și maximul corect)
    min_hp = float(df['Power(HP)'].min()) if not df['Power(HP)'].isna().all() else 0.0
    max_hp = float(df['Power(HP)'].max()) if not df['Power(HP)'].isna().all() else 2000.0
    
    # Oferim un slider cu două capete (interval) pentru cai putere
    interval_hp = st.slider(
        "Interval Cai Putere (HP):",
        min_value=min_hp,
        max_value=max_hp,
        value=(min_hp, max_hp)
    )
    
    # Aplicăm filtrele pe rând
    df_filtrat = df.copy()
    
    if marca != "Toate":
        df_filtrat = df_filtrat[df_filtrat['Company'] == marca]
        
    if segment != "Toate":
        df_filtrat = df_filtrat[df_filtrat['Segment'] == segment]
        
    if combustibil != "Toate":
        df_filtrat = df_filtrat[df_filtrat['Fuel'] == combustibil]
        
    # Filtrare după Putere
    df_filtrat = df_filtrat[(df_filtrat['Power(HP)'] >= interval_hp[0]) & (df_filtrat['Power(HP)'] <= interval_hp[1])]
    
    st.write(f"**Următoarele {len(df_filtrat)} autovehicule corespund criteriilor selectate:**")
    st.dataframe(df_filtrat)

    
    st.subheader("Baza analizei statistice descriptive:")
    st.write("Statisticile pentru coloanele numerice (`describe()`):")
    st.dataframe(df.describe())
    
    st.markdown("---")
    st.write(f"Număr total curent de coloane analizate: {len(df.columns)}")
