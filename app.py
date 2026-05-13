import streamlit as st
import pandas as pd
import numpy as np

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
