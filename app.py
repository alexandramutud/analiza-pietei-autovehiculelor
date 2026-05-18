import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import statsmodels.api as sm
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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


def describe_correlation(r):
    """
    Returnează o descriere textuală a intensității corelației (Pearson).
    """
    r_abs = abs(r)
    if r_abs >= 0.7:
        return "puternică"
    elif r_abs >= 0.4:
        return "moderată"
    elif r_abs >= 0.2:
        return "slabă"
    else:
        return "neglijabilă"


st.title("Pachete software - Analiza specificațiilor auto 2023")
st.markdown(
    """
    <style>

    h1 {
        background: linear-gradient(to top, #5B2C6F, #2E86C1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 30px !important;
        text-align: center;
        padding-bottom: 15px;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 25px !important;
    }
    h2 {
        background: linear-gradient(to top right, #1A5276, #3498DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        border-bottom: 1px solid #EAECEE;
        width: 100%;
        padding-bottom: 5px;
        margin-top: 40px !important;
        margin-bottom: 20px !important;
    }
    h3 {
        background: linear-gradient(to top right, #2E86C1, #85C1E9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600 !important;
        margin-top: 30px !important;
        margin-bottom: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Titlul a fost mutat sus in st.title

# Încărcare seturi de date folosind data caching
@st.cache_data
def func_incarcare_date_2023():
    # Subsetul necurățat extras anterior din setul de 23k observații
    return pd.read_csv(r"c:\Users\YAN\Desktop\PSW\Proiect PSW\Set de date\Set masini 2023.csv", low_memory=False)


# Bara laterală pentru navigare între secțiuni
section = st.sidebar.radio("Navigare secțiuni:", [
    "Introducere",
    "Setul de date",
    "Informații și Previzualizare",
    "Tratare de Valori Lipsă și Aberante",
    "Encodare Variabile Categoriale",
    "Normalizare și Standardizare",
    "Grupare și Agregare (Pivot)",
    "Vizualizare și Analiză Grafică",
    "Analiză Statistică (Regresie)",
    "Clusterizare (K-Means)",
    "Clusterizare (Ierarhică - HC)",
    "Clasificare Predictivă (ML)"
])

# Buton de Reset în Sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔄 RESETEAZĂ TOATE DATELE", use_container_width=True):
    # Forțăm reîncărcarea setului de date original în session_state
    raw_df = func_incarcare_date_2023().copy()
    st.session_state.df = raw_df.drop(columns=[col for col in raw_df.columns if 'Unnamed' in col], errors='ignore')
    st.session_state.cols_removed_40 = False
    st.sidebar.success("Datele au fost resetate la starea inițială.")
    st.rerun()

# Inițializăm setul de date în session_state pentru persistenta modificărilor
if 'df' not in st.session_state:
    raw_df = func_incarcare_date_2023()
    # Eliminăm coloanele reziduale (Unnamed) la prima încărcare
    st.session_state.df = raw_df.drop(columns=[col for col in raw_df.columns if 'Unnamed' in col], errors='ignore')

if 'cols_removed_40' not in st.session_state:
    # Tracker pentru blocarea curățării radicale până la confirmarea pasului fringe.
    st.session_state.cols_removed_40 = False

# Folosim df din session_state pentru restul aplicației
df = st.session_state.df

# ---------------------------
# Secțiunea: Introducere
# ---------------------------
if section == "Introducere":
    st.header("Despre Proiect")
    st.markdown("""
        ### Despre Proiect și Setul de Date
        Acest proiect este o aplicație web interactivă de top, dezvoltată în **Python folosind Streamlit**, 
        dedicată analizei amănunțite a specificațiilor tehnice ale pieței autovehiculelor din anul 2023.
        Setul de date cuprinde informații detaliate (dimensiuni, performanță, capacități, motorizare etc.) 
        pentru mii de modele lansate până în anul 2023, la nivel global. 
        S-au aplicat operații precum: curățare de date lipsă și aberante, encodări, scalări,
        analize statistice, pivotări și modele avansate de Machine Learning.

        ### Importanța Analizei Pieței Auto
        Această analiză transformă cifrele aride într-o poveste cu sens, 
        oferind oricui o viziune clară asupra drumului pe care merită să pornească. 
        Pentru cei care coordonează flote, ea devine un instrument de precizie care 
        arată exact ce modele atrag publicul și aduc profit, eliminând orice urmă de ghicire. 
        În același timp, cei care proiectează mașinile viitorului găsesc aici un reper esențial 
        pentru a înțelege ce își dorește piața cu adevărat, de la forța motorului până 
        la dimensiunile ideale pentru oraș. Chiar și pentru un simplu șofer, aceste date 
        funcționează ca un ghid prietenos care traduce opțiunile complicate în alegeri sigure, 
        ajutându-l să găsească echilibrul perfect între consum și confort pentru viața de zi 
        cu zi.
        ### Obiectivele Principale (Structura Aplicației)
        Aplicația este divizată logic pentru a acoperi întreg ciclul de viață al analizei datelor. Puteți naviga folosind meniul din stânga prin următoarele secțiuni:
        - **1. Introducere:** Prezentarea proiectului și a utilității sale practice.
        - **2. Setul de date:** Contextul și dimensiunile setului analizat.
        - **3. Informații și Previzualizare:** Explorarea brută și filtrarea dinamică a catalogului auto.
        - **4. Tratare de Valori Lipsă și Aberante:** Eliminarea rândurilor corupte și a extremelor (Outliers) pentru stabilitate matematică.
        - **5. Encodare Variabile Categoriale:** Transformarea textului (ex: Marca, Combustibil) în valori numerice.
        - **6. Normalizare și Standardizare:** Aducerea variabilelor la o scară comună.
        - **7. Grupare și Agregare (Pivot):** Extragerea statisticilor centralizate pe categorii de vehicule.
        - **8. Vizualizare și Analiză Grafică:** Explorare vizuală (Histograme, Boxplot, Scatter, Heatmap etc.).
        - **9. Analiză Statistică (Regresie):** Studiul regresiei multiple (OLS) pentru dependențe statistice.
        - **10. Clusterizare (K-Means):** Gruparea nesupervizată a mașinilor similare în funcție de performanțe.
        - **11. Clusterizare (Ierarhică - HC):** Identificarea dendrogramelor și relațiilor ierarhice din industrie.
        - **12. Clasificare Predictivă (ML):** Antrenarea algoritmilor supervizați (Arbori, Random Forest, Regresie Logistică) pentru a prezice apartenența unei mașini noi la un anumit segment auto pe baza dimensiunilor și puterii.
        """)

# ---------------------------
# Secțiunea: Setul de date
# ---------------------------
elif section == "Setul de date":
    st.header("Contextul Datelor")
    st.markdown(f"""
    Acest proiect utilizează un set de date vast și detaliat, concentrat pe specificațiile autovehiculelor disponibile pe piață până în anul **2023**. Setul de date reprezintă o colecție tehnică extrem de valoroasă ce surprinde caracteristicile de bază, performanțele și dimensiunile modelelor auto la nivel global.

    În prezent, setul de lucru activ pe care se desfășoară analiza conține **{df.shape[0]}** rânduri (reprezentând modele auto individuale) și **{df.shape[1]}** coloane (atribute tehnice și descriptive).

    ### Datele analizate se împart în 4 categorii majore:

    **1. Identificare și Clasificare:**
    - **Marca și Modelul (`Company`, `Model`)**: Esențiale pentru compararea producătorilor auto și a cotelor de piață.
    - **Segmentul Auto (`Segment`)**: Clasificarea mașinii (ex: SUV, Hatchback, Sedan, Coupe, Exotic). Această variabilă este crucială pentru a înțelege preferințele publicului.
    - **Tipul de Combustibil (`Fuel`)**: Benzină, Motorină, Hibrid etc. – o informație cheie în contextul tranziției ecologice actuale.

    **2. Inima Mașinii (Specificații Motor):**
    - **Putere și Cuplu (`Power(HP)`, `Torque(Nm)`)**: Indicatorii principali ai performanței brute și ai capacității de tracțiune.
    - **Cilindree (`Displacement`)**: Volumul motorului în centimetri cubi, un factor decisiv în politicile de taxare și eficiența termică.

    **3. Arhitectură și Dimensiuni Fizice:**
    - **Gabaritul (`Length`, `Width`, `Height`)**: Lățimea, lungimea și înălțimea dictează nu doar aspectul, ci și spațiul interior și manevrabilitatea urbană.
    - **Ampatamentul (`Wheelbase`)**: Distanța dintre punți; un ampatament mare oferă un confort sporit pasagerilor.
    - **Greutate și Utilitate (`Unladen Weight`, `Cargo Volume`)**: Masa mașinii și volumul portbagajului, detalii practice esențiale pentru familii sau flote comerciale.

    **4. Performanță și Eficiență Economică:**
    - **Dinamica (`Top Speed`, `Acceleration 0-100 kph`)**: Criterii de performanță pură, vitale pentru clienții din segmentele Sport/Premium.
    - **Eficiența (`Combined mpg`, `Fuel capacity`)**: Consumul combinat și mărimea rezervorului ajută la calculul direct al costului de exploatare (cât de „scumpă” e mașina zi de zi).
    """)
    st.info(
        "💡 **Sfat:** Poți continua în secțiunea următoare pentru a explora vizual acest tabel prin filtre dinamice!")
# Secțiunea: Informații și Previzualizare
# ---------------------------
elif section == "Informații și Previzualizare":
    st.header("Informații și Previzualizare Date")
    st.write(f"Setul curent de lucru conține **{df.shape[0]}** rânduri și **{df.shape[1]}** coloane.")

    st.subheader("Filtrare și Explorare")
    st.write("Folosește filtrele de mai jos pentru a clasifica datele afișate în tabel:")

    col1, col2, col3 = st.columns(3)
    with col1:
        lista_marci = ["Toate"] + sorted(df['Company'].dropna().unique())
        marca = st.selectbox("Marcă auto:", lista_marci)
    with col2:
        lista_segmente = ["Toate"] + sorted(df['Segment'].dropna().unique())
        segment = st.selectbox("Segment:", lista_segmente)
    with col3:
        lista_combustibil = ["Toate"] + sorted(df['Fuel'].dropna().unique())
        combustibil = st.selectbox("Combustibil:", lista_combustibil)

    min_hp = float(df['Power(HP)'].min()) if not df['Power(HP)'].isna().all() else 0.0
    max_hp = float(df['Power(HP)'].max()) if not df['Power(HP)'].isna().all() else 2000.0
    interval_hp = st.slider("Interval Cai Putere (HP):", min_value=min_hp, max_value=max_hp, value=(min_hp, max_hp))

    # Aplicare filtre
    df_preview = df.copy()
    if marca != "Toate": df_preview = df_preview[df_preview['Company'] == marca]
    if segment != "Toate": df_preview = df_preview[df_preview['Segment'] == segment]
    if combustibil != "Toate": df_preview = df_preview[df_preview['Fuel'] == combustibil]
    df_preview = df_preview[(df_preview['Power(HP)'] >= interval_hp[0]) & (df_preview['Power(HP)'] <= interval_hp[1])]

    st.write(f"**Rezultate: {len(df_preview)} înregistrări găsite.**")
    st.dataframe(df_preview)

    st.subheader("Statistici Descriptive")
    st.write("Sumar statistic pentru coloanele numerice (pentru tot setul din sesiune):")
    st.dataframe(df.describe())

# ---------------------------
# Secțiunea: Tratarea Valorilor Lipsă
# ---------------------------
elif section == "Tratare de Valori Lipsă și Aberante":
    st.header("Tratare de Valori Lipsă și Aberante")
    st.write("""Asigură-te că datele sunt curate înainte de a trece la vizualizări complexe. Le poți curăța automat prin
                ștergerea coloanelor cu procent ridicat de valori lipsă, și apoi prin ștergerea rândurilor cu valori lipsă sau 
                prin ștergerea manuală. """)

    has_nans = df.isnull().values.any()

    if has_nans:
        missing_vals = df.isnull().sum()
        missing_percent = (missing_vals / len(df)) * 100
        missing_only = pd.DataFrame({
            'Valori Lipsă': missing_vals,
            'Procent (%)': missing_percent
        })[missing_vals > 0].sort_values('Procent (%)', ascending=False)

        st.write("Coloane cu date lipsă:")
        st.dataframe(missing_only)

        st.markdown("---")

        st.subheader("Eliminare automată")
        st.markdown("#### Eliminare coloane fringe (Prag > 40%)")
        cols_to_drop = missing_percent[missing_percent > 40].index.tolist()
        if cols_to_drop:
            st.warning(f"Atenție: {len(cols_to_drop)} coloane sunt aproape goale (>40% null).")
            if st.button("🗑️ Elimină aceste coloane acum", use_container_width=True):
                st.session_state.df = st.session_state.df.drop(columns=cols_to_drop)
                st.session_state.cols_removed_40 = True
                st.success("Coloanele periferice au fost eliminate!")
                st.rerun()
        else:
            st.success("✅ Toate coloanele rămase sunt bine populate.")
            st.session_state.cols_removed_40 = True  # Putem debloca deoarece nu riscăm pierderea întregului set.

        st.markdown("#### Curățare radicală")
        if 'dropna_msg' in st.session_state:
            st.success(st.session_state['dropna_msg'])
            del st.session_state['dropna_msg']

        if not st.session_state.cols_removed_40:
            st.error(
                "🔒 **Acțiune blocată**: Mai întâi trebuie să elimini coloanele aproape goale (Prag > 40%) pentru a nu goli tot setul de date!")
        else:
            st.warning("⚠️ Această acțiune va șterge orice rând care mai are vreo celulă goală. Sigur dorești?")
            if st.button("🚀 ȘTERGE TOATE RÂNDURILE CU VALORI LIPSĂ", use_container_width=True):
                old_len = len(st.session_state.df)
                st.session_state.df = st.session_state.df.dropna()
                st.session_state['dropna_msg'] = f"Finalizat: {old_len - len(st.session_state.df)} rânduri eliminate."
                st.rerun()

        st.markdown("---")

        st.subheader("Sau ... curățare manuală")

        # Imputare
        st.markdown("#### Imputare valori (completare selectivă)")
        num_cols_with_nan = df.columns[df.isnull().any()].tolist()
        if num_cols_with_nan:
            target_col = st.selectbox("Alege coloana pentru tratare:", num_cols_with_nan)
            metoda = st.radio("Metoda:", ["Mediană", "Medie", "Modulul", "Avansată (pe Marcă)"], horizontal=True)

            if st.button("✨ Aplică Imputarea", use_container_width=True):
                m_key = 'median' if metoda == "Mediană" else 'mean' if metoda == "Medie" else 'mode'
                if metoda == "Avansată (pe Marcă)":
                    st.session_state.df = treat_missing_values(st.session_state.df, numeric_col=target_col,
                                                               group_col='Company')
                else:
                    st.session_state.df = treat_missing_values(st.session_state.df, numeric_col=target_col,
                                                               method=m_key)
                st.rerun()

        # Eliminare rânduri specifice
        st.markdown("#### Eliminare rânduri specifice")
        if 'dropcol_msg' in st.session_state:
            st.success(st.session_state['dropcol_msg'])
            del st.session_state['dropcol_msg']

        row_col = st.selectbox("Șterge rândurile unde lipsește coloana:", ["Selectează"] + num_cols_with_nan,
                               key="row_drop")
        if row_col != "Selectează":
            if st.button(f"🗑️ Șterge rândurile fără {row_col}", use_container_width=True):
                old_len = len(st.session_state.df)
                st.session_state.df = st.session_state.df.dropna(subset=[row_col])
                st.session_state['dropcol_msg'] = f"S-au eliminat {old_len - len(st.session_state.df)} rânduri."
                st.rerun()

    else:
        st.success("✅ Felicitări! Nu mai există nicio valoare lipsă în setul de date.")

    st.markdown("---")
    st.subheader("Eliminare valori aberante (Outliers)")
    if 'outlier_msg' in st.session_state:
        st.success(st.session_state['outlier_msg'])
        del st.session_state['outlier_msg']

    st.caption(
        "Regulă folosită e pragul extins Tukey, cu limite [Q1 - 3×IQR, Q3 + 3×IQR]. Rândurile care conțin cel puțin o valoare aberantă numerică sunt eliminate din setul curent.")

    numeric_cols_for_outliers = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols_for_outliers:
        st.info("Nu există coloane numerice pentru detectarea valorilor aberante.")
    else:
        if st.button("🚀 Elimină TOATE valorile aberante (3×IQR)", use_container_width=True):
            df_work = st.session_state.df.copy()
            numeric_df = df_work[numeric_cols_for_outliers]

            q1 = numeric_df.quantile(0.25)
            q3 = numeric_df.quantile(0.75)
            iqr = q3 - q1

            lower_bounds = q1 - 3 * iqr
            upper_bounds = q3 + 3 * iqr

            outlier_mask = ((numeric_df < lower_bounds) | (numeric_df > upper_bounds)).any(axis=1)
            removed_rows = int(outlier_mask.sum())

            st.session_state.df = df_work.loc[~outlier_mask].copy()
            st.session_state[
                'outlier_msg'] = f"S-au eliminat {removed_rows} rânduri care conțineau cel puțin o valoare aberantă (3×IQR)."
            st.rerun()

# ---------------------------
# Secțiunea: Encodare Variabile Categoriale
# ---------------------------
elif section == "Encodare Variabile Categoriale":
    st.header("Encodare Variabile Categoriale")
    st.write("Această secțiune transformă variabilele text în format numeric pentru modelare și predicție.")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        st.success("✅ Nu mai există coloane categoriale de encodat în setul curent.")
    else:
        with st.expander("Ghid rapid: ce metodă alegi?", expanded=True):
            st.markdown(
                """
- **Label Encoding**: util pentru arbori de decizie sau când ai multe categorii.
- **Frequency Encoding**: util când cardinalitatea este mare și vrei puține coloane.
"""
            )

        selected_cat_cols = st.multiselect(
            "Alege coloanele categoriale pentru encodare:",
            categorical_cols,
            default=categorical_cols[:1]
        )

        method_label = st.radio(
            "Metodă de encodare:",
            ["Label Encoding", "Frequency Encoding"],
            horizontal=True
        )

        selected_method = {
            "Label Encoding": "label",
            "Frequency Encoding": "frequency"
        }[method_label]

        normalize_frequency = True

        if selected_method == "frequency":
            st.caption("Frequency Encoding folosește implicit frecvență relativă (interval 0-1).")

        if selected_cat_cols:
            encoded_df, mappings = encode_categorical_columns(
                df,
                selected_cat_cols,
                method=selected_method,
                normalize_frequency=normalize_frequency
            )

            if mappings and selected_method in ["label", "frequency"]:
                st.markdown("#### Exemplu mapare")
                map_col = st.selectbox("Alege coloana pentru mapare:", selected_cat_cols, key="mapping_col")
                map_items = list(mappings.get(map_col, {}).items())
                if map_items:
                    st.dataframe(pd.DataFrame(map_items, columns=["Categorie", "Valoare Encodată"]).head(30))

            st.markdown("---")
            st.subheader("Previzualizare rezultat")
            st.write(f"Dimensiune inițială: **{df.shape[0]} x {df.shape[1]}**")
            st.write(f"Dimensiune după encodare: **{encoded_df.shape[0]} x {encoded_df.shape[1]}**")
            st.dataframe(encoded_df.head(10))

            if st.button("✅ Aplică encodarea în setul curent", use_container_width=True):
                st.session_state.df = encoded_df
                st.success("Encodarea a fost aplicată în sesiunea curentă.")
                st.rerun()
        else:
            st.info("Selectează cel puțin o coloană categorială pentru a continua.")

# ---------------------------
# Secțiunea: Normalizare și Standardizare
# ---------------------------
elif section == "Normalizare și Standardizare":
    st.header("Normalizare și Standardizare")
    st.write(
        "Aducerea variabilelor la o scară comună este esențială pentru majoritatea algoritmilor de învățare automată (ML).")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ Nu s-au detectat coloane numerice pe care să le putem scala.")
    else:
        with st.expander(
                "🎓 Scalarea cuprinde standardizarea și normalizarea. Deși în limbajul colocvial, termenii se substituiesc unul pe altul, în realitate sunt diferiți.",
                expanded=False):
            st.markdown("""
- **Standardizarea (Z-score)**: Transformă datele astfel încât **Media = 0** și **Deviația Standard = 1**. Formula: `z = (x - mean) / std`. 
- **Normalizarea (Min-Max)**: Transformă datele în intervalul fix **[0, 1]**. Formula: `x_norm = (x - min) / (max - min)`.
            """)

        st.subheader("Configurare")

        col_select_all = st.checkbox("Selectează toate coloanele numerice", value=False)
        default_cols = numeric_cols if col_select_all else []

        selected_scale_cols = st.multiselect(
            "Alege coloanele pentru scalare:",
            numeric_cols,
            default=default_cols
        )

        method_scale = st.radio(
            "Alege metoda de scalare:",
            ["Standardizare (Z-score)", "Normalizare (Min-Max)"],
            horizontal=True
        )

        scale_method_key = "standard" if "Standardizare" in method_scale else "minmax"

        if selected_scale_cols:
            scaled_df = scale_data(df, selected_scale_cols, method=scale_method_key)

            st.markdown("---")
            st.subheader("🔬 Verificare Statistică")
            st.write("Comparație sumară (Înainte vs După) pentru a valida transformarea:")

            # Construim un tabel de verificare pentru prima coloană selectată ca exemplu
            check_col = selected_scale_cols[0]
            stats_compare = pd.DataFrame({
                "Statistică": ["Minim", "Maxim", "Medie", "Std Dev"],
                "Original": [df[check_col].min(), df[check_col].max(), df[check_col].mean(), df[check_col].std()],
                "Scalat": [scaled_df[check_col].min(), scaled_df[check_col].max(), scaled_df[check_col].mean(),
                           scaled_df[check_col].std()]
            })
            st.write(f"Exemplu pentru coloana: **{check_col}**")
            st.table(stats_compare)

            if st.button("🚀 Aplică scalarea pe setul de date", use_container_width=True):
                st.session_state.df = scaled_df
                st.success(f"Scalarea ({scale_method_key}) a fost aplicată cu succes!")
                st.rerun()
        else:
            st.info("Selectează coloanele dorite pentru a vedea previzualizarea.")

# ---------------------------
# Secțiunea: Grupare și Agregare (Pivot)
# ---------------------------
elif section == "Grupare și Agregare (Pivot)":
    st.header("Grupare și Agregare Date")
    st.write("Folosește această secțiune pentru a obține statistici centralizate pe grupe (Pivot Table).")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not categorical_cols:
        st.warning("⚠️ Nu sunt coloane categoriale pentru grupare.")
    elif not numeric_cols:
        st.warning("⚠️ Nu sunt coloane numerice pentru agregare.")
    else:
        st.subheader("Configurare Pivot")

        col1, col2 = st.columns(2)
        with col1:
            group_by_col = st.selectbox("Grupare după:", categorical_cols)
        with col2:
            agg_num_cols = st.multiselect("Coloane numerice pentru analiză:", numeric_cols, default=numeric_cols[:1])

        st.markdown("---")
        st.write("Alege funcțiile de agregare statistice:")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            f_mean = st.checkbox("Media", value=True)
        with c2:
            f_median = st.checkbox("Mediana")
        with c3:
            f_max = st.checkbox("Maximul")
        with c4:
            f_min = st.checkbox("Minimul")
        with c5:
            f_count = st.checkbox("Număr înregistrări", value=True)

        agg_functions = []
        if f_mean: agg_functions.append('mean')
        if f_median: agg_functions.append('median')
        if f_max: agg_functions.append('max')
        if f_min: agg_functions.append('min')
        if f_count: agg_functions.append('count')

        if agg_num_cols and agg_functions:
            # Calculăm tabelul pivot
            pivot_table = df.groupby(group_by_col)[agg_num_cols].agg(agg_functions)

            # Curățăm un pic formatarea numelor coloanelor pentru lizibilitate
            pivot_table.columns = ['_'.join(col).strip() for col in pivot_table.columns.values]

            st.markdown("### Tabel Centralizator")
            st.dataframe(pivot_table)

            # Vizualizare rapidă (îmbunătățită și dinamică)
            available_metrics = [m for m in agg_functions if m in ['mean', 'median', 'max', 'min']]
            if available_metrics and agg_num_cols:
                metric_to_plot = available_metrics[0]
                metric_label = {"mean": "Mediei", "median": "Mediană", "max": "Maximului", "min": "Minimului"}[
                    metric_to_plot]

                st.markdown(f"### 📊 Reprezentare Vizuală a {metric_label}")
                first_col = f"{agg_num_cols[0]}_{metric_to_plot}"

                if first_col in pivot_table.columns:
                    plot_data = pivot_table[first_col].sort_values(ascending=False).head(15)

                    fig, ax = plt.subplots(figsize=(16, 10))
                    sns.barplot(x=plot_data.index, y=plot_data.values, palette="viridis", ax=ax)
                    plt.xticks(rotation=90, ha='center', fontweight='bold', fontsize=12)
                    plt.title(f"Top 15 - {metric_label} {agg_num_cols[0]} pe fiecare {group_by_col}", fontsize=16,
                              fontweight='bold')
                    plt.ylabel(f"Valoare {metric_label} ({agg_num_cols[0]})", fontsize=13)
                    plt.xlabel(group_by_col, fontsize=13)
                    st.pyplot(fig)
                    st.caption(f"Grafic generat automat pentru metricul: {metric_label}.")




# ---------------------------
# Secțiunea: Vizualizare și Analiză Grafică
# ---------------------------
elif section == "Vizualizare și Analiză Grafică":
    st.header("Vizualizare și Analiză Grafică")

    if df.empty:
        st.error("❌ Eroare: Setul de date este gol!")
        st.info("Folosește butonul **RESETEAZĂ TOATE DATELE** din stânga.")
    else:
        st.write(
            "Această secțiune explorează vizual proprietățile setului de date, oferind interpretări bazate pe grafice ale diverselor modele de mașini din ani de fabricație până în 2023")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()


        def describe_correlation(corr_value):
            abs_corr = abs(corr_value)
            if abs_corr >= 0.7:
                return "puternică"
            if abs_corr >= 0.4:
                return "moderată"
            if abs_corr >= 0.2:
                return "slabă"
            return "foarte slabă"


        st.markdown(
            """
            <style>
            div[data-testid="stTabs"] button[role="tab"] {
                font-size: 1.08rem;
                padding: 0.85rem 1.2rem;
                border-radius: 10px 10px 0 0;
                border: 1px solid #d5dbe6;
                background-color: #f5f8fc;
                margin-right: 0.25rem;
            }
            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
                color: #0b3b78;
                font-weight: 700;
                border-bottom: 3px solid #1f77b4;
                background: linear-gradient(180deg, #e8f1ff 0%, #dbeaff 100%);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Distribuții", "📦 Outliers", "📈 Relații", "🔥 Corelații", "🏢 Categorii", "🔗 Pair Plots", "🎻 Violin Plot"
        ])

        with tab1:
            st.subheader("Analiza Distribuției Datelor")
            if numeric_cols:
                col_dist = st.selectbox("Alege coloana:", numeric_cols)
                dist_series = df[col_dist].dropna()
                fig, ax = plt.subplots(1, 2, figsize=(16, 7))
                sns.histplot(df[col_dist], bins=30, kde=False, color="#3498db", ax=ax[0])
                sns.kdeplot(df[col_dist], fill=True, color="#e67e22", ax=ax[1])
                st.pyplot(fig)
                if not dist_series.empty:
                    mean_val = dist_series.mean()
                    median_val = dist_series.median()
                    std_val = dist_series.std()
                    skew_val = dist_series.skew()
                    if skew_val > 0.5:
                        skew_text = "asimetrică spre dreapta (valori mari rare)"
                    elif skew_val < -0.5:
                        skew_text = "asimetrică spre stânga (valori mici rare)"
                    else:
                        skew_text = "aproape simetrică"

                    # Interpretare logica in context auto pentru medie vs mediana
                    if abs(mean_val - median_val) > std_val * 0.2:
                        mean_vs_median_text = "Diferența notabilă dintre medie și mediană sugerează prezența unor modele 'extreme' care trag media într-o direcție, pe când mediana reprezintă mult mai bine autovehiculul 'de rând' sau 'tipic'."
                    else:
                        mean_vs_median_text = "Media și mediana sunt foarte apropiate, ceea ce înseamnă că piața este echilibrată, majoritatea mașinilor gravitând strâns în jurul acestui standard."

                    # Interpretare logica pentru dispersie
                    if std_val > mean_val * 0.3:
                        std_text = "Această deviație mare indică o ofertă extrem de diversificată (opțiuni de la variante foarte slabe/mici până la variante de top/gigantice)."
                    else:
                        std_text = "O deviație relativ mică, sugerând o piață omogenă unde producătorii merg pe specificații testate, sigure, fără variații radicale."

                    st.markdown(
                        f"""
**Interpretarea aplicată a pieței auto:**
- **Standardul industriei (Media vs. Mediana):** Din punct de vedere matematic, media este **{mean_val:.2f}**, dar jumătatea pieței (mediana) se află de fapt la **{median_val:.2f}**. *{mean_vs_median_text}*
- **Diversitatea ofertei (Dispersia):** Abaterea standard de **{std_val:.2f}** arată cât de largă este plaja de alegeri pe care le are un cumpărător. *{std_text}*
- **Tendința de design (Forma distribuției):** Graficul are o formă **{skew_text}** (indice = **{skew_val:.2f}**). Practic, asta ne arată vizual dacă piața se focusează pe modele standard, tratând valorile extreme (ex. mașinile hyper-sport) ca pe nișe de lux, sau dacă există o distribuție echilibrată a producției.
"""
                    )

        with tab2:
            st.subheader("Analiza Outliers (Boxplot)")
            if numeric_cols:
                col_num = st.selectbox("Variabilă numerică:", numeric_cols, key="box_num")
                col_cat = st.selectbox("Grupare după:", ["Toate"] + categorical_cols, key="box_cat")
                fig, ax = plt.subplots(figsize=(14, 8))
                if col_cat == "Toate":
                    sns.boxplot(y=df[col_num], color="#2ecc71", width=0.3, ax=ax)
                    outlier_series = df[col_num].dropna()
                else:
                    top_cats = df[col_cat].value_counts().nlargest(12).index
                    df_filtered = df[df[col_cat].isin(top_cats)]
                    sns.boxplot(x=col_cat, y=col_num, data=df_filtered, palette="viridis", ax=ax)
                    plt.xticks(rotation=45)
                    outlier_series = df_filtered[col_num].dropna()
                st.pyplot(fig)

                if not outlier_series.empty:
                    q1 = outlier_series.quantile(0.25)
                    q3 = outlier_series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = outlier_series[(outlier_series < lower_bound) | (outlier_series > upper_bound)]
                    outlier_pct = (len(outliers) / len(outlier_series)) * 100 if len(outlier_series) else 0

                    if outlier_pct > 5:
                        outlier_interp = "Avem un procent semnificativ de valori extreme. În piața auto, asta se traduce prin inovații sau modele de top (precum ediții limitate hyper-sport sau utilitare masive) care deviază mult de la vehiculele comune."
                    else:
                        outlier_interp = "Există puține excepții de la regulă, arătând că producătorii preferă să se încadreze în normele sigure, acceptate de majoritatea clienților."

                    st.markdown(
                        f"""
**Interpretarea aplicată a valorilor extreme (Outliers):**
- **Intervalul de normalitate:** Matematic, limitele normalității sunt între **{lower_bound:.2f}** și **{upper_bound:.2f}**. Orice mașină care trece de aceste praguri este considerată un 'Outlier' – o excepție rară pe piață.
- **Concentrarea competiției (IQR):** Jumătatea de mijloc a tuturor modelelor se încadrează într-o fereastră strânsă de doar **{iqr:.2f}** unități, arătându-ne plaja standard în care competiția între mărci este cea mai acerbă.
- **Verdict piață:** S-au detectat **{len(outliers)}** mașini atipice din totalul de **{len(outlier_series)}** (**{outlier_pct:.2f}%**). *{outlier_interp}*
"""
                    )

        with tab3:
            st.subheader("Relații (Scatter Plot)")
            if len(numeric_cols) >= 2:
                c1, c2, c3 = st.columns(3)
                with c1:
                    col_x = st.selectbox("X:", numeric_cols, index=0)
                with c2:
                    col_y = st.selectbox("Y:", numeric_cols, index=1)
                with c3:
                    col_color = st.selectbox("Culoare:", ["Fără"] + categorical_cols)
                fig, ax = plt.subplots(figsize=(14, 8))
                if col_color == "Fără":
                    sns.scatterplot(x=col_x, y=col_y, data=df, alpha=0.5, ax=ax)
                    rel_df = df[[col_x, col_y]].dropna()
                else:
                    top_cats = df[col_color].value_counts().nlargest(6).index
                    df_filtered = df[df[col_color].isin(top_cats)]
                    sns.scatterplot(x=col_x, y=col_y, hue=col_color, data=df_filtered, alpha=0.7, ax=ax)
                    rel_df = df_filtered[[col_x, col_y, col_color]].dropna()
                st.pyplot(fig)

                if len(rel_df) >= 3:
                    corr_xy = rel_df[col_x].corr(rel_df[col_y])
                    relation_strength = describe_correlation(corr_xy)
                    trend = "directă" if corr_xy > 0 else "inversă"
                    if corr_xy > 0.6:
                        rel_interp = f"O creștere a caracteristicii **{col_x}** atrage aproape garantat după sine o creștere pentru **{col_y}**."
                    elif corr_xy < -0.6:
                        rel_interp = f"Observăm un compromis clar tehnologic (trade-off): pe măsură ce **{col_x}** crește, **{col_y}** tinde să scadă puternic."
                    else:
                        rel_interp = f"Nu există o constrângere fizică clară care să lege direct **{col_x}** de **{col_y}**. Producătorii auto abordează aceste specificații independent."

                    st.markdown(
                        f"""
**Interpretarea aplicată a corelației:**
- **Natura relației:** Dinamica dintre cele două elemente este **{relation_strength}** și **{trend}** (Corelație r = **{corr_xy:.2f}**).
- **Semnificație pentru industrie:** *{rel_interp}* Un inginer auto ar putea citi acest grafic pentru a vizualiza cum o alegere de design o influențază inevitabil pe alta.
"""
                    )
                    if col_color != "Fără" and col_color in rel_df.columns:
                        best_group = rel_df.groupby(col_color)[col_y].mean().sort_values(ascending=False).head(1)
                        if not best_group.empty:
                            st.caption(
                                f"Categorie cu media cea mai mare pentru {col_y}: {best_group.index[0]} ({best_group.iloc[0]:.2f})."
                            )

        with tab4:
            st.subheader("Matricea de Corelație")
            if numeric_cols:
                corr_matrix = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(16, 12))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)

                if len(numeric_cols) >= 2:
                    corr_pairs = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).stack()
                    if not corr_pairs.empty:
                        strongest_pos = corr_pairs.idxmax()
                        strongest_neg = corr_pairs.idxmin()
                        strongest_pos_val = corr_pairs.max()
                        strongest_neg_val = corr_pairs.min()

                        st.markdown(
                            f"""
**Interpretarea aplicată a matricii termice (Heatmap):**
- **Sinergia tehnică absolută:** Cea mai puternică legătură pozitivă este între **{strongest_pos[0]}** și **{strongest_pos[1]}** (r = **{strongest_pos_val:.2f}**). În producția auto, aceste specificații cresc invariabil împreună. Reprezintă de cele mai multe ori limitări fizice pure (ex: motor mai mare -> greutate mai mare), nu simple decizii de design.
- **Cel mai sever compromis:** Cea mai dură relație negativă este **{strongest_neg[0]}** vs. **{strongest_neg[1]}** (r = **{strongest_neg_val:.2f}**). Aici inginerii se luptă cu legile fizicii, deoarece accentuarea unuia dintre acești factori duce inevitabil la scăderea celuilalt (clasicul trade-off performanță vs. eficiență).
- Pătrățelele în nuanțe pale reprezintă arii de inovație – variabile independente care oferă producătorilor libertatea de a inova fără restricții.
"""
                        )

        with tab5:
            st.subheader("Analiza Categorială")
            if categorical_cols:
                col_cat_view = st.selectbox("Variabilă:", categorical_cols)
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                counts = df[col_cat_view].value_counts().nlargest(15)
                sns.barplot(x=counts.values, y=counts.index, palette="mako", ax=ax[0])
                counts_pie = df[col_cat_view].value_counts().nlargest(8)
                ax[1].pie(counts_pie, labels=counts_pie.index, autopct='%1.1f%%')
                st.pyplot(fig)

                if not counts.empty:
                    total_count = df[col_cat_view].dropna().shape[0]
                    top_cat = counts.index[0]
                    top_val = counts.iloc[0]
                    top_share = (top_val / total_count) * 100 if total_count else 0
                    top3_share = (counts.head(3).sum() / total_count) * 100 if total_count else 0

                    if top3_share > 70:
                        monopoly_text = "Această variabilă arată o puternică monopolizare a pieței – constructorii preferă siguranța și se limitează la opțiunile dovedite a fi extrem de cerute."
                    else:
                        monopoly_text = "Piața pentru această caracteristică este foarte fragmentată, demonstrând că nu există o rețetă supremă a succesului, clienții având preferințe extrem de variate."

                    unique_cats = len(df[col_cat_view].dropna().unique())
                    if unique_cats > 3:
                        tail_cats = unique_cats - 3
                        tail_share = 100 - top3_share
                        tail_text = f"Restul de **{tail_cats} variante** (dincolo de top 3) se luptă acerb pe o felie de piață de doar **{tail_share:.2f}%**. Acestea reprezintă de obicei *nișe de lux*, *proiecte experimentale* sau branduri exotice."
                    else:
                        tail_text = "Variabila este limitată strict la aceste top categorii, arătând un ecosistem auto închis, cu reguli clare."

                    st.markdown(
                        f"""
**Interpretarea comportamentului de piață:**
- **Standardul de necontestat:** Varianta **{top_cat}** acaparează de una singură **{top_share:.2f}%** din industria analizată (reprezentând **{top_val}** de modele). Aceasta este "alegerea sigură" a pieței și zona de confort financiar a oricărui producător.
- **Gradul de concentrare (Top 3):** Cele mai populare 3 variante domină cumulat **{top3_share:.2f}%** din ofertă. *{monopoly_text}*
- **Analiza Nișelor (Coada pieței):** {tail_text}
"""
                    )

        with tab6:
            st.subheader("Pair Plots")
            multi_cols = st.multiselect("Selectează (max 4):", numeric_cols, default=numeric_cols[:3])
            if len(multi_cols) > 1 and st.button("Generează matrice"):
                pair_df = df[multi_cols].dropna()
                g = sns.pairplot(pair_df, diag_kind="kde", corner=False)
                st.pyplot(g.fig)

                pair_corr = pair_df.corr().where(~np.eye(len(multi_cols), dtype=bool)).stack()
                if not pair_corr.empty:
                    best_pair = pair_corr.abs().idxmax()
                    best_pair_val = pair_df[[best_pair[0], best_pair[1]]].corr().iloc[0, 1]
                    st.markdown(
                        f"""
**Imaginea de ansamblu (Tabloul de bord multidimensional):**
- **Analiza densității:** Diagonala matricei evidențiază concentrarea ofertei din piață. Curbele de distribuție (KDE) arată clar unde se situează standardul de fabricație pentru majoritatea producătorilor.
- **Sinergii dominante:** Dintre specificațiile selectate, cuplul **{best_pair[0]}** și **{best_pair[1]}** demonstrează cea mai mare interdependență (corelație r = **{best_pair_val:.2f}**). Aceasta sugerează o constrângere arhitecturală, unde modificarea unui element o forțează pe cealaltă.
- **Identificarea spațiilor albe:** Intersecțiile goale din grafic sunt esențiale pentru strategie. Ele reprezintă combinații tehnice inexistente, indicând fie limite fizice/economice insurmontabile, fie oportunități de nișă încă neexploatate de competiție.
"""
                    )

        with tab7:
            st.subheader("Comparație Distribuții pe Categorii (Violin Plot)")
            if numeric_cols and categorical_cols:
                c1, c2, c3 = st.columns(3)
                with c1:
                    violin_num = st.selectbox("Variabilă numerică:", numeric_cols, key="violin_num")
                with c2:
                    violin_cat = st.selectbox("Variabilă categorială:", categorical_cols, key="violin_cat")
                with c3:
                    max_cats = st.slider("Număr categorii afișate:", min_value=3, max_value=12, value=8,
                                         key="violin_top")

                top_categories = df[violin_cat].value_counts().nlargest(max_cats).index
                violin_df = df[df[violin_cat].isin(top_categories)][[violin_cat, violin_num]].dropna()

                if violin_df.empty:
                    st.warning("Nu există suficiente date pentru combinația selectată.")
                else:
                    fig, ax = plt.subplots(figsize=(15, 8))
                    sns.violinplot(
                        data=violin_df,
                        x=violin_cat,
                        y=violin_num,
                        inner="quartile",
                        cut=0,
                        palette="Set2",
                        ax=ax
                    )
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    median_by_cat = violin_df.groupby(violin_cat)[violin_num].median().sort_values(ascending=False)
                    iqr_by_cat = violin_df.groupby(violin_cat)[violin_num].quantile(0.75) - \
                                 violin_df.groupby(violin_cat)[violin_num].quantile(0.25)
                    most_variable = iqr_by_cat.sort_values(ascending=False).index[0]

                    least_variable = iqr_by_cat.sort_values(ascending=True).index[0]

                    st.markdown(
                        f"""
**Interpretarea practică a graficului "Violin":**
Acest grafic vizualizează "forma" pieței. Acolo unde forma este foarte lată (bombată), se află cea mai mare aglomerare de autovehicule.

- **Standardul de top:** Analizând **{violin_num}**, observăm că segmentul **{median_by_cat.index[0]}** impune cel mai exigent standard pieței (cu o mediană centrală de **{median_by_cat.iloc[0]:.2f}**). Cumpărătorii de aici au așteptări de bază foarte ridicate.
- **Piața fragmentată (Experimentare):** Categoria **{most_variable}** are cea mai alungită și neregulată formă (cel mai mare ecartament între valorile maxime și minime). Asta denotă inconsecvență: producătorii experimentează intens și lansează în această categorie modele cu dotări radical diferite.
- **Piața standardizată (Conformitate):** La polul opus, categoria **{least_variable}** are cea mai subțire/comprimată formă. În acest segment, regulile sunt stricte și toți producătorii construiesc mașinile urmând exact aceeași rețetă tehnologică, evitând riscurile.
"""
                    )
            else:
                st.warning("Pentru acest tip de grafic este nevoie de cel puțin o coloană numerică și una categorială.")

# ---------------------------
# Secțiunea: Analiză Statistică (Regresie Multiplă)
# ---------------------------
elif section == "Analiză Statistică (Regresie)":
    st.header("Analiză Statistică: Regresie Liniară Simplă/Multiplă")
    st.write(
        "În această secțiune folosim modelarea statistică pentru a înțelege cum parametrii tehnici influențează consumul de combustibil.")

    # Pregătirea datelor pentru regresie
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Target implicit: L/100km (Avg)
    default_target = "L/100km (Avg)" if "L/100km (Avg)" in numeric_cols else (numeric_cols[0] if numeric_cols else None)

    if len(numeric_cols) < 2:
        st.warning("⚠️ Ai nevoie de cel puțin 2 coloane numerice pentru a rula o regresie.")
    else:
        st.subheader("1. Configurare Model")

        col1, col2 = st.columns(2)
        with col1:
            target_var = st.selectbox("Variabila dependentă (Target - Y):", numeric_cols,
                                      index=numeric_cols.index(default_target) if default_target in numeric_cols else 0)
        with col2:
            potential_features = [c for c in numeric_cols if c != target_var]
            selected_features = st.multiselect("Variabile independente (Predictori - X):", potential_features,
                                               default=potential_features[:4] if len(
                                                   potential_features) >= 4 else potential_features)

        if selected_features:
            # Drop NAs for specifically selected columns
            reg_data = df[selected_features + [target_var]].dropna()

            if reg_data.empty:
                st.error("❌ Setul de date rezultat este gol după eliminarea valorilor lipsă. Verifică datele!")
            else:
                st.write(
                    f"Modelul va fi antrenat pe **{len(reg_data)}** observații (după eliminarea rândurilor cu valori lipsă).")

                # --- Antrenare Model ---
                X = reg_data[selected_features]
                y = reg_data[target_var]
                X = sm.add_constant(X)  # Adăugăm interceptul (constanta)

                model = sm.OLS(y, X).fit()

                # --- Rezultate ---
                st.subheader("2. Rezultatul Modelului (OLS Summary)")
                st.text(str(model.summary()))

                # --- Interpretare ---
                with st.expander("📝 Interpretarea Rezultatelor", expanded=False):
                    st.markdown(f"""
1.  **R-squared (Coeficientul de Determinare):** **{model.rsquared:.4f}**
    *   *Semnificație:* Modelul explică aproximativ **{model.rsquared * 100:.1f}%** din variația variabilei **{target_var}**. Cu cât e mai aproape de 1, cu atât modelul e mai precis.
2.  **Coeficienți (coef):**
    *   Arată cât de mult se modifică **{target_var}** la o creștere de o unitate a predictorului respectiv, menținând ceilalți factori constanți.
    *   Dacă coeficientul este **pozitiv**, variabila crește target-ul. Dacă e **negativ**, îl scade.
3.  **P-value (P>|t|):**
    *   Dacă este sub **0.05**, variabila este **semnificativă statistic**. Dacă e peste 0.05, acea variabilă s-ar putea să nu aibă o influență reală în model.
                    """)

                # --- Calculator Interactiv ---
                st.subheader(f"3. 🧮 Calculator de Predicție: {target_var}")
                st.info("Introdu parametri personalizați mai jos pentru a vedea ce valoare estimează modelul:")

                input_data = {}
                cols = st.columns(len(selected_features))
                for i, feature in enumerate(selected_features):
                    with cols[i % len(cols)]:
                        min_v = float(df[feature].min())
                        max_v = float(df[feature].max())
                        mean_v = float(df[feature].mean())
                        input_data[feature] = st.number_input(f"{feature}:", min_value=min_v, max_value=max_v,
                                                              value=mean_v, key=f"pred_{feature}")

                # Predictie
                input_df = pd.DataFrame([input_data])
                # Adaugam constanta manual pentru predictie
                input_df.insert(0, 'const', 1.0)

                try:
                    prediction = model.predict(input_df)[0]
                    st.success(f"Valoarea estimată pentru **{target_var}** este: **{prediction:.2f}**")
                except Exception as e:
                    st.error(f"Eroare la predicție: {e}")

                st.subheader("4. Analiza Reziduurilor și Performanței")

                c_p1, c_p2 = st.columns(2)

                with c_p1:
                    st.write("**Distribuția erorilor (Reziduurilor)**")
                    fig1, ax1 = plt.subplots()
                    sns.histplot(model.resid, kde=True, ax=ax1, color="#e74c3c")
                    st.pyplot(fig1)
                    st.caption("Pentru un model bun, erorile ar trebui să fie distribuite normal în jurul valorii 0.")

                with c_p2:
                    st.write("**Valori Reale vs Valori Predise**")
                    y_pred_all = model.predict(X)
                    fig2, ax2 = plt.subplots()
                    sns.scatterplot(x=y, y=y_pred_all, alpha=0.5, ax=ax2)
                    line_min = min(y.min(), y_pred_all.min())
                    line_max = max(y.max(), y_pred_all.max())
                    ax2.plot([line_min, line_max], [line_min, line_max], '--r', lw=2)
                    plt.xlabel("Valoare Reală")
                    plt.ylabel("Valoare Predisă")
                    st.pyplot(fig2)
                    st.caption("Cu cât punctele sunt mai aproape de linia roșie, cu atât modelul este mai precis.")

        else:
            st.info("Selectează variabilele independente (X) pentru a genera modelul.")

# ---------------------------
# Secțiunea: Clusterizare (K-Means)
# ---------------------------
elif section == "Clusterizare (K-Means)":
    st.header("Algoritm de Machine Learning: K-Means Clustering")
    st.write(
        "Acest algoritm ne supervizat grupează mașinile cu profiluri tehnice similare fără a cunoaște etichetele inițiale (cum ar fi tipul caroseriei).")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Ai nevoie de cel puțin 2 coloane numerice pentru a rula K-Means.")
    else:
        st.subheader("1. Selecția Variabilelor și Găsirea Numărului de Clustere ($k$)")
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            var_1 = st.selectbox("Alege variabila 1 (Axa X):", numeric_cols, index=0)
        with col_k2:
            default_idx2 = 1 if len(numeric_cols) > 1 else 0
            var_2 = st.selectbox("Alege variabila 2 (Axa Y):", numeric_cols, index=default_idx2)

        if var_1 != var_2:
            # Preluăm setul și drop missing values
            cluster_data = df[[var_1, var_2]].dropna()

            if len(cluster_data) > 10:
                # Extragem valorile a la `kmeans.py`
                X = cluster_data.values

                # Scalare cerută de procesul de clusterizare ML a la `kmeans.py`
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                if st.button("📈 Rulează Metoda Elbow (Inertia)"):
                    with st.spinner("Se calculează the Elbow Method..."):
                        wcss = []
                        max_clusters = min(11, len(X_scaled))
                        for i in range(1, max_clusters):
                            kmeans_test = KMeans(n_clusters=i, init='k-means++', random_state=42)
                            kmeans_test.fit(X_scaled)
                            wcss.append(kmeans_test.inertia_)

                        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 5))
                        sns.lineplot(x=range(1, max_clusters), y=wcss, marker='o', color='red', ax=ax_elbow)
                        ax_elbow.set_title('The Elbow Method')
                        ax_elbow.set_xlabel('Number of clusters')
                        ax_elbow.set_ylabel('WCSS')
                        st.pyplot(fig_elbow)
                        st.markdown("""
- Acest grafic ne ajută să găsim "numărul natural" de grupuri din piață. 
- Punctul de **„cot”** reprezintă momentul în care adăugarea unui nou grup nu mai aduce o îmbunătățire semnificativă a preciziei. 
- În context auto, un "cot" la valoarea 3 ar putea sugera că piața se împarte fundamental în: *Ieftin/Eficienț*, *Mediu* și *Lux/Performanță*.
""")

                st.subheader("2. Parametrizarea și Antrenarea Modelului")
                n_clusters = st.slider("Alege numărul de clustere (K):", min_value=2, max_value=8, value=3)

                # Fitting K-Means to the dataset explicit
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
                y_kmeans = kmeans.fit_predict(X_scaled)

                st.subheader("3. Vizualizarea Clusterelor")
                # Vizualizare fix ca în `kmeans.py` dar adaptată general pentru culori
                fig_c, ax_c = plt.subplots(figsize=(15, 7))

                colors = sns.color_palette("husl", n_clusters)
                for i in range(n_clusters):
                    sns.scatterplot(
                        x=X_scaled[y_kmeans == i, 0],
                        y=X_scaled[y_kmeans == i, 1],
                        color=colors[i],
                        label=f'Cluster {i + 1}',
                        s=50,
                        ax=ax_c
                    )

                # Afisare centroizi
                sns.scatterplot(
                    x=kmeans.cluster_centers_[:, 0],
                    y=kmeans.cluster_centers_[:, 1],
                    color='red',
                    label='Centroids',
                    s=300,
                    marker='X',
                    ax=ax_c
                )

                ax_c.grid(False)
                ax_c.set_title('Clusters of cars (Scaled)')
                ax_c.set_xlabel(f'{var_1} (Standardized)')
                ax_c.set_ylabel(f'{var_2} (Standardized)')
                plt.legend()
                st.pyplot(fig_c)

                st.markdown(f"""
**Ce observăm pe harta clusterelor:**
- **Punctele colorate:** Reprezintă mașinile grupate de algoritm. Cu cât sunt mai strâns grupate, cu atât acele mașini sunt mai similare între ele din perspectiva **{var_1}** și **{var_2}**.
- **Centroizii (X roșu):** Aceștia reprezintă "mașina teoretică ideală" a acelui grup. Dacă vrei să lansezi un model nou care să concureze în Clusterul 1, specificațiile tale ar trebui să fie cât mai aproape de coordonatele acestui X.
""")

                st.subheader("4. Evaluarea Modelului")
                # Silhouette Score
                sil_score = silhouette_score(X_scaled, y_kmeans)
                st.success(f"**Silhouette Score:** **{sil_score:.4f}**")

                with st.expander("Ce înseamnă Silhouette Score?"):
                    st.markdown("""
- Măsoară cât de apropiat este un punct de clusterul său comparativ cu celelalte clustere (între `-1` și `1`).
- **Aproape de 1:** Punctele sunt bine încadrate și delimitate clar față de clusterele vecine.
- **Aproape de 0:** Punctele se află la granița dintre două clustere (overlap).
- **Sub 0:** Punctele sunt cel mai probabil atribuite greșit.
                    """)

                # Afisare metru normal: Centri nescalați
                cluster_data['Cluster'] = [f"Cluster {i + 1}" for i in y_kmeans]
                st.write(f"🚗 Profilul Mediilor pe fiecare Cluster ({var_1} vs {var_2}):")
                profile_df = cluster_data.groupby('Cluster').mean()
                st.dataframe(profile_df)

            else:
                st.warning("Variabilele ridică un set de date prea mic (prea multe valori lipsă) pentru învățare.")
        else:
            st.error("Selectează două variabile distincte!")

# ---------------------------
# Secțiunea: Clusterizare (Ierarhică - HC)
# ---------------------------
elif section == "Clusterizare (Ierarhică - HC)":
    st.header("Clusterizare Ierarhică (HC) - analiză reinterpretată")
    st.write(
        "Modelul grupează mașinile folosind doar două variabile numerice (fără etichete) și compară cu o etichetă reală.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Ai nevoie de cel puțin 2 coloane numerice pentru HC.")
    else:
        st.subheader("1. Configurare")
        c1, c2, c3 = st.columns(3)
        with c1:
            var_1 = st.selectbox("Variabila 1 (X):", numeric_cols, index=0, key="hc_var1")
        with c2:
            default_idx2 = 1 if len(numeric_cols) > 1 else 0
            var_2 = st.selectbox("Variabila 2 (Y):", numeric_cols, index=default_idx2, key="hc_var2")
        with c3:
            compare_options = ["(Fără comparație)"] + categorical_cols
            default_cat = "Body style" if "Body style" in categorical_cols else compare_options[0]
            cat_label = st.selectbox(
                "Compară cu etichetă reală:",
                compare_options,
                index=compare_options.index(default_cat),
                key="hc_cat"
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            use_outlier_filter = st.checkbox("Filtru outliers 1.5×IQR", value=True, key="hc_use_iqr")
        with c5:
            dendro_sample = st.slider("Max puncte pentru dendrogramă", 300, 3000, 1200, step=100, key="hc_sample")
        with c6:
            k_hc = st.slider("Număr clustere (K)", 2, 8, 3, key="hc_k")

        if var_1 == var_2:
            st.error("Selectează două variabile numerice distincte.")
        else:
            cols_needed = [var_1, var_2] + ([] if cat_label == "(Fără comparație)" else [cat_label])
            hc_data = df[cols_needed].dropna().copy()

            if hc_data.empty or len(hc_data) < 15:
                st.warning("Setul rezultat este prea mic după eliminarea valorilor lipsă. Alege alte variabile.")
            else:
                initial_n = len(hc_data)
                if use_outlier_filter:
                    q1 = hc_data[[var_1, var_2]].quantile(0.25)
                    q3 = hc_data[[var_1, var_2]].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    inlier_mask = ~((hc_data[[var_1, var_2]] < lower) | (hc_data[[var_1, var_2]] > upper)).any(axis=1)
                    hc_data = hc_data.loc[inlier_mask].copy()
                    removed_outliers = initial_n - len(hc_data)
                    st.info(
                        f"Filtru IQR activ: {removed_outliers} observații eliminate, {len(hc_data)} observații rămase pentru model.")
                else:
                    st.info(f"Fără filtru outliers: modelul rulează pe {len(hc_data)} observații.")

                if len(hc_data) < max(15, k_hc * 5):
                    st.warning("Date insuficiente pentru un HC stabil cu K ales. Reduce K sau schimbă variabilele.")
                else:
                    scaler_hc = StandardScaler()
                    X_scaled_hc = scaler_hc.fit_transform(hc_data[[var_1, var_2]].values)

                    # Dendrograma e calculată pe eșantion pentru performanță, modelul final pe toate punctele.
                    if len(hc_data) > dendro_sample:
                        dendro_data = hc_data.sample(n=dendro_sample, random_state=42)
                    else:
                        dendro_data = hc_data.copy()

                    X_scaled_dendro = scaler_hc.transform(dendro_data[[var_1, var_2]].values)

                    st.subheader("2. Dendrogramă (Ward)")
                    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
                    linked = linkage(X_scaled_dendro, method="ward")

                    # Cut the dendrogram colors exactly at the distance that forms K clusters
                    cut_distance = linked[-k_hc, 2] if k_hc < len(linked) else 0

                    dendrogram(linked, truncate_mode="lastp", p=30, color_threshold=cut_distance, ax=ax_dendro)
                    ax_dendro.set_title(f"Dendrogramă HC - {var_1} vs {var_2} (Colorată pentru {k_hc} clustere)")
                    ax_dendro.set_ylabel("Distanță de fuziune")
                    st.pyplot(fig_dendro)
                    st.markdown(f"""

- **Axa Verticală (Distanța):** Indică gradul de diferențiere. Cu cât două ramuri se unesc mai sus, cu atât mașinile din acele grupuri sunt mai diferite din punct de vedere tehnic (**{var_1}** vs **{var_2}**).
- **Liniile Orizontale:** Reprezintă momentele de "fuziune". O linie lungă verticală indică o separare clară între segmentele de piață (de exemplu, o ruptură clară între utilitare și mașini sport).
- **Tăierea arborelui:** Numărul de linii verticale pe care le intersectăm dacă tragem o linie orizontală imaginară ne spune în câte clustere am împărțit piața.
""")

                    hc_model = AgglomerativeClustering(n_clusters=k_hc, metric="euclidean", linkage="ward")
                    y_hc = hc_model.fit_predict(X_scaled_hc)

                    hc_data["Cluster"] = [f"Cluster {i + 1}" for i in y_hc]
                    hc_data["HC_X"] = X_scaled_hc[:, 0]
                    hc_data["HC_Y"] = X_scaled_hc[:, 1]

                    st.subheader("3. Vizualizare")
                    v1, v2 = st.columns(2)
                    with v1:
                        st.write("**A. Clusterele calculate de model**")
                        fig_hc, ax_hc = plt.subplots(figsize=(6, 5))
                        sns.scatterplot(
                            data=hc_data,
                            x="HC_X",
                            y="HC_Y",
                            hue="Cluster",
                            palette="Set2",
                            s=40,
                            alpha=0.7,
                            ax=ax_hc
                        )
                        ax_hc.set_xlabel(f"{var_1} (standardizat)")
                        ax_hc.set_ylabel(f"{var_2} (standardizat)")
                        st.pyplot(fig_hc)

                    with v2:
                        if cat_label != "(Fără comparație)":
                            st.write(f"**B. Eticheta reală: {cat_label}**")
                            fig_true, ax_true = plt.subplots(figsize=(6, 5))
                            top_real_cats = hc_data[cat_label].value_counts().nlargest(7).index
                            plot_real_data = hc_data[hc_data[cat_label].isin(top_real_cats)]
                            sns.scatterplot(
                                data=plot_real_data,
                                x="HC_X",
                                y="HC_Y",
                                hue=cat_label,
                                palette="tab10",
                                s=40,
                                alpha=0.7,
                                ax=ax_true
                            )
                            ax_true.set_xlabel(f"{var_1} (standardizat)")
                            ax_true.set_ylabel(f"{var_2} (standardizat)")
                            st.pyplot(fig_true)
                        else:
                            st.info("Nu ai selectat etichetă reală pentru comparație vizuală.")

                    st.subheader("4. Interpretare")
                    sil = silhouette_score(X_scaled_hc, y_hc)
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Observații folosite", f"{len(hc_data)}")
                    with m2:
                        st.metric("Silhouette", f"{sil:.3f}")

                    ari_val = None
                    purity_val = None
                    if cat_label != "(Fără comparație)" and hc_data[cat_label].nunique() > 1:
                        ari_val = adjusted_rand_score(hc_data[cat_label].astype(str), y_hc)
                        ct_counts = pd.crosstab(hc_data["Cluster"], hc_data[cat_label])
                        purity_val = (ct_counts.max(axis=1).sum() / ct_counts.values.sum()) * 100
                        with m3:
                            st.metric("Puritate globală", f"{purity_val:.1f}%")
                    else:
                        with m3:
                            st.metric("Puritate globală", "N/A")

                    size_df = hc_data["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(
                        name="Număr")
                    size_df["Proporție (%)"] = (size_df["Număr"] / len(hc_data) * 100).round(1)
                    st.markdown("**Dimensiunea clusterelor**")
                    st.dataframe(size_df, use_container_width=True)

                    profile_df = hc_data.groupby("Cluster")[[var_1, var_2]].agg(["mean", "median"]).round(2)
                    st.markdown(f"**Profil numeric pe clustere ({var_1}, {var_2})**")
                    st.dataframe(profile_df, use_container_width=True)

                    if cat_label != "(Fără comparație)":
                        if ari_val is not None:
                            st.info(
                                f"💡 **OBSERVAȚIE - Scorul ARI: {ari_val:.3f}**\n\nAcest scor matematic (*Adjusted Rand Index*) arată cât de exact a intuit algoritmul etichetele reale din fabrică.\n* O valoare aproape de **1.0** înseamnă suprapunere perfectă (gruparea geometrică e identică cu cea din catalogul auto).\n* O valoare aproape de **0.0** indică o suprapunere slabă (categoriile puse de producător nu prea au legătură matematică cu selecția ta).")

# ---------------------------
# Secțiunea: Clasificare Predictivă (ML)
# ---------------------------
elif section == "Clasificare Predictivă (ML)":
    st.header("Clasificare Predictivă (ML Supervizat)")
    st.write(
        "Acest modul antrenează algoritmi care învață să prezică o categorie (cum ar fi Segmentul sau Body Style) folosind fix atributele tehnice care i sunt oferite spre analiză.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(numeric_cols) < 1 or len(categorical_cols) < 1:
        st.warning("E nevoie de coloane numerice și categoriale pentru funcționare.")
    else:
        st.subheader("Configurare Set de Date")
        target_col = st.selectbox("Alege ce vrei să prezică algoritmul (Target Y):", categorical_cols,
                                  index=categorical_cols.index('Segment') if 'Segment' in categorical_cols else 0)

        default_features = ['Top Speed', 'Power(HP)',
                            'Width'] if 'Top Speed' in numeric_cols and 'Power(HP)' in numeric_cols and 'Width' in numeric_cols else numeric_cols[
                                                                                                                                     :2]
        feature_cols = st.multiselect("Alege caracteristicile tehnice din care să învețe algoritmul (Features X):",
                                      numeric_cols, default=default_features)

        if len(feature_cols) >= 1:
            # Curățăm doar rândurile lipsă de pe coloanele vizate folosind o funcție rapidă
            ml_data = df[feature_cols + [target_col]].dropna()

            if len(ml_data) > 50:
                X = ml_data[feature_cols].values
                y = ml_data[target_col].values

                # Split pentru validare cruzată - 20% test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                tab1, tab2, tab3 = st.tabs(
                    ["🌳 1. Arbori de Decizie (Interpretare)", "🌲🌲 2. Random Forest (Performanță)",
                     "⚖️ 3. Regresie Logistică (Șanse)"])

                with tab1:
                    st.markdown("### Arbore de Decizie (Analiza Logicii Algoritmice)")
                    st.write(
                        "Acest model este extrem de transparent pentru mintea umană, deoarece ia decizii bazate pe ramuri logice secvențiale (Ex: Dacă e lat -> Dacă are cai putere -> Deduc că e SUV).")

                    depth = st.slider("Alege Adâncimea Maximă a Arborelui Desenat:", 2, 10,
                                      7)  # Setați la 7 după cerința utilizator

                    dt_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
                    dt_model.fit(X_train, y_train)

                    y_pred_dt = dt_model.predict(X_test)
                    acc_dt = accuracy_score(y_test, y_pred_dt)

                    st.success(f"**Acuratețe de testare (Precizie strict pe date nevăzute)**: {acc_dt:.2%}")

                    st.markdown("#### Schema Decizională a Modelului")
                    fig_tree, ax_tree = plt.subplots(figsize=(24, 12))
                    try:
                        classes_str = [str(c) for c in dt_model.classes_]
                        plot_tree(dt_model, feature_names=feature_cols, class_names=classes_str, filled=True,
                                  rounded=True, ax=ax_tree, fontsize=7)
                        st.pyplot(fig_tree)
                        st.markdown("""
1. **Punctul de pornire:** Algoritmul a analizat datele și a descoperit singur "întrebarea supremă" (afișată în prima cutie de sus). Acela este factorul absolut care clasifică binar cel mai bine piața auto!
2. **Traseul (IF-THEN):** Fiecare cutie reprezintă o decizie tehnică. Dacă "adevărat", mergi pe ramura din stânga. Astfel, poți extrage "rețeta" pentru a construi un anumit tip de mașină, pas cu pas.
3. **Frunzele finale:** Cutiile de jos, colorate intens, sunt predicții 100% sigure. Modelul ne arată practic logica pur matematică prin care industria împarte clasele de vehicule.
""")
                    except Exception as e:
                        st.error(
                            f"Eroare la desenarea structurii (posibil format necorespunzător al stringurilor): {str(e)}")

                with tab2:
                    st.markdown("### Random Forest (Top Acuratețe)")
                    st.write(
                        "În loc să deseneze 1 arbore, acesta antrenează în orb 100 de abori diferiți care 'votează' cel mai bun rezultat.")

                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)

                    y_pred_rf = rf_model.predict(X_test)
                    acc_rf = accuracy_score(y_test, y_pred_rf)

                    st.success(f"**Acuratețe de testare Random Forest**: {acc_rf:.2%}")

                    st.markdown("#### Care Factor Contează Cel Mai Mult?")
                    importances = rf_model.feature_importances_
                    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(
                        by='Importance', ascending=False)

                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=imp_df, x='Importance', y='Feature', palette='plasma', ax=ax_imp)
                    ax_imp.set_title(f"Ponderea factorilor în predicția `{target_col}`")
                    st.pyplot(fig_imp)

                    top_feature = imp_df.iloc[0]['Feature']
                    top_importance = imp_df.iloc[0]['Importance']
                    st.markdown(f"""

- **Cel mai puternic factor:** Algoritmul demonstrează clar că **{top_feature}** este detaliul tehnic cu cel mai mare impact (o pondere uriașă de **{top_importance:.1%}**) atunci când vrem să definim sau să prezicem **{target_col}**. În linii mari, producătorii definesc identitatea acestor clase auto în primul rând în funcție de caracteristica {top_feature}.
- Dacă vrei să proiectezi un vehicul într-o anumită categorie, graficul de mai sus îți arată clar unde NU ai voie să faci compromisuri la design/costuri și unde ai mână liberă.
""")

                    st.markdown("#### Matrice de confuzie")
                    try:
                        y_true_series = pd.Series(y_test).astype(str)
                        y_pred_series = pd.Series(y_pred_rf).astype(str)
                        all_labels = sorted(set(y_true_series).union(set(y_pred_series)))

                        if len(all_labels) > 25:
                            top_labels = y_true_series.value_counts().head(24).index.tolist()
                            y_true_plot = y_true_series.where(y_true_series.isin(top_labels), "...")
                            y_pred_plot = y_pred_series.where(y_pred_series.isin(top_labels), "...")
                            labels_for_cm = top_labels + ["..."]
                            st.info("Afișare limitată la 25 clase: primele 24 clase după frecvență + categoria '...'.")
                        else:
                            y_true_plot = y_true_series
                            y_pred_plot = y_pred_series
                            model_order = [str(lbl) for lbl in rf_model.classes_]
                            labels_for_cm = [lbl for lbl in model_order if lbl in all_labels]

                        cm = confusion_matrix(y_true_plot, y_pred_plot, labels=labels_for_cm)
                        fig_cm, ax_cm = plt.subplots(figsize=(10, 7))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', xticklabels=labels_for_cm,
                                    yticklabels=labels_for_cm, ax=ax_cm)
                        ax_cm.set_ylabel("Adevăr (clasa reală din date)")
                        ax_cm.set_xlabel("Predicție (clasa estimată de model)")
                        st.pyplot(fig_cm)
                        st.markdown("""

- **Diagonala roșie:** Astea sunt "victoriile" modelului (ex: era SUV și algoritmul a confirmat că e SUV pe baza lățimii și greutății).
- **Căsuțele din afara diagonalei (Confuziile):** Când modelul se păcălește constant (ex: clasifică un *Sedan* drept *Coupe*), acesta NU este neapărat un eșec al algoritmului, ci o revelație genială de business! Înseamnă că, tehnic vorbind, inginerii construiesc acele două clase pe aproape același șasiu / cu aceleași specificații. Acest lucru expune o "canibalizare" tehnologică între modelele de pe piață.
""")
                    except Exception as e:
                        st.warning("Prea multe categorii pentru a desena eficient heatmap-ul.")

                with tab3:
                    st.markdown("### Regresie Logistică (Probabilități și Șanse)")
                    st.write(
                        "Acest model folosește coeficienți matematici pentru a calcula o probabilitate procentuală a apartenenței unui autovehicul la un anumit grup.")

                    # Scalare date - obligatoriu pentru Regresia Logistică pentru convergență bună
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    lr_model = LogisticRegression(max_iter=1000, random_state=42)
                    lr_model.fit(X_train_scaled, y_train)

                    y_pred_lr = lr_model.predict(X_test_scaled)
                    acc_lr = accuracy_score(y_test, y_pred_lr)

                    st.success(f"**Acuratețe de testare Regresie Logistică**: {acc_lr:.2%}")

                    st.markdown("#### Direcția de Influență a Specificațiilor (Coeficienți)")
                    st.write(
                        "Spre deosebire de Random Forest, aici putem vedea **cum** influențează fiecare specificație. Valorile verzi (+) cresc probabilitatea ca mașina să aparțină de acea clasă, iar cele roșii (-) o scad.")

                    try:
                        coefs = lr_model.coef_
                        classes = lr_model.classes_

                        if len(classes) == 2:
                            # Clasificare binară
                            coef_df = pd.DataFrame({'Specificație': feature_cols, 'Coeficient': coefs[0]})
                            coef_df = coef_df.sort_values(by='Coeficient', ascending=False)
                            fig_coef, ax_coef = plt.subplots(figsize=(10, max(4, len(feature_cols) * 0.5)))
                            colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_df['Coeficient']]
                            sns.barplot(x='Coeficient', y='Specificație', data=coef_df, palette=colors, ax=ax_coef)
                            ax_coef.axvline(0, color='black', linewidth=1)
                            st.pyplot(fig_coef)
                        else:
                            # Clasificare multiclasă -> Heatmap
                            coef_df = pd.DataFrame(coefs, index=classes, columns=feature_cols)
                            # Dacă sunt prea multe clase (ex. zeci de Marci), le limităm la top 10
                            if len(classes) > 15:
                                top_classes = pd.Series(y_train).value_counts().head(12).index.tolist()
                                coef_df = coef_df.loc[[c for c in top_classes if c in coef_df.index]]
                                st.info("Afișăm coeficienții doar pentru primele 12 cele mai frecvente clase.")

                            fig_coef, ax_coef = plt.subplots(figsize=(10, max(4, len(coef_df) * 0.6)))
                            sns.heatmap(coef_df, cmap="RdYlGn", center=0, annot=True, fmt=".2f",
                                        cbar_kws={'label': 'Impact (Coeficient)'}, ax=ax_coef)
                            ax_coef.set_ylabel("Clasa Prezisă")
                            st.pyplot(fig_coef)
                    except Exception as e:
                        st.info("Nu am putut genera graficul coeficienților.")

                    st.markdown("#### 🎛️ Simulator Live de Probabilități (What-If Analysis)")
                    st.write(
                        "Modifică valorile de mai jos pentru a construi un autovehicul virtual. Modelul îți va calcula live șansele de apartenență la fiecare clasă (pe baza a ce a învățat anterior).")

                    num_features = len(feature_cols)
                    cols = st.columns(num_features if num_features <= 4 else 4)

                    user_inputs = []
                    for i, feature in enumerate(feature_cols):
                        col_idx = i % 4
                        with cols[col_idx]:
                            min_val = float(ml_data[feature].min())
                            max_val = float(ml_data[feature].max())
                            mean_val = float(ml_data[feature].mean())
                            step_val = (max_val - min_val) / 100 if max_val != min_val else 1.0
                            val = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val,
                                            step=step_val, key=f"lr_slider_{feature}")
                            user_inputs.append(val)

                    # Predicție
                    input_array = np.array([user_inputs])
                    input_scaled = scaler.transform(input_array)

                    try:
                        probs = lr_model.predict_proba(input_scaled)[0]
                        classes = lr_model.classes_

                        prob_df = pd.DataFrame({'Clasa': classes, 'Probabilitate': probs})
                        prob_df_sorted = prob_df.sort_values(by='Probabilitate', ascending=False)
                        top_df = prob_df_sorted.head(4).copy()
                        other_prob = prob_df_sorted.iloc[4:]['Probabilitate'].sum()

                        if other_prob > 0.001:
                            other_row = pd.DataFrame({'Clasa': ['Altele'], 'Probabilitate': [other_prob]})
                            pie_df = pd.concat([top_df, other_row], ignore_index=True)
                        else:
                            pie_df = top_df

                        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
                        ax_pie.pie(pie_df['Probabilitate'], labels=pie_df['Clasa'], autopct='%1.1f%%', startangle=90,
                                   colors=sns.color_palette('pastel')[0:len(pie_df)])
                        ax_pie.axis('equal')

                        col1, col2 = st.columns([1, 1.5])
                        with col1:
                            st.write("**Predicția Finală a Modelului:**")
                            st.markdown(f"<h3 style='color: #2e86c1; margin-top: 0px;'>{top_df.iloc[0]['Clasa']}</h3>",
                                        unsafe_allow_html=True)
                            st.write("Probabilități calculate (Top 4):")
                            st.dataframe(top_df.style.format({'Probabilitate': '{:.1%}'}))
                        with col2:
                            st.pyplot(fig_pie)
                    except Exception as e:
                        st.info("Predicțiile probabilistice nu sunt momentan disponibile pentru modelul curent.")

            else:
                st.warning(
                    "Nu există destule rânduri de date valide în set. Te rog debifează niște predictori care conțin valori lipsă.")
        else:
            st.info("Alege cel puțin o variabilă pentru antrenament.")

st.markdown("---")
st.info("Notă: Orice modificare pe setul de date făcută într-o secțiune devine vizibilă în toate.", icon="💡")