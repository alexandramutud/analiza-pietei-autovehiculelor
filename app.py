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
        Acest proiect este o aplicație web interactivă dezvoltată în **Python cu Streamlit**, 
        ce urmărește analiza pieței autovehiculelor, în anul 2023, pentru viitoare predicții,
        informări sau grafice. Este destinată atât persoanelor fizice, care urmează
        să achiziționeze un nou autovehicul, cât și firmelor de tip parc auto sau închirieri.
        
        ### Obiectivele Principale
        - **Curățarea Datelor**: Tratarea valorilor lipsă folosind metode statistice (Medie, Mediană, Mod).
        - **Vizualizare EDA**: Analiza distribuțiilor, corelațiilor și a valorilor extreme (outliers).
        - **Persistență**: Stocarea modificărilor pe durata sesiunii pentru un flux continuu.
        
        Navigați folosind meniul din stânga pentru a explora mai departe!
        """)

# ---------------------------
# Secțiunea: Setul de date
# ---------------------------
elif section == "Setul de date":
    st.header("Contextul Datelor")
    st.write(f"""
    Acest proiect utilizează un set de date complex filtrat pentru anul 2023.
    În prezent, setul tău de lucru conține **{df.shape[0]}** rânduri și **{df.shape[1]}** coloane.
    
    Elemente urmărite:
    - `Company`, `Model`, `Segment`, `Fuel`
    - Specificații engine (`Power(HP)`, `Torque(Nm)`, `Displacement`)
    - Dimensiuni (`Length`, `Width`, `Height`, `Wheelbase`)
    - Informații de performanță și eficiență.
    """)
    st.info("Poți continua în secțiunea următoare pentru filtrare și previzualizare detaliată.")
# Secțiunea: Informații și Previzualizare
# ---------------------------
elif section == "Informații și Previzualizare":
    st.header("Informații și Previzualizare Date")
    st.write(f"Setul curent de lucru conține **{df.shape[0]}** rânduri și **{df.shape[1]}** coloane.")
    
    st.subheader("Filtrare și Explorare")
    st.write("Folosește filtrele de mai jos pentru a rafina datele afișate în tabel:")
    
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
    st.write("Asigură-te că datele sunt curate înainte de a trece la vizualizări complexe.")

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
            st.session_state.cols_removed_40 = True # Putem debloca deoarece nu riscăm pierderea întregului set.

        st.markdown("#### Curățare radicală")
        if not st.session_state.cols_removed_40:
            st.error("🔒 **Acțiune blocată**: Mai întâi trebuie să elimini coloanele aproape goale (Prag > 40%) pentru a nu goli tot setul de date!")
        else:
            st.warning("⚠️ Această acțiune va șterge orice rând care mai are vreo celulă goală. Sigur dorești?")
            if st.button("🚀 ȘTERGE TOATE RÂNDURILE CU VALORI LIPSĂ", use_container_width=True):
                old_len = len(st.session_state.df)
                st.session_state.df = st.session_state.df.dropna()
                st.success(f"Finalizat: {old_len - len(st.session_state.df)} rânduri eliminate.")
                st.rerun()

        st.markdown("---")

        st.subheader("SAU... curățare manuală")

        # Imputare
        st.markdown("#### Imputare valori (completare selectivă)")
        num_cols_with_nan = df.columns[df.isnull().any()].tolist()
        if num_cols_with_nan:
            target_col = st.selectbox("Alege coloana pentru tratare:", num_cols_with_nan)
            metoda = st.radio("Metoda:", ["Mediană", "Medie", "Modulul", "Avansată (pe Marcă)"], horizontal=True)

            if st.button("✨ Aplică Imputarea", use_container_width=True):
                m_key = 'median' if metoda == "Mediană" else 'mean' if metoda == "Medie" else 'mode'
                if metoda == "Avansată (pe Marcă)":
                    st.session_state.df = treat_missing_values(st.session_state.df, numeric_col=target_col, group_col='Company')
                else:
                    st.session_state.df = treat_missing_values(st.session_state.df, numeric_col=target_col, method=m_key)
                st.rerun()

        # Eliminare rânduri specifice
        st.markdown("#### Eliminare rânduri specifice")
        row_col = st.selectbox("Șterge rândurile unde lipsește coloana:", ["Selectează"] + num_cols_with_nan, key="row_drop")
        if row_col != "Selectează":
            if st.button(f"🗑️ Șterge rândurile fără {row_col}", use_container_width=True):
                old_len = len(st.session_state.df)
                st.session_state.df = st.session_state.df.dropna(subset=[row_col])
                st.success(f"S-au eliminat {old_len - len(st.session_state.df)} rânduri.")
                st.rerun()

    else:
        st.success("✅ Felicitări! Nu mai există nicio valoare lipsă în setul de date.")

    st.markdown("---")
    st.subheader("Eliminare valori aberante (Outliers)")
    st.caption("Regulă folosită: prag extins Tukey, cu limite [Q1 - 3×IQR, Q3 + 3×IQR]. Rândurile care conțin cel puțin o valoare aberantă numerică sunt eliminate din setul curent.")

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
            st.success(f"S-au eliminat {removed_rows} rânduri care conțineau cel puțin o valoare aberantă (3×IQR).")
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
    st.write("Aducerea variabilelor la o scară comună este esențială pentru majoritatea algoritmilor de învățare automată.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ Nu s-au detectat coloane numerice pe care să le putem scala.")
    else:
        with st.expander("🎓 Învață despre scalare", expanded=True):
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
                "Scalat": [scaled_df[check_col].min(), scaled_df[check_col].max(), scaled_df[check_col].mean(), scaled_df[check_col].std()]
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
    st.write("Folosește această secțiune pentru a obține statistici centralizate pe segmente (Pivot Table).")

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
            group_by_col = st.selectbox("Grupare după (Categoria):", categorical_cols)
        with col2:
            agg_num_cols = st.multiselect("Coloane numerice pentru analiză:", numeric_cols, default=numeric_cols[:1])

        st.markdown("---")
        st.write("Alege funcțiile de agregare statistice:")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: f_mean = st.checkbox("Media", value=True)
        with c2: f_median = st.checkbox("Mediana")
        with c3: f_max = st.checkbox("Maximul")
        with c4: f_min = st.checkbox("Minimul")
        with c5: f_count = st.checkbox("Număr înregistrări", value=True)

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
                metric_label = {"mean": "Mediei", "median": "Mediană", "max": "Maximului", "min": "Minimului"}[metric_to_plot]
                
                st.markdown(f"### 📊 Reprezentare Vizuală a {metric_label}")
                first_col = f"{agg_num_cols[0]}_{metric_to_plot}"
                
                if first_col in pivot_table.columns:
                    plot_data = pivot_table[first_col].sort_values(ascending=False).head(15)
                    
                    fig, ax = plt.subplots(figsize=(16,10))
                    sns.barplot(x=plot_data.index, y=plot_data.values, palette="viridis", ax=ax)
                    plt.xticks(rotation=90, ha='center', fontweight='bold', fontsize=12)
                    plt.title(f"Top 15 - {metric_label} {agg_num_cols[0]} pe fiecare {group_by_col}", fontsize=16, fontweight='bold')
                    plt.ylabel(f"Valoare {metric_label} ({agg_num_cols[0]})", fontsize=13)
                    plt.xlabel(group_by_col, fontsize=13)
                    st.pyplot(fig)
                    st.caption(f"Grafic generat automat pentru metricul: {metric_label}.")

            # Descriere statistică detaliată
            with st.expander("🔍 Ce s-a întâmplat în spate? (Explicație Tehnică)", expanded=True):
                st.markdown(f"""
                Procesul pe care tocmai l-ai executat se numește în Pandas **Split-Apply-Combine** (Împarte-Aplică-Combină):

                1.  **Split (Împarțirea)**: Pandas a scanat coloana `{group_by_col}` și a creat grupuri virtuale. De exemplu, toate rândurile pentru 'BMW' au fost puse într-o listă separată de rândurile pentru 'Dacia'.
                2.  **Apply (Aplicarea)**: Pentru fiecare grup în parte, s-au calculat funcțiile matematice alese ({', '.join(agg_functions)}) pe coloanele `{', '.join(agg_num_cols)}`. Această operație ignoră valorile `NaN` pentru a nu altera rezultatul.
                3.  **Combine (Combinarea)**: Rezultatele de la fiecare grup au fost "lipite" la loc într-un singur tabel nou, unde indexul (rândurile) este acum categoria `{group_by_col}`.

                **Interpretare**: 
                - Dacă vezi o diferență mare între **Media** și **Mediana** unui grup, înseamnă că în acel grup ai *outliers* (mașini cu specificații care "trag" media în sus sau în jos în mod nefiresc).
                - Coloana `count` (Număr înregistrări) îți spune cât de reprezentativ este grupul. Dacă un brand are doar 1 mașină, media lui nu este relevantă statistic pentru întreg brandul.
                """)

# ---------------------------
# Secțiunea: Vizualizare și Analiză Grafică
# ---------------------------
elif section == "Vizualizare și Analiză Grafică":
    st.header("Vizualizare și Analiză Grafică")
    
    if df.empty:
        st.error("❌ Eroare: Setul de date este gol!")
        st.info("Folosește butonul **RESETEAZĂ TOATE DATELE** din stânga.")
    else:
        st.write("Explorează vizual proprietățile parcului auto 2023.")

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
                st.info("Ghid: Histograma arată frecvența, KDE arată densitatea probabilității.")

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

                    st.markdown("**Ce proces are loc:** Histogramă + KDE pentru evaluarea formei distribuției și a dispersiei.")
                    st.markdown(
                        f"""
**Ce indică outputul curent:**
- Media: **{mean_val:.2f}**, mediana: **{median_val:.2f}**.
- Dispersia (deviația standard): **{std_val:.2f}**.
- Forma distribuției este **{skew_text}** (skewness = **{skew_val:.2f}**).
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

                    st.markdown("**Ce proces are loc:** Boxplot-ul identifică variabilitatea centrală și valorile extreme pe baza regulii IQR.")
                    st.markdown(
                        f"""
**Ce indică outputul curent:**
- Interval intercuartilic (IQR): **{iqr:.2f}**.
- Prag inferior/superior outliers: **{lower_bound:.2f}** / **{upper_bound:.2f}**.
- Număr outliers detectați: **{len(outliers)}** din **{len(outlier_series)}** observații (**{outlier_pct:.2f}%**).
"""
                    )

        with tab3:
            st.subheader("Relații (Scatter Plot)")
            if len(numeric_cols) >= 2:
                c1, c2, c3 = st.columns(3)
                with c1: col_x = st.selectbox("X:", numeric_cols, index=0)
                with c2: col_y = st.selectbox("Y:", numeric_cols, index=1)
                with c3: col_color = st.selectbox("Culoare:", ["Fără"] + categorical_cols)
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
                    st.markdown("**Ce proces are loc:** Scatter plot-ul verifică dependența dintre două variabile și potențiale clustere.")
                    st.markdown(
                        f"""
**Ce indică outputul curent:**
- Corelația Pearson dintre **{col_x}** și **{col_y}** este **{corr_xy:.2f}**.
- Relația observată este **{relation_strength}** și **{trend}**.
- Număr puncte analizate: **{len(rel_df)}**.
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

                        st.markdown("**Ce proces are loc:** Se calculează corelațiile pereche pentru variabile numerice și se evidențiază intensitatea relațiilor.")
                        st.markdown(
                            f"""
**Ce indică outputul curent:**
- Cea mai puternică relație pozitivă: **{strongest_pos[0]} - {strongest_pos[1]}** (r = **{strongest_pos_val:.2f}**).
- Cea mai puternică relație negativă: **{strongest_neg[0]} - {strongest_neg[1]}** (r = **{strongest_neg_val:.2f}**).
- Aceste perechi sunt candidate bune pentru modelare predictivă sau reducere dimensională.
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

                    st.markdown("**Ce proces are loc:** Se compară frecvențele categoriilor pentru a identifica dominația și gradul de concentrare.")
                    st.markdown(
                        f"""
**Ce indică outputul curent:**
- Categoria dominantă este **{top_cat}** cu **{top_val}** observații (**{top_share:.2f}%**).
- Primele 3 categorii cumulează **{top3_share:.2f}%** din total.
- Dacă acest procent este ridicat, variabila este concentrată în puține categorii.
"""
                    )

        with tab6:
            st.subheader("Pair Plots")
            multi_cols = st.multiselect("Selectează (max 4):", numeric_cols, default=numeric_cols[:3])
            if len(multi_cols) > 1 and st.button("🚀 Generează Matrix"):
                pair_df = df[multi_cols].dropna()
                g = sns.pairplot(pair_df, diag_kind="kde", corner=True)
                st.pyplot(g.fig)

                pair_corr = pair_df.corr().where(~np.eye(len(multi_cols), dtype=bool)).stack()
                if not pair_corr.empty:
                    best_pair = pair_corr.abs().idxmax()
                    best_pair_val = pair_df[[best_pair[0], best_pair[1]]].corr().iloc[0, 1]
                    st.markdown("**Ce proces are loc:** Pair plot-ul oferă simultan distribuțiile univariate și relațiile bivariate dintre variabilele selectate.")
                    st.markdown(
                        f"""
**Ce indică outputul curent:**
- Cea mai puternică asociere din selecție este între **{best_pair[0]}** și **{best_pair[1]}**.
- Corelația perechii este **{best_pair_val:.2f}** ({describe_correlation(best_pair_val)}).
- Diagonala arată forma distribuțiilor, iar panourile din afara diagonalei arată tendințe și posibile clustere.
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
                    max_cats = st.slider("Număr categorii afișate:", min_value=3, max_value=12, value=8, key="violin_top")

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
                    iqr_by_cat = violin_df.groupby(violin_cat)[violin_num].quantile(0.75) - violin_df.groupby(violin_cat)[violin_num].quantile(0.25)
                    most_variable = iqr_by_cat.sort_values(ascending=False).index[0]

                    st.markdown("**Ce proces are loc:** Violin plot compară simultan forma distribuției, medianele și dispersia pentru mai multe categorii.")
                    st.markdown(
                        f"""
**Ce indică outputul curent:**
- Categoria cu mediana cea mai ridicată pentru **{violin_num}** este **{median_by_cat.index[0]}** ({median_by_cat.iloc[0]:.2f}).
- Categoria cu cea mai mare variabilitate (IQR) este **{most_variable}** ({iqr_by_cat.loc[most_variable]:.2f}).
- Graficul ajută la identificarea segmentelor unde valorile sunt concentrate sau foarte dispersate.
"""
                    )
            else:
                st.warning("Pentru acest tip de grafic este nevoie de cel puțin o coloană numerică și una categorială.")

# ---------------------------
# Secțiunea: Analiză Statistică (Regresie Multiplă)
# ---------------------------
elif section == "Analiză Statistică (Regresie)":
    st.header("Analiză Statistică: Regresie Liniară Multiplă")
    st.write("În această secțiune folosim modelarea statistică pentru a înțelege cum parametrii tehnici influențează consumul de combustibil.")

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
            target_var = st.selectbox("Variabila dependentă (Target - Y):", numeric_cols, index=numeric_cols.index(default_target) if default_target in numeric_cols else 0)
        with col2:
            potential_features = [c for c in numeric_cols if c != target_var]
            selected_features = st.multiselect("Variabile independente (Predictori - X):", potential_features, default=potential_features[:4] if len(potential_features) >= 4 else potential_features)

        if selected_features:
            # Drop NAs for specifically selected columns
            reg_data = df[selected_features + [target_var]].dropna()
            
            if reg_data.empty:
                st.error("❌ Setul de date rezultat este gol după eliminarea valorilor lipsă. Verifică datele!")
            else:
                st.write(f"Modelul va fi antrenat pe **{len(reg_data)}** observații (după eliminarea rândurilor cu valori lipsă).")
                
                # --- Antrenare Model ---
                X = reg_data[selected_features]
                y = reg_data[target_var]
                X = sm.add_constant(X) # Adăugăm interceptul (constanta)
                
                model = sm.OLS(y, X).fit()
                
                # --- Rezultate ---
                st.subheader("2. Rezultatul Modelului (OLS Summary)")
                st.text(str(model.summary()))
                
                # --- Interpretare ---
                with st.expander("📝 Interpretarea Rezultatelor (Ghid Educațional)", expanded=True):
                    st.markdown(f"""
### Cum citim cifrele de mai sus?

1.  **R-squared (Coeficientul de Determinare):** **{model.rsquared:.4f}**
    *   *Semnificație:* Modelul explică aproximativ **{model.rsquared*100:.1f}%** din variația variabilei **{target_var}**. Cu cât e mai aproape de 1, cu atât modelul e mai precis.
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
                        input_data[feature] = st.number_input(f"{feature}:", min_value=min_v, max_value=max_v, value=mean_v, key=f"pred_{feature}")
                
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
    st.write("Acest algoritm ne supervizat grupează mașinile cu profiluri tehnice similare fără a cunoaște etichetele inițiale (cum ar fi tipul caroseriei).")
    
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
                        st.info("💡 **SFAT:** Punctul de „cot” (unde curba devine mai plată) reprezintă de obicei numărul ideal de clustere.")
                
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
                        label=f'Cluster {i+1}', 
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
                cluster_data['Cluster'] = [f"Cluster {i+1}" for i in y_kmeans]
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
    st.write("Modelul grupează mașinile folosind doar două variabile numerice (fără etichete). Mai jos ai un flux orientat pe interpretare: separare, profiluri de cluster și comparație cu o etichetă reală.")

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
                    st.info(f"Filtru IQR activ: {removed_outliers} observații eliminate, {len(hc_data)} observații rămase pentru model.")
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
                    st.caption("Salturile mari pe axa verticală sugerează tăieri naturale ale arborelui (valori candidate pentru K).")

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

                    size_df = hc_data["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Număr")
                    size_df["Proporție (%)"] = (size_df["Număr"] / len(hc_data) * 100).round(1)
                    st.markdown("**Dimensiunea clusterelor**")
                    st.dataframe(size_df, use_container_width=True)

                    profile_df = hc_data.groupby("Cluster")[[var_1, var_2]].agg(["mean", "median"]).round(2)
                    st.markdown(f"**Profil numeric pe clustere ({var_1}, {var_2})**")
                    st.dataframe(profile_df, use_container_width=True)

                    if cat_label != "(Fără comparație)":
                        if ari_val is not None:
                            st.info(f"💡 **OBSERVAȚIE - Scorul ARI: {ari_val:.3f}**\n\nAcest scor matematic (*Adjusted Rand Index*) arată cât de exact a intuit algoritmul etichetele reale din fabrică.\n* O valoare aproape de **1.0** înseamnă suprapunere perfectă (gruparea geometrică e identică cu cea din catalogul auto).\n* O valoare aproape de **0.0** indică o suprapunere slabă (categoriile puse de producător nu prea au legătură matematică cu selecția ta).")

# ---------------------------
# Secțiunea: Clasificare Predictivă (ML)
# ---------------------------
elif section == "Clasificare Predictivă (ML)":
    st.header("Clasificare Predictivă (ML Supervizat)")
    st.write("Acest modul antrenează algoritmi care învață să prezică o categorie (cum ar fi Segmentul sau Body Style) folosind fix atributele tehnice pe care i le oferi spre analiză.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 1 or len(categorical_cols) < 1:
        st.warning("E nevoie de coloane numerice și categoriale pentru funcționare.")
    else:
        st.subheader("Configurare Set de Date")
        target_col = st.selectbox("Alege ce vrei să prezică algoritmul (Target Y):", categorical_cols, index=categorical_cols.index('Segment') if 'Segment' in categorical_cols else 0)
        
        default_features = ['Top Speed', 'Power(HP)', 'Width'] if 'Top Speed' in numeric_cols and 'Power(HP)' in numeric_cols and 'Width' in numeric_cols else numeric_cols[:2]
        feature_cols = st.multiselect("Alege caracteristicile tehnice din care să învețe algoritmul (Features X):", numeric_cols, default=default_features)
        
        if len(feature_cols) >= 1:
            # Curățăm doar rândurile lipsă de pe coloanele vizate folosind o funcție rapidă
            ml_data = df[feature_cols + [target_col]].dropna()
            
            if len(ml_data) > 50:
                X = ml_data[feature_cols].values
                y = ml_data[target_col].values
                
                # Split pentru validare cruzată - 20% test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                tab1, tab2 = st.tabs(["🌳 1. Arbori de Decizie (Interpretare)", "🌲🌲 2. Random Forest (Performanță)"])
                
                with tab1:
                    st.markdown("### Arbore de Decizie (Analiza Logicii Algoritmice)")
                    st.write("Acest model este extrem de transparent pentru mintea umană, deoarece ia decizii bazate pe ramuri logice secvențiale (Ex: Dacă e lat -> Dacă are cai putere -> Deduc că e SUV).")
                    
                    depth = st.slider("Alege Adâncimea Maximă a Arborelui Desenat:", 2, 10, 7) # Setați la 7 după cerința utilizator
                    
                    dt_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
                    dt_model.fit(X_train, y_train)
                    
                    y_pred_dt = dt_model.predict(X_test)
                    acc_dt = accuracy_score(y_test, y_pred_dt)
                    
                    st.success(f"**Acuratețe de testare (Precizie strict pe date nevăzute)**: {acc_dt:.2%}")
                    
                    st.markdown("#### Schema Decizională a Modelului")
                    fig_tree, ax_tree = plt.subplots(figsize=(24, 12))
                    try:
                        classes_str = [str(c) for c in dt_model.classes_]
                        plot_tree(dt_model, feature_names=feature_cols, class_names=classes_str, filled=True, rounded=True, ax=ax_tree, fontsize=7)
                        st.pyplot(fig_tree)
                        st.caption("Fiecare 'cutie' arată condiția matematică folosită. Cutiile intens colorate jos înseamnă o predicție sigură și unică.")
                    except Exception as e:
                        st.error(f"Eroare la desenarea structurii (posibil format necorespunzător al stringurilor): {str(e)}")
                        
                with tab2:
                    st.markdown("### Random Forest (Top Acuratețe)")
                    st.write("În loc să deseneze 1 arbore, acesta antrenează în orb 100 de abori diferiți care 'votează' cel mai bun rezultat.")
                    
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    
                    y_pred_rf = rf_model.predict(X_test)
                    acc_rf = accuracy_score(y_test, y_pred_rf)
                    
                    st.success(f"**Acuratețe de testare Random Forest**: {acc_rf:.2%}")
                    
                    st.markdown("#### Care Factor Contează Cel Mai Mult?")
                    importances = rf_model.feature_importances_
                    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
                    
                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=imp_df, x='Importance', y='Feature', palette='plasma', ax=ax_imp)
                    ax_imp.set_title(f"Ponderea factorilor în predicția `{target_col}`")
                    st.pyplot(fig_imp)
                    st.caption(f"Graficul arată care din atribute (Lățime vs Putere etc.) afectează cel mai mult formațional clasa '{target_col}' în industria auto.")

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
                        sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', xticklabels=labels_for_cm, yticklabels=labels_for_cm, ax=ax_cm)
                        ax_cm.set_ylabel("Adevăr (clasa reală din date)")
                        ax_cm.set_xlabel("Predicție (clasa estimată de model)")
                        st.pyplot(fig_cm)
                        st.caption("Interpretare: pe diagonală sunt clasificările corecte (...); în afara diagonalei sunt confuziile modelului (...).")
                    except Exception as e:
                        st.warning("Prea multe categorii pentru a desena eficient heatmap-ul.")
                            
            else:
                st.warning("Nu există destule rânduri de date valide în set. Te rog debifează niște predictori care conțin valori lipsă.")
        else:
            st.info("Alege cel puțin o variabilă pentru antrenament.")

st.markdown("---")
st.write("📊 *Notă: Orice curățare făcută anterior se reflectă aici.*")
