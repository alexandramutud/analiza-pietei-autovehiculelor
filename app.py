import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

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
section = st.sidebar.radio("Navigare secțiuni:", ["Introducere", "Setul de date", "Informații și Previzualizare", "Tratarea Valorilor Lipsă", "Encodare Variabile Categoriale", "Vizualizare și Analiză Grafică"])

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
elif section == "Tratarea Valorilor Lipsă":
    st.header("Tratarea Valorilor Lipsă")
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
            st.info("Curățarea radicală rămâne blocată până când apeși butonul de eliminare coloane fringe.")

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

    st.markdown("---")
    st.write("💡 *Notă: Orice curățare făcută anterior se reflectă aici.*")
