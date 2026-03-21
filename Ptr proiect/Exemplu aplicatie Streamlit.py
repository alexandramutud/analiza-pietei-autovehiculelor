import streamlit as st
import pandas as pd
import numpy as np


st.title("Streamlit & Pandas")
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

st.markdown('<h1 class="custom-title">Seminar 2</h1>', unsafe_allow_html=True)

# Bara laterală pentru navigare între secțiuni
section = st.sidebar.radio("Navigați la:",
                           ["Streamlit", "Ce este Pandas", "Serii", "DataFrame-uri",
                            "Selecție și Filtrare",
                            "Exerciții Interactive",
                            "Quiz", "Interacțiuni cu coloanele dataFrame"])

# ---------------------------
# Secțiunea: Noțiuni de bază Streamlit
# ---------------------------
if section == "Streamlit":
    st.header("Noțiuni de bază")
    st.markdown(r"""
        ### Ce este Streamlit?

        **Streamlit** este o bibliotecă Python open-source care permite transformarea scripturilor Python în aplicații web interactive, fără a fi necesare cunoștințe avansate de dezvoltare web (HTML, CSS sau JavaScript). Este ideal pentru:

        - **Dashboard-uri de date:** Explorarea și vizualizarea seturilor de date prin grafice interactive.
        - **Prototipare rapidă:** Transformarea rapidă a scripturilor și a modelelor de Machine Learning în aplicații demonstrative.
        - **Raportare dinamică:** Rapoarte interactive, care se actualizează în timp real, permițând filtrarea și interacțiunea cu datele.

        ### Cum se folosește?

        În esență, Streamlit funcționează prin importul modulului ca un alias (de obicei `st`), și prin utilizarea funcțiilor sale pentru a construi interfața aplicației:

        - **Elemente de afișare:**  
          - `st.title("Titlu")` – setează titlul aplicației.  
          - `st.write("Text sau Date")` – afișează text, date sau grafice, detectând automat tipul conținutului.  
          - `st.markdown("Text formatat cu Markdown")` – permite afișarea textului cu formatare Markdown.

        - **Widget-uri interactive:**  
          - `st.button("Apasă-mă!")` – creează un buton.  
          - `st.text_input("Introdu textul:")` – oferă o casetă de text pentru input.  
          - `st.slider("Selectează un număr", 0, 100)` – creează un slider pentru selecția numerică.

        - **Layout și organizare:**  
          - `st.sidebar` – permite adăugarea de elemente pe o bară laterală, utilă pentru meniuri și filtre.  
          - `st.columns()` – organizează elementele în coloane pentru o prezentare structurată.

        - **Funcționalități avansate:**  
          - **Caching:** `@st.experimental_memo` și `@st.experimental_singleton` pentru optimizarea performanței prin memorarea rezultatelor funcțiilor costisitoare.  
          - **Gestionarea stării:** `st.session_state` pentru a păstra datele între reîncărcări.

        ### Atentie!

        - **Reîncărcarea scriptului:**  
          Fiecare interacțiune (ex. apăsarea unui buton) reexecută întregul script, ceea ce poate duce la probleme dacă nu este gestionată corect logica sau starea aplicației. Utilizarea `st.session_state` este esențială pentru a păstra datele între reîncărcări.

        - **Persistența datelor:**  
          Deoarece scriptul este reîncărcat constant, este important să salvezi datele critice (ex. DataFrame-uri, setări) în `st.session_state` sau să le memorezi prin caching pentru a evita recalculările inutile.

        - **Personalizarea temei:**  
          Streamlit permite personalizarea aspectului aplicației prin setări în fișierul de configurare (`config.toml`) sau prin injectarea de CSS custom. Totuși, personalizarea excesivă poate complica menținerea codului și actualizările.

        - **Limitări în personalizare:**  
          Deși Streamlit este excelent pentru prototipuri și dashboard-uri, nu este la fel de flexibil ca framework-urile web tradiționale (de ex. Django sau Flask) atunci când vine vorba de controlul detaliat al interfeței sau al fluxului de date.

        - **Dependența de versiune:**  
          Unele funcționalități, precum `st.experimental_rerun`, pot să nu fie disponibile în versiunile mai vechi. Este recomandat să folosești o versiune actualizată pentru a beneficia de toate funcționalitățile.

        ### Avantaje

        - **Rapiditate și Simplitate:**  
          Poți transforma rapid un script Python într-o aplicație web interactivă fără a învăța tehnologii web suplimentare.

        - **Interactivitate:**  
          Widget-urile integrate permit crearea de aplicații interactive pentru analiză de date, prototipare de modele ML sau prezentări dinamice.

        - **Actualizări în timp real:**  
          Orice modificare a codului se reflectă imediat în interfață, facilitând un ciclu rapid de dezvoltare și feedback.

        ### Dezavantaje

        - **Reîncărcarea întregului script:**  
          Fiecare interacțiune reexecută întregul cod, ceea ce poate duce la probleme de performanță sau la necesitatea unei gestionări riguroase a stării.

        - **Limitări în personalizare:**  
          Deși oferă opțiuni pentru personalizare, Streamlit poate fi restrictiv dacă ai nevoie de un control foarte detaliat al interfeței sau al fluxului de date.

        - **Scalabilitate:**  
          Pentru aplicații de producție sau cu cerințe foarte complexe, soluțiile tradiționale de dezvoltare web pot oferi mai multă flexibilitate și scalabilitate.

        ### Când să folosești Streamlit?

        - **Prototipare rapidă:**  
          Pentru a testa idei și pentru a demonstra concepte în domeniul analizei de date și al Machine Learning.

        - **Dashboard-uri și vizualizări de date:**  
          Pentru a crea aplicații interactive de vizualizare a datelor, unde utilizatorii pot explora și filtra informațiile.

        - **Proiecte educaționale:**  
          Pentru tutoriale, prezentări și workshop-uri, unde simplitatea codului ajută la înțelegerea conceptelor.

        **Rularea Aplicației:**
        - Salvează scriptul ca `app.py`.
        - Rulează aplicația folosind comanda:

          ```
          streamlit run app.py
          ```
        """, unsafe_allow_html=True)
    st.title("Exemplu Streamlit")
    st.header("Demo")
    st.markdown("""
    Acest exemplu integrează:
    - Elementele de afișare (text, markdown, cod, LaTeX)
    - Widget-uri de input (buton, text input, slider, selectbox, etc.)
    - Afișarea datelor (dataframe, tabel, JSON, metric)
    - Componente de layout (sidebar, container, coloane, expander, tabs)
    
    """)

    # --- Secțiunea 1: Afișare și text ---
    st.subheader("1. Elemente de Afișare")
    st.write("Acesta este un exemplu de text afișat cu st.write()")
    st.markdown("**Text bold în Markdown**")
    st.code("print('Hello, world!')", language="python")
    st.latex(r"\int_{a}^{b} f(x)\,dx")

    # --- Secțiunea 2: Widget-uri de Input ---
    st.subheader("2. Widget-uri de Input")

    # Buton
    if st.button("Apasă-mă!"):
        st.write("Butonul a fost apăsat!")

    # Text input
    user_text = st.text_input("Introdu textul tău:")
    st.write("Ai introdus:", user_text)

    # Text area
    user_text_area = st.text_area("Introdu un text mai lung:")
    st.write("Textul tău:", user_text_area)

    # Number input
    user_number = st.number_input("Introdu un număr:", value=0)
    st.write("Numărul introdus:", user_number)

    # Slider
    user_slider = st.slider("Selectează un număr între 0 și 100", 0, 100, 50)
    st.write("Valoarea slider-ului:", user_slider)

    # Selectbox
    user_select = st.selectbox("Selectează o opțiune:", ["Opțiunea 1", "Opțiunea 2", "Opțiunea 3"])
    st.write("Ai selectat:", user_select)

    # Multiselect
    user_multiselect = st.multiselect("Selectează mai multe opțiuni:", ["A", "B", "C", "D"])
    st.write("Ai selectat:", user_multiselect)

    # Radio button
    user_radio = st.radio("Selectează o opțiune:", ["Radio 1", "Radio 2", "Radio 3"])
    st.write("Opțiunea selectată:", user_radio)

    # Checkbox
    user_checkbox = st.checkbox("Bifează această opțiune")
    st.write("Checkbox este:", user_checkbox)

    # Date input
    user_date = st.date_input("Selectează o dată")
    st.write("Data aleasă:", user_date)

    # Time input
    user_time = st.time_input("Selectează o oră")
    st.write("Ora aleasă:", user_time)

    # File uploader
    uploaded_file = st.file_uploader("Încarcă un fișier", type=["csv", "txt", "json"])
    if uploaded_file is not None:
        st.write("Fișier încărcat:", uploaded_file.name)

    # --- Secțiunea 3: Afișarea Datelor ---
    st.subheader("3. Afișarea Datelor")
    # DataFrame și tabel
    df = pd.DataFrame(np.random.randn(10, 3), columns=["Coloana 1", "Coloana 2", "Coloana 3"])
    st.write("DataFrame cu st.dataframe():")
    st.dataframe(df)
    st.write("Tabel cu st.table():")
    st.table(df)

    # JSON display
    data_json = {"cheie1": "valoare1", "cheie2": 123, "cheie3": [1, 2, 3]}
    st.json(data_json)

    # Metric (pentru dashboard-uri)
    st.metric(label="Viteza", value="120 km/h", delta="5 km/h")

    # Grafic: Linie
    st.line_chart(df)

    # --- Secțiunea 4: Layout și Organizare ---
    st.subheader("4. Componente de Layout")

    # Sidebar
    st.sidebar.header("Bară laterală")
    st.sidebar.write("Acesta este un widget plasat în sidebar.")
    sidebar_select = st.sidebar.selectbox("Alege o opțiune:", ["X", "Y", "Z"])
    st.sidebar.write("Ai selectat:", sidebar_select)

    # Container
    with st.container():
        st.write("Acesta este un container care grupează elemente.")

    # Coloane
    col1, col2 = st.columns(2)
    with col1:
        st.write("Continutul din Coloana 1")
    with col2:
        st.write("Continutul din Coloana 2")

    # Expander
    with st.expander("Click pentru a vedea mai multe"):
        st.write("Acesta este conținutul dintr-un expander.")

    # Tabs
    tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
    with tab1:
        st.write("Conținutul din Tab 1")
    with tab2:
        st.write("Conținutul din Tab 2")

    # Placeholder (widget dinamic)
    placeholder = st.empty()
    placeholder.write("Acesta poate fi actualizat ulterior...")


    # --- Secțiunea 5: Managementul stării (Session State) ---
    st.subheader("5. Managementul Stării (Session State)")
    # Inițializăm o variabilă în session_state dacă nu există deja
    if 'numar_clickuri' not in st.session_state:
        st.session_state.numar_clickuri = 0


    def creste_clickuri():
        st.session_state.numar_clickuri += 1


    st.button("Click!", on_click=creste_clickuri)
    st.write("Număr de click-uri:", st.session_state.numar_clickuri)
    st.write("Utilizați bara laterală pentru a explora celelalte secțiuni.")

# ---------------------------
# Secțiunea: Introducere în Pandas
# ---------------------------
elif section == "Ce este Pandas":
    st.header("Introducere în Pandas")
    st.write("""
    Pandas este o bibliotecă puternică pentru manipularea și analiza datelor.
    Aceasta oferă două structuri principale de date:
    - **Series (Serii):** Un obiect unidimensional asemănător unui array.
    - **DataFrame:** Un tabel bidimensional de date.

    
    """)

# ---------------------------
# Secțiunea: Serii
# ---------------------------
elif section == "Serii":
    st.header("Pandas Serii")
    st.write("""
    O **Serie** este un obiect unidimensional asemănător unui array care conține date și indexul asociat.
    Dacă nu specificați un index, Pandas atribuie automat un index numeric.
    """)

    st.subheader("Crearea unei Serii dintr-o lista")
    data = [10, 20, 30, 40]
    series_default = pd.Series(data)
    st.code("""
import pandas as pd
data = [10, 20, 30, 40]
series_default = pd.Series(data)
print(series_default)
    """, language="python")
    st.write("Serie cu indexul implicit:")
    st.write(series_default)

    st.subheader("Crearea unei Serii cu Index Personalizat")
    custom_index = ['a', 'b', 'c', 'd']
    series_custom = pd.Series(data, index=custom_index)
    st.code("""
custom_index = ['a', 'b', 'c', 'd']
series_custom = pd.Series(data, index=custom_index)
print(series_custom)
    """, language="python")
    st.write("Serie cu index personalizat:")
    st.write(series_custom)

    st.subheader("Crearea unei Serii dintr-un Dicționar")
    data_dict = {'California': 10, 'Texas': 20, 'New York': 30}
    series_from_dict = pd.Series(data_dict)
    st.code("""
data_dict = {'California': 10, 'Texas': 20, 'New York': 30}
series_from_dict = pd.Series(data_dict)
print(series_from_dict)
    """, language="python")
    st.write("Serie dintr-un dicționar (valorile lipsă apar ca NaN):")
    st.write(series_from_dict)

# ---------------------------
# Secțiunea: DataFrame-uri
# ---------------------------
elif section == "DataFrame-uri":
    st.header("Pandas: DataFrame-uri")
    st.write("""
    Un **DataFrame** este o structură de date bidimensională, asemănătoare unui tabel, cu rânduri și coloane.
    Este similar unui tabel de calcul și poate fi creat din liste, dicționare sau alte structuri de date.
    """)

    st.subheader("Crearea unui DataFrame dintr-o Listă")
    data_list = ['A', 'B', 'C']
    df_from_list = pd.DataFrame(data_list, columns=['Literă'])
    st.code("""
data_list = ['A', 'B', 'C']
df_from_list = pd.DataFrame(data_list, columns=['Literă'])
print(df_from_list)
    """, language="python")
    st.write("DataFrame creat dintr-o listă:")
    st.write(df_from_list)

    st.subheader("Crearea unui DataFrame dintr-un Dicționar de Liste")
    data_dict = {
        'Nume': ['Alice', 'Bob', 'Charlie'],
        'Vârstă': [25, 30, 35],
        'Oraș': ['New York', 'Los Angeles', 'Chicago']
    }
    df_from_dict = pd.DataFrame(data_dict)
    st.code("""
data_dict = {
    'Nume': ['Alice', 'Bob', 'Charlie'],
    'Vârstă': [25, 30, 35],
    'Oraș': ['New York', 'Los Angeles', 'Chicago']
}
df_from_dict = pd.DataFrame(data_dict)
print(df_from_dict)
    """, language="python")
    st.write("DataFrame creat dintr-un dicționar:")
    st.write(df_from_dict)

    st.subheader("Crearea unui DataFrame din Structuri Existente")
    series1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    series2 = pd.Series([4, 5, 6], index=['a', 'b', 'c'])
    df_from_series = pd.DataFrame({'col1': series1, 'col2': series2})
    st.code("""
series1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
series2 = pd.Series([4, 5, 6], index=['a', 'b', 'c'])
df_from_series = pd.DataFrame({'col1': series1, 'col2': series2})
print(df_from_series)
    """, language="python")
    st.write("DataFrame creat din serii:")
    st.write(df_from_series)

# ---------------------------
# Secțiunea: Selecție și Filtrare
# ---------------------------
elif section == "Selecție și Filtrare":
    st.header("Selecția și filtrarea datelor")
    st.write("""
    Pandas oferă mai multe metode pentru a selecta și filtra datele.

    **Selecția Coloanelor:**
    - Notare cu punct: `df.nume_coloană`
    - Paranteze pătrate: `df['nume_coloană']`
    - Indexare numerică: `df.iloc[:, index_coloană]`

    **Selecția Rândurilor:**
    - Slicing: `df[start:end]`
    - Indexare după etichetă cu `.loc[]`
    - Indexare după poziție cu `.iloc[]`
    """)

    # DataFrame exemplu
    df_sample = pd.DataFrame({
        'Produs': ['Măr', 'Banana', 'Morcov', 'Capsuni'],
        'Preț': [1.2, 0.5, 0.8, 1.5],
        'Cantitate': [100, 150, 80, 200]
    })
    st.subheader("Exemplu")
    st.write(df_sample)

    st.subheader("Selecția coloanelor și rândurilor")
    st.write("Selectați coloana 'Preț' folosind notarea cu punct:")
    st.code("df_sample.Pret", language="python")
    st.write(df_sample.Pret if hasattr(df_sample, 'Pret') else df_sample['Preț'])

    st.write("Selectați coloana 'Cantitate' folosind parantezele pătrate:")
    st.code("df_sample['Cantitate']", language="python")
    st.write(df_sample['Cantitate'])

    st.write("Selectați primele două rânduri folosind slicing:")
    st.code("df_sample[:2]", language="python")
    st.write(df_sample[:2])

    st.write("Selectați rândurile după etichetă cu `.loc[]` (rândurile cu indexul 1 și 2):")
    st.code("df_sample.loc[1:2]", language="python")
    st.write(df_sample.loc[1:2])

    st.write("Selectați rândurile după poziție cu `.iloc[]` (primul și al treilea rând):")
    st.code("df_sample.iloc[[0, 2]]", language="python")
    st.write(df_sample.iloc[[0, 2]])

    st.subheader("Filtrarea rândurilor după condiții")
    st.write("Filtrați rândurile unde 'Cantitate' este mai mare sau egală cu 100:")
    st.code("df_sample[df_sample['Cantitate'] >= 100]", language="python")
    st.write(df_sample[df_sample['Cantitate'] >= 100])

    st.subheader("Actualizări cu loc")
    st.write("""
    Actualizați coloana `Preț` aplicând o reducere pentru produsele care costă mai mult de 1.0.
    Se creează o nouă coloană 'Preț Redus'.
    """)
    df_sample['Preț Redus'] = df_sample['Preț']
    discount_condition = df_sample['Preț'] > 1.0
    df_sample.loc[discount_condition, 'Preț Redus'] = df_sample.loc[discount_condition, 'Preț'] * 0.9
    st.code("""
discount_condition = df_sample['Preț'] > 1.0
df_sample.loc[discount_condition, 'Preț Redus'] = df_sample.loc[discount_condition, 'Preț'] * 0.9
    """, language="python")
    st.write("DataFrame după actualizarea reducerii:")
    st.write(df_sample)

    st.markdown("---")
    st.header("Exemple cu loc și iloc")

    st.subheader("Exemple cu loc")
    st.write("1. **Filtrare și Selectarea Coloanelor Specifice:**")
    filtered_df = df_sample.loc[df_sample['Preț'] < 1.0, ['Produs', 'Cantitate']]
    st.code("""
filtered_df = df_sample.loc[df_sample['Preț'] < 1.0, ['Produs', 'Cantitate']]
print(filtered_df)
    """, language="python")
    st.write("Rânduri unde preț este mai mic de 1.0 (afișând Produs și Cantitate):")
    st.write(filtered_df)

    st.write("2. **Actualizare amai multor coloane pe bază de condiție:**")
    df_update = df_sample.copy()
    promo_condition = df_update['Cantitate'] < 100
    df_update.loc[promo_condition, ['Preț']] = 0.99
    df_update.loc[promo_condition, 'Promo'] = True
    st.code("""
promo_condition = df_update['Cantitate'] < 100
df_update.loc[promo_condition, ['Preț']] = 0.99
df_update.loc[promo_condition, 'Promo'] = True
    """, language="python")
    st.write("DataFrame după aplicarea reducerii promo (pentru rândurile cu Cantitate < 100):")
    st.write(df_update)

    st.subheader("Exemple  cu iloc")
    st.write("1. **Selectarea unui bloc de rânduri și coloane:**")
    block_df = df_sample.iloc[1:3, 0:2]  # Selectează rândurile 1 până la 2 și coloanele 0 până la 1.
    st.code("""
block_df = df_sample.iloc[1:3, 0:2]
print(block_df)
    """, language="python")
    st.write("Bloc de date (rândurile 1-2, coloanele 0-1):")
    st.write(block_df)

    st.write("2. **Selectarea rândurilor și coloanelor Non-Consecutive:**")
    selected_df = df_sample.iloc[[0, 3], [0, 2]]  # Primul și al patrulea rând, prima și a treia coloană.
    st.code("""
selected_df = df_sample.iloc[[0, 3], [0, 2]]
print(selected_df)
    """, language="python")
    st.write("Rânduri și coloane non-consecutive:")
    st.write(selected_df)
elif section=="Quiz":
    st.title("Quiz")


    # Definim un DataFrame cu întrebări, opțiuni și răspunsuri
    data = {
        "Question": [
            "1. Avem un DataFrame `df` cu indexul [1, 2, 3, 4, 5]. Ce returnează `df.loc[3]` comparativ cu `df.iloc[3]`?",
            "2. Pentru un DataFrame cu index custom, de exemplu, index = ['a', 'b', 'c'], ce returnează `df.loc['b']` comparativ cu `df.iloc[1]`?",
            "3. Dacă avem un DataFrame cu index non-integer, ex: index = ['apple', 'banana', 'cherry'], ce se întâmplă când se folosește `df.iloc['banana']`?",
            "4. Avem un DataFrame cu un index numeric care nu este ordonat crescător, de exemplu, index = [10, 3, 7, 2]. Ce returnează `df.loc[7]` și `df.iloc[2]`?",
            "5. Avem un DataFrame cu index duplicat, de exemplu, index = [1, 2, 2, 3]. Ce se întâmplă când se utilizează `df.loc[2]` comparativ cu `df.iloc[1]`?"
        ],
        "Option1": [
            "df.loc[3] returnează rândul cu eticheta 3, iar df.iloc[3] returnează rândul de pe poziția 3 (al patrulea rând).",
            "Ambele returnează același rând, deoarece 'b' este a doua etichetă.",
            "df.iloc['banana'] returnează rândul cu eticheta 'banana'.",
            "df.loc[7] returnează rândul cu eticheta 7, iar df.iloc[2] returnează al treilea rând, care este același cu cel cu eticheta 7.",
            "df.loc[2] returnează toate rândurile cu eticheta 2, iar df.iloc[1] returnează primul rând al DataFrame-ului."
        ],
        "Option2": [
            "df.loc[3] returnează rândul cu eticheta 3, iar df.iloc[3] returnează rândul de pe poziția 3 (al patrulea rând).",
            "df.loc['b'] returnează rândul cu eticheta 'b', dar df.iloc[1] returnează eroare.",
            "df.iloc['banana'] generează o eroare deoarece .iloc necesită un index integer.",
            "df.loc[7] returnează rândul cu eticheta 7, iar df.iloc[2] returnează al treilea rând, care poate fi diferit din cauza ordinii indexului.",
            "df.loc[2] returnează rândul cu eticheta 2, iar df.iloc[1] returnează al doilea rând, fără a ține cont de duplicate."
        ],
        "Option3": [
            "df.loc[3] returnează rândul cu eticheta 3, iar df.iloc[3] returnează primul rând din DataFrame.",
            "Ambele returnează rânduri diferite, deoarece .loc folosește etichete, iar .iloc folosește poziții.",
            "df.iloc['banana'] returnează o eroare deoarece .iloc necesită un index integer.",
            "df.loc[7] returnează eroare deoarece nu există eticheta 7, iar df.iloc[2] returnează al treilea rând.",
            "df.loc[2] returnează toate rândurile cu eticheta 2, iar df.iloc[1] returnează al doilea rând din DataFrame."
        ],
        "Option4": [
            "df.loc[3] returnează rândul cu eticheta 3, iar df.iloc[3] returnează rândul de pe poziția 3 (al patrulea rând).",
            "Ambele returnează același rând, deoarece indexul este implicit numeric.",
            "df.iloc['banana'] returnează rândul cu eticheta 'banana'.",
            "df.loc[7] returnează rândul cu eticheta 7, iar df.iloc[2] returnează eroare.",
            "df.loc[2] returnează doar primul rând cu eticheta 2, iar df.iloc[1] returnează eroare din cauza duplicatelor."
        ],
        "Answer": [
            "df.loc[3] returnează rândul cu eticheta 3, iar df.iloc[3] returnează rândul de pe poziția 3 (al patrulea rând).",
            "Ambele returnează același rând, deoarece 'b' este a doua etichetă.",
            "df.iloc['banana'] generează o eroare deoarece .iloc necesită un index integer.",
            "df.loc[7] returnează rândul cu eticheta 7, iar df.iloc[2] returnează al treilea rând, care poate fi diferit din cauza ordinii indexului.",
            "df.loc[2] returnează toate rândurile cu eticheta 2, iar df.iloc[1] returnează al doilea rând din DataFrame."
        ]
    }

    df_quiz = pd.DataFrame(data)

    st.write("Răspunde la următoarele întrebări despre `.loc` și `.iloc`:")

    # Dicționar pentru stocarea răspunsurilor utilizatorului
    user_answers = {}

    # Afișăm fiecare întrebare cu widget-ul radio pentru selecția răspunsului
    for index, row in df_quiz.iterrows():
        st.markdown(f"**Întrebarea {index + 1}:** {row['Question']}")
        options = [row['Option1'], row['Option2'], row['Option3'], row['Option4']]
        user_answer = st.radio("Selectează răspunsul:", options, key=f"quiz_{index}")
        user_answers[index] = user_answer

    # Buton pentru verificarea răspunsurilor
    if st.button("Verifică răspunsurile"):
        score = 0
        for index, row in df_quiz.iterrows():
            if user_answers.get(index) == row['Answer']:
                score += 1
        st.write(f"Scorul tău: {score} din {len(df_quiz)}")
# ---------------------------
# Secțiunea: Exerciții Interactive
# ---------------------------
elif section == "Exerciții Interactive":
    st.header("Exerciții Interactive")
    st.write("""
    

    1. **Creați un DataFrame de Produse:**  
       Introduceți numele produselor, prețurile și cantitățile pentru a construi un DataFrame.

    2. **Filtrare și Actualizare:**  
       Folosiți DataFrame-ul de Produse pentru a actualiza starea stocului (marcați ca "Stoc Scăzut" dacă este sub un anumit prag).
    """)

    # Inițializați starea de sesiune pentru df_products dacă nu există deja
    # Streamlit reîncarcă întregul script la fiecare interacțiune,
    # iar starea de sesiune (st.session_state) ajută la păstrarea valorilor între aceste reîncărcări.
    if "df_products" not in st.session_state:
        st.session_state.df_products = None

    st.subheader("Exercițiul 1: Creați un DataFrame de Produse")
    product_names = st.text_input("Introduceți numele produselor (separate prin virgulă)", "Laptop, Smartphone, Tablet")
    prices = st.text_input("Introduceți prețurile corespunzătoare (separate prin virgulă)", "1200, 800, 500")
    stock = st.text_input("Introduceți cantitățile (separate prin virgulă)", "50, 80, 120")

    if st.button("Creează DataFrame"):
        try:
            names = [x.strip() for x in product_names.split(",")]
            price_list = [float(x.strip()) for x in prices.split(",")]
            stock_list = [int(x.strip()) for x in stock.split(",")]
            st.session_state.df_products = pd.DataFrame({
                "Produs": names,
                "Preț": price_list,
                "Stoc": stock_list
            })
            st.write("DataFrame-ul de Produse:")
            st.write(st.session_state.df_products)
        except Exception as e:
            st.error(f"Eroare: {e}")

    st.subheader("Exercițiul 2: Filtrare și actualizare")
    st.write("""
    Actualizați DataFrame-ul de Produse adăugând o coloană 'Stare Stoc'.
    Marcați produsele cu stoc sub 70 ca "Stoc Scăzut" și pe celelalte ca "OK".
    """)
    if st.button("Aplică Actualizarea Stocului"):
        if st.session_state.df_products is not None:
            df = st.session_state.df_products
            df['Stare Stoc'] = "OK"
            df.loc[df['Stoc'] < 70, 'Stare Stoc'] = "Stoc Scăzut"
            st.write("DataFrame-ul de Produse Actualizat:")
            st.write(st.session_state.df_products)
        else:
            st.warning("Vă rugăm să creați mai întâi DataFrame-ul de Produse.")

    # Set de date exemplu
    data = pd.DataFrame({
        "Produs": ["Produs A", "Produs B", "Produs C", "Produs D"],
        "Preț": [100, 150, 200, 250]
    })

    st.subheader("Filtrare după preț")
    valoare_min = st.slider("Selectează prețul minim:", min_value=50, max_value=300, value=100)
    valoare_max = st.slider("Selectează prețul maxim:", min_value=50, max_value=300, value=250)

    # Filtrarea datelor
    data_filtrata = data[(data["Preț"] >= valoare_min) & (data["Preț"] <= valoare_max)]
    st.write("Produsele filtrate:", data_filtrata)

# ---------------------------
# Secțiunea: Analiza Exploratorie a Datelor
# ---------------------------
elif section == "Interacțiuni cu coloanele dataFrame":
    st.title("Interacțiuni cu coloanele dataFrame")

    # Inițializarea DataFrame-ului în session_state, dacă nu există deja
    if "df" not in st.session_state:
        data = {
            "A": [1, 2, np.nan, 4],
            "B": [5, np.nan, 7, 8],
            "C": [9, 10, 11, 12]
        }
        st.session_state.df = pd.DataFrame(data)

    # Obținem DataFrame-ul curent din session_state
    df = st.session_state.df

    st.write("### DataFrame curent:")
    st.dataframe(df)

    # Secțiunea pentru selectarea unei coloane
    col_names = list(df.columns)
    if col_names:
        selected_col = st.selectbox("Selectează o coloană:", col_names)
    else:
        st.write("Nu există coloane în DataFrame.")
        selected_col = None

    if selected_col is not None:
        st.subheader(f"Operații pentru coloana: **{selected_col}**")

        # Opțiunea 1: Redenumirea coloanei
        new_name = st.text_input("Introdu noul nume pentru coloană:", value=selected_col)
        if st.button("Redenumește coloana"):
            if new_name != selected_col:
                st.session_state.df = st.session_state.df.rename(columns={selected_col: new_name})
                st.success(f"Coloana '{selected_col}' a fost redenumită în '{new_name}'!")
                st.rerun()  # Rulăm din nou pentru a actualiza selecția
            else:
                st.info("Noul nume este același cu numele curent.")

        # Opțiunea 2: Ștergerea coloanei
        if st.button("Șterge coloana"):
            st.session_state.df = st.session_state.df.drop(columns=[selected_col])
            st.success(f"Coloana '{selected_col}' a fost ștearsă!")
            st.rerun()

        # Opțiunea 3: Afișarea numărului de valori lipsă din coloana selectată
        if st.button("Afișează numărul de valori lipsă"):
            missing_count = st.session_state.df[selected_col].isna().sum()
            st.info(f"Coloana '{selected_col}' are {missing_count} valori lipsă.")
            st.info(f"Coloana '{selected_col}' are {missing_count} valori lipsă.")

st.markdown("---")
st.write("Nu uitati de prezenta!")
