import pandas as pd
import numpy as np
import re
import os

# Căile către fișiere
raw_data_path = r"c:\Users\YAN\Desktop\PSW\Proiect PSW\Set de date\cars-dataset.csv"
output_path = r"c:\Users\YAN\Desktop\PSW\Proiect PSW\Set de date\Set masini 2023.csv"

# Funcție pentru extragere valori numerice (din transform_data.py)
def extract_number(text):
    if pd.isna(text):
        return np.nan
    text = str(text)
    # Extrage primul grup de cifre (inclusiv zecimale)
    match = re.search(r'[\d]+\.?[\d]*', text.replace(',', ''))
    if match:
        try:
            return float(match.group())
        except:
            return np.nan
    return np.nan

def normalize_dimension(val):
    """Normalizează dimensiunile la cm."""
    if pd.isna(val): return val
    if val > 600: return val / 10.0   # mm -> cm
    if val < 200: return val * 2.54   # inci -> cm
    return val # Deja în cm

def normalize_weight(val):
    """Normalizează greutatea la kg (presupunem lbs -> kg)."""
    if pd.isna(val): return val
    if val > 500: return val * 0.453592
    return val

print(f"Încărcare date brute din: {raw_data_path}...")
df_raw = pd.read_csv(raw_data_path, low_memory=False)

# Eliminăm coloana index reziduală dacă există
if 'Unnamed: 0' in df_raw.columns:
    df_raw = df_raw.drop(columns=['Unnamed: 0'])

print("Filtrare pentru anul 2023...")
df_2023 = df_raw[df_raw['Production years'].str.contains('2023', na=False)].copy()

# Listă coloane ce trebuie transformate în numerice (din transform_data.py)
numeric_cols = [
    'Power(HP)', 'Torque(Nm)', 'Displacement', 'Length', 'Width', 'Height',
    'Wheelbase', 'Cargo Volume', 'Unladen Weight', 'Top Speed',
    'Acceleration 0-62 Mph (0-100 kph)', 'Combined mpg', 'Fuel capacity'
]

print("Transformare coloane în format numeric (păstrând valorile lipsă)...")
for col in numeric_cols:
    if col in df_2023.columns:
        df_2023[col] = df_2023[col].apply(extract_number)

# --- NORMALIZARE UNITĂȚI ---
print("Normalizare unități (conversie la CM și KG)...")
dimension_cols = ['Wheelbase', 'Length', 'Width', 'Height']
for col in dimension_cols:
    if col in df_2023.columns:
        df_2023[col] = df_2023[col].apply(normalize_dimension)

if 'Unladen Weight' in df_2023.columns:
    df_2023['Unladen Weight'] = df_2023['Unladen Weight'].apply(normalize_weight)

print(f"Salvare subset 2023 ({len(df_2023)} rânduri) în: {output_path}...")
df_2023.to_csv(output_path, index=False)

print("Operațiune finalizată cu succes!")
