"""
Script pentru pregatirea datelor pentru ACP - VERSIUNEA FINALA
Cars Specification Dataset - 2023
Cu curatare OUTLIERS
"""

import pandas as pd
import numpy as np
import sys
import io
import re

# Fix pentru encoding Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Citire date
df = pd.read_csv('cars-dataset.csv', low_memory=False)

print("=" * 80)
print("PREGATIRE DATE PENTRU ACP - CARS 2023 (FINAL)")
print("=" * 80)

# PASUL 1: Filtram pe 2023
df_2023 = df[df['Production years'].str.contains('2023', na=False)].copy()
print(f"\n[1] FILTRARE PE ANUL 2023")
print(f"    Observatii initiale: {len(df_2023)}")

# Functie pentru extragere valori numerice
def extract_number(text):
    if pd.isna(text):
        return np.nan
    text = str(text)
    match = re.search(r'[\d]+\.?[\d]*', text.replace(',', ''))
    if match:
        try:
            return float(match.group())
        except:
            return np.nan
    return np.nan

# VARIABILE SELECTATE (13)
variabile_optime = {
    'Power(HP)': 'Putere motor (HP)',
    'Torque(Nm)': 'Cuplu motor (Nm)',
    'Displacement': 'Cilindree (cm3)',
    'Length': 'Lungime (inch)',
    'Width': 'Latime (inch)',
    'Height': 'Inaltime (inch)',
    'Wheelbase': 'Ampatament (inch)',
    'Cargo Volume': 'Volum portbagaj (cuFT)',
    'Unladen Weight': 'Greutate (lbs)',
    'Top Speed': 'Viteza maxima (mph)',
    'Acceleration 0-62 Mph (0-100 kph)': 'Acceleratie 0-100 (secunde)',
    'Combined mpg': 'Consum combinat (mpg)',
    'Fuel capacity': 'Capacitate rezervor (gallons)',
}

print(f"\n[2] VARIABILE SELECTATE ({len(variabile_optime)})")
print("-" * 60)
for col, desc in variabile_optime.items():
    print(f"    - {desc}")

# Extragem valorile numerice
df_numeric = pd.DataFrame()
df_numeric['Company'] = df_2023['Company']
df_numeric['Model'] = df_2023['Model']
df_numeric['Segment'] = df_2023['Segment']
df_numeric['Fuel'] = df_2023['Fuel']

for col in variabile_optime.keys():
    if col in df_2023.columns:
        df_numeric[col] = df_2023[col].apply(extract_number)

# Eliminam randurile cu valori lipsa
numeric_cols = list(variabile_optime.keys())
df_clean = df_numeric.dropna(subset=numeric_cols).copy()

print(f"\n[3] DUPA ELIMINARE VALORI LIPSA: {len(df_clean)} randuri")

# PASUL 4: CURATARE OUTLIERS (valori imposibile)
print(f"\n[4] CURATARE OUTLIERS (valori aberante)")
print("-" * 60)

# Definim limite realiste pentru fiecare variabila
limite = {
    'Power(HP)': (30, 2000),           # Cai putere: 30-2000 (de la Smart la Bugatti)
    'Torque(Nm)': (30, 2500),          # Cuplu: 30-2500 Nm
    'Displacement': (500, 8500),       # Cilindree: 500cc - 8.5L
    'Length': (100, 250),              # Lungime: 100-250 inch (~2.5m - 6.3m)
    'Width': (55, 90),                 # Latime: 55-90 inch (~1.4m - 2.3m)
    'Height': (40, 90),                # Inaltime: 40-90 inch (~1.0m - 2.3m)
    'Wheelbase': (70, 170),            # Ampatament: 70-170 inch
    'Cargo Volume': (0.5, 150),        # Portbagaj: 0.5-150 cuFT
    'Unladen Weight': (1000, 8000),    # Greutate: 1000-8000 lbs
    'Top Speed': (50, 300),            # Viteza: 50-300 mph
    'Acceleration 0-62 Mph (0-100 kph)': (1.5, 25),  # Acceleratie: 1.5-25 secunde
    'Combined mpg': (5, 150),          # Consum: 5-150 mpg
    'Fuel capacity': (5, 50),          # Rezervor: 5-50 gallons (19-190 litri)
}

inainte = len(df_clean)
for col, (min_val, max_val) in limite.items():
    mask = (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
    eliminati = (~mask).sum()
    if eliminati > 0:
        print(f"    {col}: eliminat {eliminati} randuri (afara din [{min_val}, {max_val}])")
    df_clean = df_clean[mask]

print(f"\n    Total eliminat ca outliers: {inainte - len(df_clean)}")
print(f"    Ramas: {len(df_clean)} randuri")

# Verificam raportul n:p
n = len(df_clean)
p = len(numeric_cols)
print(f"\n    RAPORT n:p = {n}:{p} = {n/p:.1f}:1", end="")
if n/p >= 10:
    print(" (EXCELENT!)")
elif n/p >= 5:
    print(" (BUN)")
else:
    print(" (ACCEPTABIL)")

# Statistici FINALE
print(f"\n[5] STATISTICI FINALE")
print("=" * 80)
stats = df_clean[numeric_cols].describe().T
stats = stats[['count', 'mean', 'std', 'min', 'max']]
print(stats.round(2).to_string())

# Marci
print(f"\n[6] MARCI DISPONIBILE ({df_clean['Company'].nunique()} marci)")
print("-" * 60)
brand_counts = df_clean['Company'].value_counts()
for brand, count in brand_counts.head(30).items():
    print(f"    {brand}: {count}")

# Salvare
output_file = 'cars_2023_ACP.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n" + "=" * 80)
print(f"DATE FINALE SALVATE IN: {output_file}")
print(f"Observatii: {len(df_clean)}")
print(f"Variabile numerice: {len(numeric_cols)}")
print(f"Marci: {df_clean['Company'].nunique()}")
print("=" * 80)
