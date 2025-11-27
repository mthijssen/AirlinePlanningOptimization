import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# --- 1. SETUP PATHS ---
# Your specific directory
#BASE_DIR = r'/Users/jimvanerp/Documents/GitHub/AirlinePlanningOptimization'

# Define full paths to your files
demand_file = 'DemandGroup12.xlsx'
pop_file = 'pop.xlsx'

# --- CONSTANTS ---
R_E = 6371       # Radius of Earth (km)
F_COST = 1.42    # Fuel cost factor

# --- 2. LOAD DATA ---
print("Loading files...")
# We use header=None to keep the row indices consistent with the raw Excel structure
# engine='openpyxl' is the standard for reading .xlsx files
df_raw = pd.read_excel(demand_file, header=None, engine='openpyxl')
df_pop_raw = pd.read_excel(pop_file, header=None, engine='openpyxl')

# --- 3. EXTRACT AIRPORT DATA ---
# Based on the file structure:
# Row 4 (Index 3): City Names
# Row 5 (Index 4): ICAO Codes
# Row 6 (Index 5): Latitude
# Row 7 (Index 6): Longitude
# Columns 2 to 21 (C to V) contain the airport data

airport_cols = list(range(2, 22)) 

cities = df_raw.iloc[3, airport_cols].values
icaos = df_raw.iloc[4, airport_cols].values
lats = df_raw.iloc[5, airport_cols].values
lons = df_raw.iloc[6, airport_cols].values

df_airports = pd.DataFrame({
    'City': [str(c).strip() for c in cities],
    'ICAO': [str(i).strip() for i in icaos],
    'Lat': pd.to_numeric(lats),
    'Lon': pd.to_numeric(lons)
})

# --- 4. EXTRACT POPULATION & GDP ---
# From pop.xlsx (starting Row 3 / Index 2)
# Col 0: City, Col 1: Pop2021, Col 4: Country, Col 5: GDP2021
df_pop = df_pop_raw.iloc[2:, [0, 1, 4, 5]].copy()
df_pop.columns = ['City', 'Pop_2021', 'Country', 'GDP_2021']

# Clean data
df_pop['City'] = df_pop['City'].astype(str).str.strip()
df_pop['Pop_2021'] = pd.to_numeric(df_pop['Pop_2021'])
df_pop['GDP_2021'] = pd.to_numeric(df_pop['GDP_2021'])

# --- 5. MERGE DATA ---
df_master = pd.merge(df_airports, df_pop, on='City', how='inner')
print(f"Successfully matched {len(df_master)} airports with population data.")

# --- 6. EXTRACT DEMAND MATRIX ---
# Matrix Headers (Destinations) are at Row 12 (Index 11)
# Data starts at Row 13 (Index 12)
demand_list = []
start_row = 12

for i in range(len(df_master)):
    curr_row = start_row + i
    if curr_row >= len(df_raw): break
    
    # Origin is in Column 1 (B)
    origin_icao = str(df_raw.iloc[curr_row, 1]).strip()
    
    # Demand values are in Columns 2-21
    row_demands = df_raw.iloc[curr_row, 2:22].values
    
    for j, dest_icao in enumerate(icaos):
        val = row_demands[j]
        try:
            val = float(val)
        except:
            val = 0
            
        # Filter valid demands (ignore diagonals and zeros)
        if val > 0 and origin_icao != str(dest_icao).strip():
            demand_list.append({
                'Origin': origin_icao,
                'Destination': str(dest_icao).strip(),
                'Demand': val
            })

df_demand = pd.DataFrame(demand_list)

# --- 7. PREPARE REGRESSION MODEL ---
# Merge Origin Info
df_model = pd.merge(df_demand, df_master, left_on='Origin', right_on='ICAO')
df_model = df_model.rename(columns={'Pop_2021': 'Pop_O', 'GDP_2021': 'GDP_O', 'Lat': 'Lat_O', 'Lon': 'Lon_O'})

# Merge Destination Info
df_model = pd.merge(df_model, df_master, left_on='Destination', right_on='ICAO', suffixes=('_O', '_D'))
df_model = df_model.rename(columns={'Pop_2021': 'Pop_D', 'GDP_2021': 'GDP_D', 'Lat': 'Lat_D', 'Lon': 'Lon_D'})

# Calculate Distance
def calc_dist(row):
    lat1, lon1 = np.radians(row['Lat_O']), np.radians(row['Lon_O'])
    lat2, lon2 = np.radians(row['Lat_D']), np.radians(row['Lon_D'])
    dphi = lat1 - lat2
    dlam = lon1 - lon2
    a = np.sin(dphi/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlam/2)**2
    return 2 * R_E * np.arcsin(np.sqrt(a))

df_model['Distance'] = df_model.apply(calc_dist, axis=1)

# Linearization (Log transformation)
df_model['ln_D'] = np.log(df_model['Demand'])
df_model['ln_PP'] = np.log(df_model['Pop_O'] * df_model['Pop_D'])
df_model['ln_GP'] = np.log(df_model['GDP_O'] * df_model['GDP_D'])
df_model['ln_C'] = np.log(F_COST * df_model['Distance'])

# --- 8. RUN OLS REGRESSION ---
X = df_model[['ln_PP', 'ln_GP', 'ln_C']]
X = sm.add_constant(X) # Adds the intercept for 'k'
Y = df_model['ln_D']

model = sm.OLS(Y, X).fit()

# Extract Parameters
k = np.exp(model.params['const'])
b1 = model.params['ln_PP']
b2 = model.params['ln_GP']
b3 = -model.params['ln_C']

print("\n" + "="*40)
print("CALIBRATION RESULTS")
print("="*40)
print(f"k  (Scaling Factor): {k:.5e}")
print(f"b1 (Population):     {b1:.5f}")
print(f"b2 (GDP):            {b2:.5f}")
print(f"b3 (Distance):       {b3:.5f}")
print("="*40)
print(model.summary())

# --- 9. PLOT RESULTS ---
df_model['Est_Demand'] = k * (
    (df_model['Pop_O']*df_model['Pop_D'])**b1 * (df_model['GDP_O']*df_model['GDP_D'])**b2
) / ((F_COST*df_model['Distance'])**b3)

plt.figure(figsize=(10,6))
plt.scatter(df_model['Demand'], df_model['Est_Demand'], alpha=0.6)
plt.plot([0, df_model['Demand'].max()], [0, df_model['Demand'].max()], 'r--', label='Perfect Fit')
plt.xlabel("Actual Demand (2021)")
plt.ylabel("Estimated Demand (Model)")
plt.title(f"Gravity Model Calibration\nR2: {model.rsquared:.3f}")
plt.legend()
plt.grid(True)
plt.show()