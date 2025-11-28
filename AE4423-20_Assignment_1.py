# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:37:41 2025

@author: silha, jimvanerp, mathijssen
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

airport_data = pd.read_excel('DemandGroup12.xlsx', index_col=1, header=2)
airport_data = airport_data.head(6).T
demand_data = pd.read_excel('DemandGroup12.xlsx', index_col=1, header=11)
pop = pd.read_excel('pop.xlsx', header=2)

""" Plan is nu om voor elke D_ij de pop data en gdp data aan de juiste i en j te koppelen
en dan voor alle D_ij's de OLS toe te passen """

"""Eerste stap daar in is elke ICAO code matchen met alle city names, pop data, en gdp data. Dat gaan we hier onder doen"""

airport_data = airport_data.rename(columns={airport_data.columns[0]: "City"})

df = pd.merge(airport_data, pop, on="City", how="left")
df = df.drop(columns=["Unnamed: 3"], index=0).reset_index(drop=True)
df = df.rename(columns={
    2021: "Population_2021",
    2024: "Population_2024",
    "2021.1": "GDP_2021",
    "2024.1": "GDP_2024"
})

cols_to_convert = ["Latitude (deg)", "Longitude (deg)", "Runway (m)", "Available slots"]

df[cols_to_convert] = df[cols_to_convert].apply(
    pd.to_numeric, errors="coerce"
)

"""Alle data is nu gematcht aan elkaar. Volgende stap is de origin en destination data koppelen aan alle entries in de tabel"""

demand_data = demand_data.drop(columns=["Demand per week"])

OD = demand_data #OD staat voor Origin en Destination
OD_long = OD.stack().reset_index()
OD_long.columns = ["Origin", "Destination", "Demand"]

OD_long = OD_long.merge(
    df.add_prefix("O_"),
    left_on="Origin",
    right_on="O_ICAO Code",
    how="left"
)

OD_long = OD_long.merge(
    df.add_prefix("D_"),
    left_on="Destination",
    right_on="D_ICAO Code",
    how="left"
)

OD_long = OD_long[OD_long["Origin"] != OD_long["Destination"]]

"""Nu wordt de lengte tussen twee vliegvelden uitgerekend"""

R_E = 6371       # Radius of Earth (km)
f_cost = 1.42    # Fuel cost factor

def calc_dist(row):
    lat1, lon1 = np.radians(row['O_Latitude (deg)']), np.radians(row['O_Longitude (deg)'])
    lat2, lon2 = np.radians(row['D_Latitude (deg)']), np.radians(row['D_Longitude (deg)'])
    dphi = lat1 - lat2
    dlam = lon1 - lon2
    a = np.sin(dphi/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlam/2)**2
    return 2 * R_E * np.arcsin(np.sqrt(a))

OD_long['Distance'] = OD_long.apply(calc_dist, axis=1)

# print(OD_long.columns)

"""Nu hebben we alle informatie om de linearisering te doen dus nu die ln formules toepassen en de OLS uit te rekenen"""

OD_long['ln_D'] = np.log(OD_long['Demand'])
OD_long['ln_PP'] = np.log(OD_long['O_Population_2021'] * OD_long['D_Population_2021'])
OD_long['ln_GDP'] = np.log(OD_long['O_GDP_2021'] * OD_long['D_GDP_2021'])
OD_long['ln_C'] = np.log(f_cost * OD_long['Distance'])

X = OD_long[['ln_PP', 'ln_GDP', 'ln_C']]
X = sm.add_constant(X) 
Y = OD_long['ln_D']

model = sm.OLS(Y, X).fit()

k = np.exp(model.params['const'])
b1 = model.params['ln_PP']
b2 = model.params['ln_GDP']
b3 = -model.params['ln_C']

print("\n" + "="*40)
print("CALIBRATION RESULTS")
print("="*40)
print(f"k  (Scaling Factor): {k}")
print(f"b1 (Population):     {b1}")
print(f"b2 (GDP):            {b2}")
print(f"b3 (Distance):       {b3}")
print("="*40)
print(model.summary())

"""Plotten van de data"""

OD_long['Est_Demand'] = k * (
    (OD_long['O_Population_2021']*OD_long['D_Population_2021'])**b1 * (OD_long['O_GDP_2021']*OD_long['D_GDP_2021'])**b2
) / ((f_cost*OD_long['Distance'])**b3)

plt.figure(figsize=(10,6))
plt.scatter(OD_long['Demand'], OD_long['Est_Demand'], alpha=0.6)
plt.plot([0, OD_long['Demand'].max()], [0, OD_long['Demand'].max()], 'r--', label='Perfect Fit')
plt.xlabel("Actual Demand (2021)")
plt.ylabel("Estimated Demand (Model)")
plt.title(f"Gravity Model Calibration\nR2: {model.rsquared:.3f}")
plt.legend()
plt.grid(True)
plt.show()