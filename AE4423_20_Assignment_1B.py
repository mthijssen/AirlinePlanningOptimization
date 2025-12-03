import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from AE4423_20_Assignment_1A import build_df, build_OD_long

df, demand_data = build_df()
OD_long, model, k, b1, b2, b3 = build_OD_long()


# print(demand_data)

# aircraft_data = pd.read_excel('AircraftData.xlsx')

# print(aircraft_data.head())

print(OD_long)