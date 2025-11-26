# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:37:41 2025

@author: silha
"""
import numpy as np
import pandas as pd
# import statsmodels.api as sm
#import mathplotlib.pyplot as plt

airport_data = pd.read_excel('DemandGroup12.xlsx', index_col=1, header=3)
demand_data = pd.read_excel('DemandGroup12.xlsx', index_col=1, header=11)
pop = pd.read_excel('pop.xlsx', header=2)

# data.head()
# pop.head()

# print(airport_data.head())
# print(airport_data.head())
# print(pop.head())

""" Plan is nu om voor elke D_ij de pop data en gdp data aan de juiste i en j te koppelen
en dan voor alle D_ij's de OLS toe te passen """

for idx, row in airport_data:
    icao = row['ICAO Code']

print(icao)
