#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 17:34:20 2025

@author: jimvanerp
"""

import pandas as pd
from gurobipy import Model, GRB, quicksum
import sys
import os

# =============================================================================
# 1. DATA LOADING
# =============================================================================

# Define filenames
df_flights = pd.read_excel('Group_12.xlsx', sheet_name=0)
df_itineraries = pd.read_excel('Group_12.xlsx', sheet_name=1)
df_recapture = pd.read_excel('Group_12.xlsx', sheet_name=2)

# --- Data Structures ---
L = df_flights['Flight No.'].tolist()
capacity = df_flights.set_index('Flight No.')['Capacity'].to_dict()
demand_data = df_itineraries.set_index('Itinerary')['Demand'].to_dict()
fare_data = df_itineraries.set_index('Itinerary')['Price [EUR]'].to_dict()

# Map Itinerary -> Legs
itin_legs = {}
for idx, row in df_itineraries.iterrows():
    p = row['Itinerary']
    legs = []
    if pd.notna(row['Flight 1']): legs.append(row['Flight 1'])
    if pd.notna(row['Flight 2']): legs.append(row['Flight 2'])
    itin_legs[p] = legs

# Recapture options
recapture_options = []
for idx, row in df_recapture.iterrows():
    recapture_options.append({
        'orig': row['From Itinerary'],
        'recap': row['To Itinerary'],
        'rate': row['Recapture Rate']
    })

# =============================================================================
# 2. GUROBI MODEL SETUP (Extensive / Path-Based)
# =============================================================================

m = Model("Path_Based_Extensive")

# --- Decision Variables ---
# x_orig[p]: Passengers on original itinerary p
x_orig = {} 
for p in demand_data:
    x_orig[p] = m.addVar(obj=fare_data[p], vtype=GRB.CONTINUOUS, lb=0, name=f"X_orig_{p}")

# x_recap[orig, recap]: Passengers on recapture itinerary
# Note: Objective = Expected Revenue = Rate * Fare_recap
# (This matches the standard "Relaxed" Path-Based logic)
x_recap = {}
for opt in recapture_options:
    orig = opt['orig']
    recap = opt['recap']
    rate = opt['rate']
    
    if orig in demand_data:
        # Expected Revenue Logic
        revenue = rate * fare_data.get(recap, 0)
        x_recap[(orig, recap)] = m.addVar(obj=revenue, vtype=GRB.CONTINUOUS, lb=0, name=f"X_recap_{orig}_{recap}")

m.modelSense = GRB.MAXIMIZE
m.update()

# --- Constraints ---

# 1. Demand Constraints
# Total passengers served for original demand p <= Demand_p
# Passengers served = x_orig_p + sum(x_recap_pr)
for p in demand_data:
    # Find all recapture variables starting from p
    recap_vars = [x_recap[(o, r)] for (o, r) in x_recap if o == p]
    
    m.addConstr(x_orig[p] + quicksum(recap_vars) <= demand_data[p], name=f"Dem_{p}")

# 2. Capacity Constraints
# Total load on leg l <= Capacity_l
# Load = Passengers on Original paths using l + Passengers on Recapture paths using l
for l in L:
    terms = []
    
    # A. Original Paths using leg l
    for p in demand_data:
        if l in itin_legs.get(p, []):
            terms.append(x_orig[p])
            
    # B. Recapture Paths using leg l
    # Iterate through all recapture variables x_recap[(orig, recap)]
    # Check if the 'recap' itinerary uses leg l
    for (orig, recap) in x_recap:
        if l in itin_legs.get(recap, []):
            terms.append(x_recap[(orig, recap)])
            
    m.addConstr(quicksum(terms) <= capacity[l], name=f"Cap_{l}")
    


# =============================================================================
# 3. SOLVE AND REPORT
# =============================================================================

m.optimize()

print("\n" + "="*50)
print("PATH-BASED EXTENSIVE RESULT (NO COL GEN)")
print("="*50)

if m.status == GRB.OPTIMAL:
    print(f"Optimal Objective (Total Revenue): €{m.ObjVal:,.2f}")
    
    # Calculate Equivalent Spill Cost
    total_potential = sum(demand_data[p] * fare_data[p] for p in demand_data)
    equiv_spill = total_potential - m.ObjVal
    
    print("-" * 50)
    print(f"Total Potential Revenue:      €{total_potential:,.2f}")
    print(f"Equivalent Spill Cost (Lost): €{equiv_spill:,.2f}")
    print("-" * 50)
    
    # Stats
    total_orig = sum(v.X for v in x_orig.values())
    total_recap_pax = sum(v.X for v in x_recap.values())
    print(f"Total Pax on Original Paths: {int(total_orig)}")
    print(f"Total Pax on Recapture Paths: {int(total_recap_pax)}")

else:
    print("Model did not solve to optimality.")