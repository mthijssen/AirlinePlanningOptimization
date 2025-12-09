#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:43:05 2025

@author: jimvanerp
"""
import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

from AE4423_20_Assignment_1A import build_df, build_OD_long

df, demand_data = build_df()
OD_long, model, k, b1, b2, b3 = build_OD_long()

# =============================================================================
# 1. LOAD AIRCRAFT DATA (Directly using Excel Headers)
# =============================================================================

aircraft_file = 'AircraftData.xlsx'
print(f"Loading aircraft data from {aircraft_file}...")

try:
    # Read and Transpose
    df_raw = pd.read_excel(aircraft_file, index_col=0)
    df_aircraft = df_raw.T
    
    # Cleaning
    df_aircraft.dropna(how='all', inplace=True)
    for col in df_aircraft.columns:
        df_aircraft[col] = pd.to_numeric(df_aircraft[col], errors='ignore')

    print("Aircraft Data Loaded Successfully:")
    # print(df_aircraft.head())

except Exception as e:
    print(f"\nCRITICAL ERROR LOADING EXCEL FILE: {e}")
    exit()

# =============================================================================
# 2. LOAD DEMAND & NETWORK DATA
# =============================================================================

HUB = 'EDDF' 
LF = 0.75    
FUEL_PRICE = 1.42 
WEEKS_PER_YEAR = 1 
OPERATING_HOURS_PER_WEEK = 10 * 7 
TAT_HUB_FACTOR = 1.5 
COST_HUB_DISCOUNT = 0.70 

df['Available slots'] = df['Available slots'].fillna(0).astype(int)

# print(df.head())
airports = df['ICAO Code'].tolist()
runway_lengths = df.set_index('ICAO Code')['Runway (m)'].to_dict()
slots_limit = df.set_index('ICAO Code')['Available slots'].to_dict()

OD_dist = OD_long[['Origin', 'Destination', 'Distance']]

dist_matrix = OD_dist.pivot(
    index="Origin",
    columns="Destination",
    values="Distance"
)

dist_matrix = dist_matrix.fillna(0)
dist_matrix = dist_matrix.reindex(index=airports, columns=airports)
demand_matrix = demand_data

# print(dist_matrix)
# print(demand_matrix)

"""
airports = ['EDDF', 'EGLL', 'LFPG', 'EHAM'] 
runway_lengths = {'EDDF': 4000, 'EGLL': 3900, 'LFPG': 4200, 'EHAM': 3800}
slots_limit = {'EDDF': 10000, 'EGLL': 9000, 'LFPG': 9500, 'EHAM': 8000}

# Placeholder Matrices (Replace with Q1A data)
dist_matrix = pd.DataFrame([
    [0, 650, 450, 360], [650, 0, 340, 370], [450, 340, 0, 400], [360, 370, 400, 0]
], index=airports, columns=airports)

demand_matrix = pd.DataFrame([
    [0, 2000, 1500, 1200], [2000, 0, 500, 400], [1500, 500, 0, 300], [1200, 400, 300, 0]
], index=airports, columns=airports)

print(dist_matrix)
print(demand_matrix)
"""

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def calculate_yield(distance):
    if distance == 0: return 0
    return 5.9 * (distance ** -0.76) + 0.043

def calculate_operating_cost(ac_type, origin, dest, dist):
    if dist == 0: return 0
    
    ac = df_aircraft.loc[ac_type]
    
    # Excel Headers
    c_fixed = ac['Fixed operating cost C_X [€]']
    speed   = ac['Speed [km/h]']
    c_time_param = ac['Time cost parameter C_T [€/hr]']
    c_fuel_param = ac['Fuel cost parameter C_F']
    
    # Logic
    flight_time = dist / speed
    c_time = c_time_param * flight_time
    c_fuel = (c_fuel_param * FUEL_PRICE / 1.5) * dist
    
    total_cost = c_fixed + c_time + c_fuel
    
    if origin == HUB or dest == HUB:
        total_cost *= COST_HUB_DISCOUNT
        
    return total_cost

# =============================================================================
# 4. GUROBI MODEL
# =============================================================================

m = Model('Airline_Network_1B')

A = airports
K = df_aircraft.index.tolist()

# Decision Variables
f = {} # Frequency
x = {} # Passengers
ac_count = {} # Fleet Size

print("\nInitializing Variables...")
for k in K:
    ac_count[k] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"Fleet_{k}")
    for i in A:
        for j in A:
            if i != j:
                f[k,i,j] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"Freq_{k}_{i}_{j}")

for i in A:
    for j in A:
        if i != j:
            x[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Pax_{i}_{j}")

m.update()

# --- Objective Function ---
total_revenue = 0
total_op_cost = 0
total_lease_cost = 0

# Revenue
for i in A:
    for j in A:
        if i == j: continue
        dist = dist_matrix.loc[i,j]
        yld = calculate_yield(dist)
        total_revenue += x[i,j] * yld * dist

# Operating Costs
for k in K:
    for i in A:
        for j in A:
            if i == j: continue
            dist = dist_matrix.loc[i,j]
            cost_per_flight = calculate_operating_cost(k, i, j, dist)
            total_op_cost += f[k,i,j] * cost_per_flight

# Leasing Costs
for k in K:
    total_lease_cost += ac_count[k] * df_aircraft.loc[k, 'Weekly lease cost [€]']

m.setObjective(total_revenue - total_op_cost - total_lease_cost, GRB.MAXIMIZE)

# --- Constraints ---
print("Adding Constraints...")

# C1 & C2: Demand and Capacity
for i in A:
    for j in A:
        if i == j: continue
        m.addConstr(x[i,j] <= demand_matrix.loc[i,j], name=f"Demand_{i}_{j}")
        
        total_capacity = quicksum(f[k,i,j] * df_aircraft.loc[k, 'Seats'] for k in K)
        m.addConstr(x[i,j] <= total_capacity * LF, name=f"Cap_{i}_{j}")

# C3: Flow Balance
for k in K:
    for i in A:
        arriving = quicksum(f[k,j,i] for j in A if j != i)
        departing = quicksum(f[k,i,j] for j in A if j != i)
        m.addConstr(arriving == departing, name=f"Balance_{k}_{i}")

# C4: Fleet Utilization
for k in K:
    total_time_needed = 0
    for i in A:
        for j in A:
            if i == j: continue
            dist = dist_matrix.loc[i,j]
            
            flight_time = dist / df_aircraft.loc[k, 'Speed [km/h]']
            tat_minutes = df_aircraft.loc[k, 'Average TAT [mins]']
            
            if j == HUB:
                tat_minutes *= TAT_HUB_FACTOR
            
            block_hours = flight_time + (tat_minutes / 60.0)
            total_time_needed += f[k,i,j] * block_hours
            
    m.addConstr(total_time_needed <= ac_count[k] * OPERATING_HOURS_PER_WEEK, name=f"Util_{k}")

# C5: Range and Runway
for k in K:
    ac_range = df_aircraft.loc[k, 'Maximum range [km]']
    ac_runway = df_aircraft.loc[k, 'Runway required [m]']
    
    for i in A:
        for j in A:
            if i == j: continue
            dist = dist_matrix.loc[i,j]
            r_origin = runway_lengths[i]
            r_dest = runway_lengths[j]
            
            if (dist > ac_range) or (ac_runway > r_origin) or (ac_runway > r_dest):
                m.addConstr(f[k,i,j] == 0, name=f"Inf_{k}_{i}_{j}")

# --- NEWLY ADDED: C6 Hub-and-Spoke Topology ---
for k in K:
    for i in A:
        for j in A:
            if i != j:
                # If NEITHER Origin NOR Dest is the Hub, force frequency to 0
                if i != HUB and j != HUB:
                    m.addConstr(f[k,i,j] == 0, name=f"HubSpoke_{k}_{i}_{j}")

# C7: Slots
for i in A:
    total_movements = quicksum((f[k,i,j] + f[k,j,i]) for k in K for j in A if j != i)
    m.addConstr(total_movements <= slots_limit[i], name=f"Slots_{i}")

# =============================================================================
# 5. OPTIMIZE
# =============================================================================

m.optimize()

if m.status == GRB.OPTIMAL:
    print("\n" + "="*60)
    print("                 OPTIMAL SOLUTION FOUND")
    print("="*60)
    print(f"Total Profit: €{m.ObjVal:,.2f}")
    
    print("\n--- FLEET PLAN ---")
    for k in K:
        count = ac_count[k].X
        if count > 0:
            print(f"  {k}: {int(count)} aircraft")
            
    print("\n--- NETWORK & FREQUENCY PLAN (Standard Week) ---")
    print(f"{'Aircraft':<20} | {'Origin':<6} -> {'Dest':<6} | {'Frequency':<10}")
    print("-" * 55)
    
    for k in K:
        for i in A:
            for j in A:
                if i != j:
                    freq = f[k,i,j].X
                    if freq > 0.5:
                        print(f"{k:<20} | {i:<6} -> {j:<6} | {int(round(freq))}")
    print("-" * 55)

else:
    print("Model Infeasible or Unbounded")