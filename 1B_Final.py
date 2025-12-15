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
LF = 0.75                           #Average load factor 75% 
FUEL_PRICE = 1.42                   #Fixed fuel price per EUR/gallon
WEEKS_PER_YEAR = 1                  #Flight schedule for 1 week
OPERATING_HOURS_PER_WEEK = 10 * 7   #10 operating hours per week
TAT_HUB_FACTOR = 1.5                #50% extra turn around time at the hub
COST_HUB_DISCOUNT = 0.70            #All operating costs 30% lower at the hub

df['Available slots'] = df['Available slots'].fillna(0).astype(int)

# =============================================================================
# 1. Fix Hub Slots (Convert hub slots 0 to 99999)

df.loc[df['ICAO Code'] == HUB, 'Available slots'] = 99999

# =============================================================================

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


# =============================================================================
# 3. Yield & Costs calculations
# =============================================================================

def calculate_yield(distance):      #Yield per fligtht per distance in EURO        
    if distance == 0: return 0      #Prevent Yield from flying to own airport
    return 5.9 * (distance ** -0.76) + 0.043

def calculate_operating_cost(ac_type, origin, dest, dist): #Operating costs, bases on Fixed, Time-based and Fuel costs
    if dist == 0: return 0          
    
    ac = df_aircraft.loc[ac_type]
    
    # Excel Headers from Excel
    c_fixed = ac['Fixed operating cost C_X [€]']        #Fixed costs
    speed   = ac['Speed [km/h]']
    c_time_param = ac['Time cost parameter C_T [€/hr]']
    c_fuel_param = ac['Fuel cost parameter C_F']
    
    # Calculations
    flight_time = dist / speed
    c_time = c_time_param * flight_time                 #Time-Based costs
    c_fuel = (c_fuel_param * FUEL_PRICE / 1.5) * dist   #Fuel Costs
    
    total_cost = c_fixed + c_time + c_fuel              #Total costs
    
    if origin == HUB or dest == HUB:                    #Implement HUB discount
        total_cost *= COST_HUB_DISCOUNT
        
    return total_cost

# =============================================================================
# 4. GUROBI MODEL
# =============================================================================
#Define the Hub Parameter 'g'
# g[i] = 0 if i is the HUB, 1 otherwise
g = {node: 0 if node == HUB else 1 for node in airports}

m = Model('Airline_Network_1B')

N = airports                    #AANPASSEN AAN DATA  #Set of airports N 
K = df_aircraft.index.tolist()  #AANPASSSEN AAN DATA #Set of aircrafts type K

# Decision Variables
f = {}          # Frequency 
x = {}          # Passengers
ac_count = {}   # Fleet Size
w = {}          # flow from airport i to airport j that transfers at hub 


print("\nInitializing Variables...")
for k in K:
    ac_count[k] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"Fleet_{k}")
    for i in N:
        for j in N:
            if i != j:
                f[k,i,j] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"Freq_{k}_{i}_{j}")

for i in N:
    for j in N:
        if i != j:
            x[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Pax_{i}_{j}")
            
for i in N:
    for j in N:
        if i != j:
            w[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"PaxHub_{i}_{j}")

m.update()

# --- Objective Function ---
total_revenue = 0
total_op_cost = 0
total_lease_cost = 0

# Revenue
for i in N:
    for j in N:
        if i == j: continue
        dist = dist_matrix.loc[i,j]
        yld = calculate_yield(dist)
        
        total_revenue += (x[i,j] + w[i,j]) * yld * dist #added flow to revenue

# Operating Costs
for k in K:
    for i in N:
        for j in N:
            if i == j: continue
            dist = dist_matrix.loc[i,j]
            cost_per_flight = calculate_operating_cost(k, i, j, dist)
            total_op_cost += f[k,i,j] * cost_per_flight

# Leasing Costs
for k in K:
    total_lease_cost += ac_count[k] * df_aircraft.loc[k, 'Weekly lease cost [€]']

#Objective function
m.setObjective(total_revenue - total_op_cost - total_lease_cost, GRB.MAXIMIZE)

# --- Constraints ---
print("Adding Constraints...")

for i in N:
    for j in N:
        if i == j: continue

        # --- 1. Volume Constraint (Demand) ---
        # Total passengers (Direct + Transfer) <= Demand
        m.addConstr(x[i,j] + w[i,j] <= demand_matrix.loc[i,j], name=f"Demand_{i}_{j}") # Total flow (passenger flow + hub flow) smaller than demand

        # --- 2. Transfer Feasibility (Hub Logic) ---
        # If i is Hub OR j is Hub, w[i,j] becomes 0.
        # This forces transfers to only exist for Spoke-to-Spoke pairs.
        m.addConstr(w[i,j] <= demand_matrix.loc[i,j] * g[i] * g[j], name=f"TransferLogic_{i}_{j}")

# C1 & C2: Demand and Capacity
for i in N:
    for j in N:
        if i == j: continue
        
        total_capacity = quicksum(f[k,i,j] * df_aircraft.loc[k, 'Seats'] for k in K)   #
        m.addConstr(
            x[i,j] +                                               #load local Pax (origen to hub)
            quicksum(w[i,m] * (1 - g[j]) for m in N if m != i) +   #load inbound Pax (origen to hub as transfer) m!= i TOEVOEGEN AAN CONSTRAINS
            quicksum(w[m,j] * (1 - g[i]) for m in N if m != j)     #load outbound Pax (hub to destination as tranfer)
            <= total_capacity * LF, name=f"Cap_{i}_{j}")

# C3: Flow Balance
for k in K:
    for i in N:
        arriving = quicksum(f[k,j,i] for j in N if j != i)
        departing = quicksum(f[k,i,j] for j in N if j != i)
        m.addConstr(arriving == departing, name=f"Balance_{k}_{i}")

# C4: Fleet Utilization
for k in K:
    total_time_needed = 0
    for i in N:
        for j in N:
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
    
    for i in N:
        for j in N:
            if i == j: continue
            dist = dist_matrix.loc[i,j]
            r_origin = runway_lengths[i]
            r_dest = runway_lengths[j]
            
            if (dist > ac_range) or (ac_runway > r_origin) or (ac_runway > r_dest):
                m.addConstr(f[k,i,j] == 0, name=f"Inf_{k}_{i}_{j}")

# --- NEWLY ADDED: C6 Hub-and-Spoke Topology ---
for k in K:
    for i in N:
        for j in N:
            if i != j:
                # If NEITHER Origin NOR Dest is the Hub, force frequency to 0
                if i != HUB and j != HUB:
                    m.addConstr(f[k,i,j] == 0, name=f"HubSpoke_{k}_{i}_{j}")

# C7: Slots
for i in N:
    total_movements = quicksum((f[k,i,j] + f[k,j,i]) for k in K for j in N if j != i)
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
        for i in N:
            for j in N:
                if i != j:
                    freq = f[k,i,j].X
                    if freq > 0.5:
                        print(f"{k:<20} | {i:<6} -> {j:<6} | {int(round(freq))}")
    print("-" * 55)

else:
    print("Model Infeasible or Unbounded")