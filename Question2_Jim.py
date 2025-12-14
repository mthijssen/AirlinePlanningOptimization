#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:43:05 2025

@author: jimvanerp
"""
import pandas as pd
from gurobipy import Model, GRB, quicksum

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("Loading data files...")

# Use the CSV filenames provided in the upload
# If running locally with the original Excel, you would use pd.read_excel('Group_12.xlsx', sheet_name='...')
try:
    df_flights = pd.read_excel('Group_12.xlsx', sheet_name=0)
    df_itineraries = pd.read_excel('Group_12.xlsx', sheet_name=1)
    df_recapture = pd.read_excel('Group_12.xlsx', sheet_name=2)
except FileNotFoundError:
    print("Error: CSV files not found. Please ensure 'Group_12.xlsx - Flights.csv', etc. are in the folder.")
    exit()

# --- CLEANING & PREPARATION ---

# 1. Sets
L = df_flights['Flight No.'].tolist()  # Set of Flight Legs
P = df_itineraries['Itinerary'].tolist() # Set of Itineraries

# 2. Parameters (Dictionaries for fast lookup)
# Capacity: CAP[l]
capacity = df_flights.set_index('Flight No.')['Capacity'].to_dict()

# Demand: D[p]
demand = df_itineraries.set_index('Itinerary')['Demand'].to_dict()

# Fare: Fare[p]
fare = df_itineraries.set_index('Itinerary')['Price [EUR]'].to_dict()

# Recapture Rates: b[p, r]
# Dictionary keys will be tuples (from_itin, to_itin)
recapture_rates = {}
for index, row in df_recapture.iterrows():
    p = row['From Itinerary']
    r = row['To Itinerary']
    rate = row['Recapture Rate']
    recapture_rates[(p, r)] = rate

# 3. Incidence Matrix (Delta[p, l])
# Mapping which itinerary uses which flight leg(s)
# Itinerary CSV has 'Flight 1' and optional 'Flight 2'
itin_flights = {}
for index, row in df_itineraries.iterrows():
    p = row['Itinerary']
    legs = []
    if pd.notna(row['Flight 1']):
        legs.append(row['Flight 1'])
    if pd.notna(row['Flight 2']):
        legs.append(row['Flight 2'])
    itin_flights[p] = legs

# Helper to check if itinerary p uses leg l
def uses_leg(p, l):
    return l in itin_flights.get(p, [])

print(f"Data Loaded: {len(P)} itineraries, {len(L)} flights.")

# =============================================================================
# 2. GUROBI MODEL SETUP
# =============================================================================

m = Model('Passenger_Mix_Flow')

# --- Decision Variables ---
# t[p]: Transported on original itinerary p
# s[p]: Spilled from itinerary p (t_p^0 in your slide notation)
# recap[p, r]: Recaptured from p to r (t_p^r in your slide notation)

t = {} # Original
s = {} # Spilled
recap = {} # Recaptured

# Create variables
for p in P:
    t[p] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Trans_{p}")
    s[p] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Spill_{p}")
    
    # Recapture variables only exist if a rate is defined in the file
    # Iterate through possible recapture targets for p
    # We look at the recapture_rates keys that start with p
    for (origin_p, target_r), rate in recapture_rates.items():
        if origin_p == p:
            recap[p, target_r] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Recap_{p}_{target_r}")

m.update()

# --- Objective Function (Minimize Spill Costs) ---
# Objective: Minimize sum(Fare_p * s_p) + sum((Fare_p - b_pr * Fare_r) * t_pr)
# Note: Slide 13 uses exactly this cost structure for lost revenue/opportunity cost.

spill_cost = quicksum(fare[p] * s[p] for p in P)

recap_cost = quicksum(
    (fare[p] - rate * fare[target_r]) * recap[p, target_r]
    for (p, target_r), rate in recapture_rates.items()
    if (p, target_r) in recap # check if var exists
)

m.setObjective(spill_cost + recap_cost, GRB.MINIMIZE)

# --- Constraints ---

# 1. Demand Conservation
# t_p + s_p + sum(recaptured from p) = Demand_p
for p in P:
    # Sum of all people moving FROM p TO somewhere else
    # We iterate over recapture_rates to find pairs starting with p
    total_recaptured_out = quicksum(
        recap[p, r] 
        for (orig, r) in recapture_rates 
        if orig == p and (p, r) in recap
    )
    
    m.addConstr(t[p] + s[p] + total_recaptured_out == demand[p], name=f"Demand_{p}")

# 2. Capacity Constraints
# For each flight l, sum of (original pax using l) + (recaptured pax using l) <= Capacity
for l in L:
    # A. Direct passengers using this leg
    # (Iterate only itineraries that actually use leg l)
    direct_pax = quicksum(t[p] for p in P if uses_leg(p, l))
    
    # B. Recaptured passengers using this leg
    # These are people moving FROM some p TO an itinerary r that uses leg l
    # We need to sum recap[p, r] where 'r' uses leg 'l'
    recaptured_pax = quicksum(
        recap[p, r] 
        for (p, r) in recapture_rates 
        if uses_leg(r, l) and (p, r) in recap
    )
    
    m.addConstr(direct_pax + recaptured_pax <= capacity[l], name=f"Cap_{l}")
    
for (p, r), rate in recapture_rates.items():
    if (p, r) in recap:
        # Constraint: Recaptured pax <= Recapture Rate * Spilled Pax
        m.addConstr(recap[p, r] <= rate * s[p], name=f"RateLimit_{p}_{r}")

# 3. Non-negativity is handled by lb=0 in variable creation.

# =============================================================================
# 3. SOLVE AND REPORT
# =============================================================================

m.optimize()

if m.status == GRB.OPTIMAL:
    print("\n" + "="*50)
    print("OPTIMAL SOLUTION FOUND")
    print("="*50)
    
    # 1. Optimal Airline Cost (Objective Value)
    print(f"Optimal Spill Cost (Lost Revenue): €{m.ObjVal:,.2f}")
    
    # Calculate Total Potential Revenue (if we carried everyone perfectly)
    total_potential_revenue = sum(demand[p] * fare[p] for p in P)
    realized_revenue = total_potential_revenue - m.ObjVal
    print(f"Realized Revenue: €{realized_revenue:,.2f}")

    # 2. Total Number of Passengers Spilled
    # Sum of all s[p] variables
    total_spilled = sum(s[p].X for p in P)
    print(f"Total Passengers Spilled: {int(total_spilled)}")
    
    # 3. Optimal Decision Variables for First 5 Itineraries
    print("\n--- First 5 Itineraries ---")
    for i, p in enumerate(P[:5]):
        print(f"Itinerary {p}:")
        print(f"  Demand: {demand[p]}")
        print(f"  Transported (Original): {t[p].X:.1f}")
        print(f"  Spilled: {s[p].X:.1f}")
        
        # Check if any recapture happened from this itinerary
        any_recap = False
        for (orig, target), var in recap.items():
            if orig == p and var.X > 0.1:
                print(f"  -> Recaptured to Itinerary {target}: {var.X:.1f}")
                any_recap = True
        if not any_recap:
            print("  -> No Recapture")
            
    # 4. Optimal Dual Variables (Shadow Prices) for First 5 Flights
    print("\n--- Dual Variables (First 5 Flights) ---")
    for i, l in enumerate(L[:5]):
        try:
            constr = m.getConstrByName(f"Cap_{l}")
            print(f"Flight {l} | Capacity: {capacity[l]} | Shadow Price (Pi): {constr.Pi:.2f}")
        except:
            print(f"Flight {l} | Capacity: {capacity[l]} | Shadow Price (Pi): N/A (Constraint not found)")

else:
    print("Model did not solve to optimality.")