#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 16:43:02 2025

@author: jimvanerp
"""

import pandas as pd
from gurobipy import Model, GRB, quicksum
import sys
import os

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("Loading data files...")

try:
    df_flights = pd.read_excel('Group_12.xlsx', sheet_name=0)
    df_itineraries = pd.read_excel('Group_12.xlsx', sheet_name=1)
    df_recapture = pd.read_excel('Group_12.xlsx', sheet_name=2)
except FileNotFoundError:
    print("Error: Excel file not found.")
    sys.exit()

# --- CLEANING & PREPARATION ---

# 1. Sets
L = df_flights['Flight No.'].tolist()  # Set of Flight Legs
P = df_itineraries['Itinerary'].tolist() # Set of Itineraries

# 2. Parameters
capacity = df_flights.set_index('Flight No.')['Capacity'].to_dict()
demand = df_itineraries.set_index('Itinerary')['Demand'].to_dict()
fare = df_itineraries.set_index('Itinerary')['Price [EUR]'].to_dict()

# Recapture Rates
recapture_rates = {}
for index, row in df_recapture.iterrows():
    p = row['From Itinerary']
    r = row['To Itinerary']
    rate = row['Recapture Rate']
    recapture_rates[(p, r)] = rate

# 3. Incidence Matrix & Q_l Calculation
itin_flights = {}
Q_l = {l: 0 for l in L} # Initialize Unconstrained Leg Demand

for index, row in df_itineraries.iterrows():
    p = row['Itinerary']
    d = row['Demand']
    legs = []
    
    # Check Flight 1
    f1 = row['Flight 1']
    if pd.notna(f1) and str(f1).strip() != '':
        legs.append(f1)
        if f1 in Q_l: Q_l[f1] += d
        
    # Check Flight 2
    f2 = row['Flight 2']
    if pd.notna(f2) and str(f2).strip() != '':
        legs.append(f2)
        if f2 in Q_l: Q_l[f2] += d
        
    itin_flights[p] = legs

print(f"Calculated Q_l (Unconstrained Demand) for {len(L)} legs.")

# Helper
def uses_leg(p, l):
    return l in itin_flights.get(p, [])

# =============================================================================
# 2. GUROBI MODEL SETUP
# =============================================================================

m = Model('Passenger_Mix_Flow')

# --- Decision Variables ---
t = {} # Transported
s = {} # Spilled
recap = {} # Recaptured

for p in P:
    t[p] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Trans_{p}")
    s[p] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Spill_{p}")
    
    for (origin_p, target_r), rate in recapture_rates.items():
        if origin_p == p:
            recap[p, target_r] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Recap_{p}_{target_r}")

m.update()

# --- Objective Function (Minimize Spill Costs) ---
spill_cost = quicksum(fare[p] * s[p] for p in P)

recap_cost = quicksum(
    (fare[p] - rate * fare[target_r]) * recap[p, target_r]
    for (p, target_r), rate in recapture_rates.items()
    if (p, target_r) in recap
)

m.setObjective(spill_cost + recap_cost, GRB.MINIMIZE)

# --- Constraints ---

# 1. Demand Conservation
for p in P:
    total_recaptured_out = quicksum(
        recap[p, r] 
        for (orig, r) in recapture_rates 
        if orig == p and (p, r) in recap
    )
    m.addConstr(t[p] + s[p] + total_recaptured_out == demand[p], name=f"Demand_{p}")

# 2. Capacity Constraints (Using Q_l formulation)
# Formula: Sum(Spilled from p on leg l) >= Q_l - Capacity_l + Sum(Recaptured away from l) - Sum(Recaptured TO l)
# Actually, the slide formula ">= Qi - CAPi" usually simplifies to:
# Total Load <= Capacity
# But let's rewrite it in terms of spills if that's the professor's format.
# Standard Identity: Transported_p = Demand_p - Spilled_p - Recaptured_Out_p
# Constraint: Sum(Transported_p) + Sum(Recaptured_In_p) <= Capacity_l
# Substitute Transported:
# Sum(Demand_p - Spilled_p - Recaptured_Out_p) + Sum(Recaptured_In_p) <= Capacity_l
# Sum(Demand_p) - Sum(Spilled_p) - Sum(Recaptured_Out_p) + Sum(Recaptured_In_p) <= Capacity_l
# Q_l - Sum(Spilled_p) - Sum(Recaptured_Out_p) + Sum(Recaptured_In_p) <= Capacity_l
# Move variables to LHS, constants to RHS:
# Sum(Spilled_p) + Sum(Recaptured_Out_p) - Sum(Recaptured_In_p) >= Q_l - Capacity_l

for l in L:
    # 1. Sum of Spilled Passengers for itineraries using leg l
    sum_spilled = quicksum(s[p] for p in P if uses_leg(p, l))
    
    # 2. Sum of Recaptured OUT (Passengers who WANTED leg l but moved elsewhere)
    # They effectively "free up" a seat on leg l by leaving.
    # Logic: If p uses leg l, and we move them to r, they are no longer on l (unless r ALSO uses l).
    # We sum recap[p, r] for all p that use l.
    sum_recap_out = quicksum(
        recap[p, r]
        for p in P if uses_leg(p, l)
        for (orig, r) in recapture_rates if orig == p and (p, r) in recap
    )
    
    # 3. Sum of Recaptured IN (Passengers who didn't want leg l but were moved TO it)
    # These effectively "take up" a seat on leg l.
    sum_recap_in = quicksum(
        recap[p, r] 
        for (p, r) in recapture_rates 
        if uses_leg(r, l) and (p, r) in recap
    )
    
    # Combined Constraint:
    # (Spill + Moved Away) - (Moved In) >= Excess Demand
    # Note: If a passenger moves from p (using l) to r (using l), they appear in both OUT and IN sums, canceling out. Correct.
    
    m.addConstr(
        sum_spilled + sum_recap_out - sum_recap_in >= Q_l[l] - capacity[l],
        name=f"Cap_Qi_{l}"
    )


# =============================================================================
# 3. SOLVE AND REPORT
# =============================================================================

m.optimize()

if m.status == GRB.OPTIMAL:
    print("\n" + "="*50)
    print("OPTIMAL SOLUTION FOUND (Qi Formulation)")
    print("="*50)
    
    print(f"Optimal Spill Cost (Lost Revenue): €{m.ObjVal:,.2f}")
    
    total_potential_revenue = sum(demand[p] * fare[p] for p in P)
    realized_revenue = total_potential_revenue - m.ObjVal
    print(f"Realized Revenue: €{realized_revenue:,.2f}")

    total_spilled = sum(s[p].X for p in P)
    print(f"Total Passengers Spilled: {int(total_spilled)}")
    
else:
    print("Model did not solve to optimality.")