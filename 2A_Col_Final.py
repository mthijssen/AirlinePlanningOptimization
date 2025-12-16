#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 17:29:23 2025

@author: jimvanerp
"""

import pandas as pd
from gurobipy import Model, GRB, Column, quicksum, LinExpr
import sys
import os
import time

# =============================================================================
# 1. DATA LOADING
# =============================================================================

# Define filenames
# Assuming files are local as per user context
try:
    df_flights = pd.read_excel('Group_12.xlsx', sheet_name=0)
    df_itineraries = pd.read_excel('Group_12.xlsx', sheet_name=1)
    df_recapture = pd.read_excel('Group_12.xlsx', sheet_name=2)
except FileNotFoundError:
    print("Error: Excel file not found.")
    sys.exit()

# --- Data Structures ---
L = df_flights['Flight No.'].tolist()
capacity = df_flights.set_index('Flight No.')['Capacity'].to_dict()
demand_data = df_itineraries.set_index('Itinerary')['Demand'].to_dict()
fare_data = df_itineraries.set_index('Itinerary')['Price [EUR]'].to_dict()

# Map Itinerary -> Legs
itin_legs = {}
# Also calculate Q_l (Unconstrained Demand per Leg)
Q_l = {l: 0 for l in L}

for idx, row in df_itineraries.iterrows():
    p = row['Itinerary']
    d = demand_data[p]
    legs = []
    
    if pd.notna(row['Flight 1']): 
        f1 = row['Flight 1']
        legs.append(f1)
        if f1 in Q_l: Q_l[f1] += d
        
    if pd.notna(row['Flight 2']): 
        f2 = row['Flight 2']
        legs.append(f2)
        if f2 in Q_l: Q_l[f2] += d
        
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
# 2. MASTER PROBLEM (RMP) SETUP
# =============================================================================

mp = Model("RMP_Qi_ColGen_Relaxed")
mp.setParam('OutputFlag', 0)

# -- Constraints --

# 1. Demand Balance (Dual: Sigma)
# t_p + s_p + sum(t_pr) = D_p
con_demand = {}
for p in demand_data:
    con_demand[p] = mp.addConstr(LinExpr() == demand_data[p], name=f"Dem_{p}")

# 2. Capacity Constraints (Qi Formulation)
# Sum(Spill_p on l) + Sum(Recap_OUT_p on l) - Sum(Recap_IN_p on l) >= Q_l - Cap_l
con_capacity_qi = {}
for l in L:
    rhs = Q_l[l] - capacity[l]
    # We create constraint: LHS >= RHS
    con_capacity_qi[l] = mp.addConstr(LinExpr() >= rhs, name=f"CapQi_{l}")

# -- Initialize Variables --
variables = {} 
recap_vars = {} # Store recapture vars separately for easy access

# A. Spill Variables (s_p)
# Objective: Fare_p
# Demand Constraint: +1
# Capacity (Qi) Constraint: +1 for every leg l used by p (Spill contributes to "Sum Spill")
for p in demand_data:
    col = Column()
    col.addTerms(1.0, con_demand[p])
    
    # Add to Capacity Constraints (Qi form: Spill counts as +1)
    for l in itin_legs.get(p, []):
        if l in con_capacity_qi:
            col.addTerms(1.0, con_capacity_qi[l])
            
    var = mp.addVar(obj=fare_data[p], vtype=GRB.CONTINUOUS, column=col, name=f"Spill_{p}")
    variables[f"spill_{p}"] = var

# B. Original Transport Variables (t_p)
# Objective: 0
# Demand Constraint: +1
# Capacity (Qi) Constraint: 0 contribution?
# Logic: t_p = D_p - s_p - t_pr.
# The Qi constraint is derived by substituting t_p out. 
# So t_p does NOT appear in the Qi constraint.
for p in demand_data:
    col = Column()
    col.addTerms(1.0, con_demand[p])
    # No terms in Capacity or Limit constraints for t_p in this formulation!
    var = mp.addVar(obj=0.0, vtype=GRB.CONTINUOUS, column=col, name=f"Trans_{p}")
    variables[f"trans_{p}"] = var

mp.modelSense = GRB.MINIMIZE
mp.update()

initial_cols = mp.NumVars
print(f"RMP Initialized with {initial_cols} columns.")

# =============================================================================
# 3. COLUMN GENERATION LOOP
# =============================================================================

start_time = time.time()
iteration = 0
cols_added_total = 0

while True:
    iteration += 1
    mp.optimize()
    
    if mp.status != GRB.OPTIMAL:
        print(f"RMP Status not OPTIMAL: {mp.status}")
        break
        
    # --- Get Dual Variables ---
    # Sigma (Demand)
    sigma = {p: con_demand[p].Pi for p in demand_data}
    
    # Pi (Capacity Qi) - Renamed from 'mu' to 'pi' 
    pi = {l: con_capacity_qi[l].Pi for l in L}
    
    cols_added_this_iter = 0
    
    for opt in recapture_options:
        orig = opt['orig']
        recap = opt['recap']
        rate = opt['rate']
        
        if orig not in demand_data: continue

        var_name = f"Recap_{orig}_{recap}"
        if mp.getVarByName(var_name) is not None:
            continue
            
        # --- Reduced Cost Calculation ---
        # Variable: t_pr (Recaptured Pax)
        # Objective Coeff: Cost_pr = Fare_orig - rate * Fare_recap (Spill Cost Minimization)
        cost_coeff = fare_data[orig] - (rate * fare_data.get(recap, 0))
        dual_sum = sigma[orig]
        
        # Add capacity duals (Qi Formulation Logic)
        # Recaptured pax means:
        # 1. They leave ORIG legs (so they count as "Recap OUT"). Coeff +1 in capacity constraint.
        # 2. They enter RECAP legs (so they count as "Recap IN"). Coeff -1 in capacity constraint.
        for l in itin_legs.get(orig, []):
            if l in pi:
                dual_sum += 1.0 * pi[l]
                
        for l in itin_legs.get(recap, []):
            if l in pi:
                dual_sum += -1.0 * pi[l]
        
        reduced_cost = cost_coeff - dual_sum
        
        if reduced_cost < -0.0001:
            col = Column()
            col.addTerms(1.0, con_demand[orig])
            
            # Capacity Qi terms
            for l in itin_legs.get(orig, []):
                if l in con_capacity_qi:
                    col.addTerms(1.0, con_capacity_qi[l]) # OUT (+1)
            
            for l in itin_legs.get(recap, []):
                if l in con_capacity_qi:
                    col.addTerms(-1.0, con_capacity_qi[l]) # IN (-1)
            
            var = mp.addVar(obj=cost_coeff, vtype=GRB.CONTINUOUS, column=col, name=var_name)
            variables[var_name] = var
            recap_vars[(orig, recap)] = var
            cols_added_this_iter += 1
            cols_added_total += 1
            
    if cols_added_this_iter == 0:
        break 

runtime = time.time() - start_time

# =============================================================================
# 4. REPORTING
# =============================================================================

mp.optimize()

print("\n" + "="*50)
print("COLUMN GENERATION RESULTS (Part 1 - Qi Relaxed)")
print("="*50)

if mp.status == GRB.OPTIMAL:
    # 1. Performance Stats
    print(f"Total Runtime: {runtime:.4f} seconds")
    print(f"Iterations to Converge: {iteration}")
    print(f"Columns in RMP (Before): {initial_cols}")
    print(f"Columns in RMP (After): {mp.NumVars}")
    print(f"New Columns Generated: {cols_added_total}")
    
    # 2. Optimal Airline Cost
    print(f"\nOptimal Spill Cost (Lost Revenue): €{mp.ObjVal:,.2f}")

    # 3. Total Passengers Spilled
    # Make sure we sum the CURRENT values of the variables
    total_spilled = sum(variables[f"spill_{p}"].X for p in demand_data)
    print(f"Total Passengers Spilled: {int(total_spilled)}")

    # 4. Optimal Decision Variables (First 5 Itineraries)
    print("\n--- Decision Variables (First 5 Itineraries) ---")
    for i, p in enumerate(list(demand_data.keys())[:5]):
        t_val = variables[f"trans_{p}"].X
        s_val = variables[f"spill_{p}"].X
        print(f"Itinerary {p}:")
        print(f"  Demand: {demand_data[p]}")
        print(f"  Transported (Original): {t_val:.1f}")
        print(f"  Spilled: {s_val:.1f}")
        
        # Check recapture
        found_recap = False
        for (orig, target), var in recap_vars.items():
            if orig == p and var.X > 0.1:
                print(f"  -> Recaptured to {target}: {var.X:.1f}")
                found_recap = True
        if not found_recap:
            print("  -> No Recapture")

    # 5. Optimal Dual Variables (Pi and Sigma)
    print("\n--- Dual Variables: Capacity Constraints (Pi) for First 5 Flights ---")
    print(f"{'Flight':<10} | {'Capacity':<10} | {'Shadow Price (Pi)':<20}")
    print("-" * 50)
    
    count = 0
    for l in L:
        if count >= 5: break
        constr = mp.getConstrByName(f"CapQi_{l}")
        if constr:
            print(f"{l:<10} | {capacity[l]:<10} | {constr.Pi:.2f}")
            count += 1

    print("\n--- Dual Variables: Demand Constraints (Sigma) for First 5 Itineraries ---")
    print(f"{'Itinerary':<10} | {'Demand':<10} | {'Shadow Price (Sigma)':<20}")
    print("-" * 50)
    
    for i, p in enumerate(list(demand_data.keys())[:5]):
        constr = mp.getConstrByName(f"Dem_{p}")
        if constr:
            print(f"{p:<10} | {demand_data[p]:<10} | {constr.Pi:.2f}")
            
else:
    print(f"Model Failed. Status: {mp.status}")