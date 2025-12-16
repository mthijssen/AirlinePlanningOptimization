#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 14:49:01 2025

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
# 2. MASTER PROBLEM (RMP) SETUP
# =============================================================================

mp = Model("RMP_SpillMin_Strict")
mp.setParam('OutputFlag', 0)

# -- Constraints --

# 1. Demand Balance (Dual: Sigma)
con_demand = {}
for p in demand_data:
    con_demand[p] = mp.addConstr(LinExpr() == demand_data[p], name=f"Dem_{p}")

# 2. Capacity Constraints (Dual: Pi)
con_capacity = {}
for l in L:
    con_capacity[l] = mp.addConstr(LinExpr() <= capacity[l], name=f"Cap_{l}")

# 3. Recapture Limit Constraints (Dual: Gamma)
# t_pr <= b_pr * s_p  =>  t_pr - b_pr * s_p <= 0
# We pre-add these constraints for ALL potential recapture pairs.
# Most will be inactive (slack) if the column isn't generated yet.
con_recap_limit = {}
for opt in recapture_options:
    p = opt['orig']
    r = opt['recap']
    # Constraint: 0 <= 0 initially. Variables will add terms to LHS.
    con_recap_limit[(p, r)] = mp.addConstr(LinExpr() <= 0, name=f"Lim_{p}_{r}")

# -- Initialize Variables --
variables = {} 

# A. Spill Variables (s_p)
# Cost = Fare_p
# Contributes +1 to Demand Balance of p
# Contributes -rate to Recapture Limit of EVERY recapture option starting at p
for p in demand_data:
    col = Column()
    col.addTerms(1.0, con_demand[p])
    
    # Add terms for Recapture Limits: -rate * s_p
    # Find all recapture options starting at p
    for opt in recapture_options:
        if opt['orig'] == p:
            r = opt['recap']
            rate = opt['rate']
            # Constraint is: t_pr - rate * s_p <= 0
            # So s_p has coefficient -rate
            col.addTerms(-rate, con_recap_limit[(p, r)])
            
    var = mp.addVar(obj=fare_data[p], vtype=GRB.CONTINUOUS, column=col, name=f"Spill_{p}")
    variables[f"spill_{p}"] = var

# B. Original Transport Variables (t_p)
# Cost = 0
for p in demand_data:
    col = Column()
    col.addTerms(1.0, con_demand[p])
    
    valid_legs = [l for l in itin_legs.get(p, []) if l in con_capacity]
    for l in valid_legs:
        col.addTerms(1.0, con_capacity[l])
    
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
    pi = {l: con_capacity[l].Pi for l in L}
    sigma = {p: con_demand[p].Pi for p in demand_data}
    
    # Gamma (Dual for Recapture Limits)
    # Most will be 0 if the constraint is not binding
    gamma = {} 
    for key, constr in con_recap_limit.items():
        gamma[key] = constr.Pi
    
    cols_added_this_iter = 0
    
    for opt in recapture_options:
        orig = opt['orig']
        recap = opt['recap']
        rate = opt['rate']
        
        if orig not in demand_data: continue

        var_name = f"Recap_{orig}_{recap}"
        if mp.getVarByName(var_name) is not None:
            continue
            
        # Cost Logic (Same as before: Spill Cost Minimization)
        # Cost = Fare_p - b * Fare_r (Effective loss per recaptured pax)
        # WAIT! If we enforce the physical limit, does the cost change?
        # The Slide 13 objective was: sum (Fare_p - b * Fare_r) * t_pr.
        # This implies t_pr is "effective passengers" or "attempted passengers"?
        # Usually, if t_pr is limited by b * s_p, then t_pr represents SUCCESSFUL passengers.
        # If t_pr is successful, revenue is Fare_r. Cost = Fare_p - Fare_r.
        #
        # BUT, if your manual "Constraint" run used the cost (Fare_p - b*Fare_r) AND the constraint t < b*s,
        # then we must use that cost here too.
        
        cost_coeff = fare_data[orig] - (rate * fare_data.get(recap, 0))
        
        # Duals
        # 1. Capacity: sum(pi)
        # 2. Demand: sigma[orig] (Does it consume demand? Yes, t_pr is part of D_p)
        # 3. Limit: gamma[(orig, recap)] (Coefficient of t_pr in limit constraint is +1)
        
        recap_legs = itin_legs.get(recap, [])
        valid_recap_legs = [l for l in recap_legs if l in pi]
        
        dual_cap = sum(pi[l] for l in valid_recap_legs)
        dual_dem = sigma[orig]
        dual_lim = gamma[(orig, recap)] # New dual!
        
        reduced_cost = cost_coeff - (dual_dem + dual_cap + dual_lim)
        
        if reduced_cost < -0.0001:
            col = Column()
            col.addTerms(1.0, con_demand[orig])
            for l in valid_recap_legs:
                col.addTerms(1.0, con_capacity[l])
            
            # Recapture Limit Constraint: t_pr has coeff +1
            # t_pr - b*s_p <= 0
            col.addTerms(1.0, con_recap_limit[(orig, recap)])
            
            var = mp.addVar(obj=cost_coeff, vtype=GRB.CONTINUOUS, column=col, name=var_name)
            variables[var_name] = var
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
print("COLUMN GENERATION RESULTS (STRICT)")
print("="*50)

if mp.status == GRB.OPTIMAL:
    print(f"Total Runtime: {runtime:.4f} seconds")
    print(f"Iterations: {iteration}")
    print(f"Columns Initial: {initial_cols}")
    print(f"Columns Added: {cols_added_total}")
    print(f"Total Columns Final: {mp.NumVars}")
    print(f"Optimal Objective (Spill Cost): €{mp.ObjVal:,.2f}")

    total_spilled = sum(variables[f"spill_{p}"].X for p in demand_data)
    print(f"Total Passengers Spilled: {int(total_spilled)}")

    total_recap = 0
    for v in mp.getVars():
        if v.VarName.startswith("Recap_"):
            total_recap += v.X
    print(f"Total Passengers Recaptured: {int(total_recap)}")
    
    print("\nCheck: Does this match your constrained model (€1,845,732.02)?")
else:
    print(f"Model Infeasible. Status: {mp.status}")