#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 17:23:17 2025

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

print(f"Calculated Q_l parameters. Max Leg Demand: {max(Q_l.values())}")

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

mp = Model("RMP_Qi_ColGen")
mp.setParam('OutputFlag', 0)

# -- Constraints --

# 1. Demand Balance (Dual: Sigma)
# t_p + s_p + sum(t_pr) = D_p
con_demand = {}
for p in demand_data:
    con_demand[p] = mp.addConstr(LinExpr() == demand_data[p], name=f"Dem_{p}")

# 2. Capacity Constraints (Qi Formulation)
# Sum(Spill_p on l) + Sum(Recap_OUT_p on l) - Sum(Recap_IN_p on l) >= Q_l - Cap_l
# Note: Gurobi standard form is usually <= or ==. >= is fine.
# Dual: Mu (Expected to be positive if constraint is >= ?)
con_capacity_qi = {}
for l in L:
    rhs = Q_l[l] - capacity[l]
    # We create constraint: LHS >= RHS
    con_capacity_qi[l] = mp.addConstr(LinExpr() >= rhs, name=f"CapQi_{l}")

# 3. Recapture Limit Constraints (Dual: Gamma)
# t_pr - b_pr * s_p <= 0
con_recap_limit = {}
for opt in recapture_options:
    p = opt['orig']
    r = opt['recap']
    con_recap_limit[(p, r)] = mp.addConstr(LinExpr() <= 0, name=f"Lim_{p}_{r}")

# -- Initialize Variables --
variables = {} 

# A. Spill Variables (s_p)
# Objective: Fare_p
# Demand Constraint: +1
# Capacity (Qi) Constraint: +1 for every leg l used by p (Spill contributes to "Sum Spill")
# Limit Constraint: -rate for every recapture option from p
for p in demand_data:
    col = Column()
    col.addTerms(1.0, con_demand[p])
    
    # Add to Capacity Constraints (Qi form: Spill counts as +1)
    for l in itin_legs.get(p, []):
        if l in con_capacity_qi:
            col.addTerms(1.0, con_capacity_qi[l])
            
    # Add to Limit Constraints (-rate * s_p)
    for opt in recapture_options:
        if opt['orig'] == p:
            r = opt['recap']
            rate = opt['rate']
            col.addTerms(-rate, con_recap_limit[(p, r)])
            
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
    
    # Mu (Capacity Qi)
    mu = {l: con_capacity_qi[l].Pi for l in L}
    
    # Gamma (Limits)
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
            
        # --- Reduced Cost Calculation ---
        # Variable: t_pr (Recaptured Pax)
        # Objective Coeff: Cost_pr = Fare_orig - rate * Fare_recap (Spill Cost Minimization)
        # Constraints coefficients:
        # 1. Demand (orig): +1
        # 2. Limit (orig, recap): +1
        # 3. Capacity (Qi) Constraints:
        #    - It is an "Outbound" flow from legs in 'orig' -> Coeff +1
        #    - It is an "Inbound" flow to legs in 'recap' -> Coeff -1
        
        cost_coeff = fare_data[orig] - (rate * fare_data.get(recap, 0))
        
        # Dual terms sum
        dual_sum = sigma[orig] + gamma[(orig, recap)]
        
        # Add capacity duals
        # +1 * Mu_l for l in ORIG path
        for l in itin_legs.get(orig, []):
            if l in mu:
                dual_sum += 1.0 * mu[l]
                
        # -1 * Mu_l for l in RECAP path
        for l in itin_legs.get(recap, []):
            if l in mu:
                dual_sum += -1.0 * mu[l]
        
        reduced_cost = cost_coeff - dual_sum
        
        if reduced_cost < -0.0001:
            col = Column()
            col.addTerms(1.0, con_demand[orig])
            col.addTerms(1.0, con_recap_limit[(orig, recap)])
            
            # Capacity Qi terms
            for l in itin_legs.get(orig, []):
                if l in con_capacity_qi:
                    col.addTerms(1.0, con_capacity_qi[l]) # OUT (+1)
            
            for l in itin_legs.get(recap, []):
                if l in con_capacity_qi:
                    col.addTerms(-1.0, con_capacity_qi[l]) # IN (-1)
            
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
print("COLUMN GENERATION RESULTS (Qi Formulation)")
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
    
else:
    print(f"Model Failed. Status: {mp.status}")