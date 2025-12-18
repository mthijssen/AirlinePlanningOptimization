# =========================
# AE4423 – Assignment 2
# Passenger Mix with Recapture
# =========================

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time

# =========================
# 1. Load data
# =========================

file_path = "Group_12.xlsx"

flights = pd.read_excel(file_path, sheet_name="Flights")
itins = pd.read_excel(file_path, sheet_name="Itineraries")
recap = pd.read_excel(file_path, sheet_name="Recapture")

# Sanity checks
assert "Flight No." in flights.columns
assert "Itinerary" in itins.columns
assert "From Itinerary" in recap.columns

# =========================
# 2. Sets
# =========================

P = itins["Itinerary"].tolist()           # itineraries
L = flights["Flight No."].tolist()         # flights (legs)

# =========================
# 3. Parameters
# =========================

# Demand and fares
demand = dict(zip(itins["Itinerary"], itins["Demand"]))
fare = dict(zip(itins["Itinerary"], itins["Price [EUR]"]))

# Flight capacities
capacity = dict(zip(flights["Flight No."], flights["Capacity"]))

# Delta[p,i] = 1 if itinerary p uses flight i
delta = {(p, i): 0 for p in P for i in L}

for _, row in itins.iterrows():
    p = row["Itinerary"]
    if not pd.isna(row["Flight 1"]):
        delta[p, row["Flight 1"]] = 1
    if not pd.isna(row["Flight 2"]):
        delta[p, row["Flight 2"]] = 1

# Recapture rates alpha[p,r]
alpha = {
    (row["From Itinerary"], row["To Itinerary"]): row["Recapture Rate"]
    for _, row in recap.iterrows()
}

# =========================
# 4. Build model
# =========================

model = gp.Model("Passenger_Mix_with_Recapture")
model.Params.OutputFlag = 1

start_time = time.time()

# =========================
# 5. Decision variables
# =========================

y = model.addVars(P, lb=0, vtype=GRB.INTEGER, name="y")
s = model.addVars(P, lb=0, vtype=GRB.INTEGER, name="spill")


# =========================
# 6. Objective: Max revenue
# =========================

model.setObjective(
    gp.quicksum(fare[p] * y[p] for p in P),
    GRB.MAXIMIZE
)

# =========================
# 7. Capacity constraints
# =========================

for i in L:
    model.addConstr(
        gp.quicksum(delta[p, i] * y[p] for p in P) <= capacity[i],
        name=f"capacity_{i}"
    )

# =========================
# 8. Demand + recapture constraints
# =========================

# y_p ≤ D_p + recaptured passengers
for p in P:
    model.addConstr(
        y[p] <= demand[p] + gp.quicksum(
            alpha[q, p] * s[q]
            for q in P
            if (q, p) in alpha
        ),
        name=f"demand_recapture_{p}"
    )

# =========================
# 9. Spill definition
# =========================

# Spill only if demand exceeds carried pax
for p in P:
    model.addConstr(
        s[p] >= demand[p] - y[p],
        name=f"spill_def_{p}"
    )

# =========================
# 10. Solve
# =========================

model.optimize()

runtime = time.time() - start_time

# =========================
# 11. Results
# =========================

print("\n================ RESULTS ================")
print(f"Optimal revenue: {model.ObjVal:.2f}")
print(f"Total spill: {sum(s[p].X for p in P):.2f}")
print(f"Runtime: {runtime:.2f} seconds")

print("\nFirst 5 itinerary decisions (y_p):")
for p in P[:5]:
    print(f"Itinerary {p}: {y[p].X:.2f}")

print("\nFirst 5 flight dual variables:")
for i in L[:5]:
    print(f"Flight {i}: Dual = {model.getConstrByName(f'capacity_{i}').Pi:.2f}")

print("\nSample recapture flows:")
for (q, p), a in alpha.items():
    if s[q].X > 1e-6 and a > 0:
        print(f"Spill from {q} → {p}: {a * s[q].X:.2f}")
