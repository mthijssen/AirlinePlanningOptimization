import time
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

flights = pd.read_excel("Group_12.xlsx", sheet_name=0)
itins = pd.read_excel("Group_12.xlsx", sheet_name=1)
recap = pd.read_excel("Group_12.xlsx", sheet_name=2)

# Basic checks
assert "Flight No." in flights.columns
assert "Itinerary" in itins.columns or "Itinerary" in itins.columns
# Normalize column names (strip whitespace)
flights.columns = flights.columns.str.strip()
itins.columns   = itins.columns.str.strip()
recap.columns   = recap.columns.str.strip()

# ---------- Build sets and parameters ----------
# legs: identify each flight by its Flight No.
legs = flights["Flight No."].astype(str).tolist()
leg_cap = flights.set_index("Flight No.")["Capacity"].to_dict()

# itineraries: use index or the Itinerary col
# ensure itinerary id is string/unique index
if "Itinerary" in itins.columns:
    itins = itins.copy()
    itins["ItineraryID"] = itins["Itinerary"].astype(str)
else:
    itins = itins.copy()
    itins["ItineraryID"] = itins.index.astype(str)

itins = itins.set_index("ItineraryID", drop=False)
P = list(itins.index)

# parameters from itineraries
fare = itins["Price [EUR]"].to_dict()
demand = itins["Demand"].to_dict()

# Build delta_{p,ell} (does itinerary p use leg ell?)
# Itineraries may have Flight 1 and possibly Flight 2
delta = {}
for p in P:
    f1 = itins.at[p, "Flight 1"]
    f2 = itins.at[p, "Flight 2"] if "Flight 2" in itins.columns else None
    # Normalize to strings if available
    used = []
    if pd.notna(f1):
        used.append(str(f1))
    if pd.notna(f2):
        used.append(str(f2))
    for ell in legs:
        delta[(p, ell)] = 1 if ell in used else 0

# ---------- Build the Gurobi model ----------
model = gp.Model("PassengerMix_LP")
model.setParam("OutputFlag", 1)   # set to 0 to mute solver output

# Decision vars: y_p (accepted pax)
y = model.addVars(P, lb=0.0, name="y", vtype=GRB.CONTINUOUS)

# Set UB for each itinerary according to its demand
for p in P:
    dp = 0.0 if pd.isna(demand[p]) else float(demand[p])
    y[p].UB = dp


# Objective: maximize sum fare_p * y_p
obj = gp.quicksum((0.0 if pd.isna(fare[p]) else float(fare[p])) * y[p] for p in P)
model.setObjective(obj, GRB.MAXIMIZE)

# Capacity constraints
cap_cons = {}
for ell in legs:
    cap_cons[ell] = model.addLConstr(
        gp.quicksum(delta[(p, ell)] * y[p] for p in P),
        GRB.LESS_EQUAL,
        leg_cap[ell],
        name=f"cap_{ell}"
    )

# Solve
start = time.time()
model.optimize()
runtime = time.time() - start

# ---------- Results ----------
status = model.Status
if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT or status == GRB.SUBOPTIMAL:
    total_revenue = model.objVal
    # compute spilled
    spilled = {p: (float(demand[p]) - y[p].X) if pd.notna(demand[p]) else 0.0 for p in P}
    total_spilled = sum(max(0.0, s) for s in spilled.values())

    print(f"\nSolver status: {model.Status}")
    print(f"Optimal revenue (objective): {total_revenue:.2f} EUR")
    print(f"Total spilled passengers: {total_spilled:.2f}")

    # First 5 itineraries decisions
    print("\nFirst 5 itineraries y_p (accepted passengers):")
    for p in P[:5]:
        print(p, "-> accepted:", y[p].X, "demand:", demand[p], "spilled:", spilled[p])

    # Duals for the first 5 flights (capacity constraints)
    print("\nDuals (shadow prices) for first 5 flights:")
    for ell in legs[:5]:
        con = cap_cons[ell]
        print(ell, "dual:", con.Pi)

    # Number of columns in RMP (here the RMP is full formulation: number of y vars)
    n_columns_before = len(P)
    n_columns_after = len(P)   # if you don't do column generation, same
    iterations = 1  # single solve
    print("\nColumns in RMP before running CG (initial):", n_columns_before)
    print("Columns in RMP after running CG (final):", n_columns_after)
    print("Number of iterations (if no CG):", iterations)
    print(f"Runtime: {runtime:.3f} s")

else:
    print("Model did not solve to optimality. Status:", status)
