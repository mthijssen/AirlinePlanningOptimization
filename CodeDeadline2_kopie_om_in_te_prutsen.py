#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 21:11:48 2026

@author: jimvanerp
"""

#Assumption: An exchange rate of 1 EUR = 1 USD was used as no rate was provided.

import pandas as pd
import numpy as np
import math
from dataclasses import dataclass

# --- CONFIGURATION ---
TIME_STEP_MIN = 6
STEPS_PER_HOUR = 60 // TIME_STEP_MIN
TOTAL_STEPS = 24 * STEPS_PER_HOUR  # 240 steps (0 to 240)
FUEL_COST_PER_GALLON = 1.42
FUEL_FACTOR = 1.5
DEMAND_FACTOR = 1.0

# --- DATA STRUCTURES ---

@dataclass
class Airport:
    id: str
    runway: int
    dist_from_hub: float = 0.0
    lat: float = 0.0
    lon: float = 0.0

@dataclass
class AircraftType:
    name: str
    speed: float     # km/h
    seats: int
    tat: int         # minutes
    max_range: float # km
    runway_req: float # m
    lease_cost: float # EUR/day
    fixed_cost: float # EUR/leg
    time_cost_param: float # EUR/hour
    fuel_cost_param: float # param
    count: int       # Number of aircraft available

    def calc_flight_time_min(self, dist):
        # Time = (Dist / Speed) * 60 + 30 (includes taxi)
        return (dist / self.speed) * 60 + 30

    def calc_steps(self, dist):
        flight_time = self.calc_flight_time_min(dist)
        total_time = flight_time + self.tat
        return math.ceil(total_time / TIME_STEP_MIN)

    def calc_trip_cost(self, dist):
        # Time Cost
        c_t = self.time_cost_param * (dist / self.speed) #Moet hier niet de 30 min + TaT ook bij?
        # Fuel Cost
        c_f = (self.fuel_cost_param * FUEL_COST_PER_GALLON / FUEL_FACTOR) * dist
        # Total = Fixed + Time + Fuel
        return self.fixed_cost + c_t + c_f

# --- HELPER FUNCTIONS ---

def calculate_yield(distance):
    # Formula from Appendix B
    if distance == 0: return 0
    return 5.9 * (distance ** -0.76) + 0.043

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_data():
    print("Loading data...")
    
    # 1. LOAD FLEET
    try:
        df_fleet = pd.read_excel("FleetType.xlsx", header=None)
        fleet = []
        num_cols = len(df_fleet.columns)
        for col_idx in range(1, num_cols):
            try:
                fleet.append(AircraftType(
                    name=str(df_fleet.iloc[0, col_idx]),
                    speed=float(df_fleet.iloc[1, col_idx]),
                    seats=int(df_fleet.iloc[2, col_idx]),
                    tat=int(df_fleet.iloc[3, col_idx]),
                    max_range=float(df_fleet.iloc[4, col_idx]),
                    runway_req=float(df_fleet.iloc[5, col_idx]),
                    lease_cost=float(df_fleet.iloc[6, col_idx]),
                    fixed_cost=float(df_fleet.iloc[7, col_idx]),
                    time_cost_param=float(df_fleet.iloc[8, col_idx]),
                    fuel_cost_param=float(df_fleet.iloc[9, col_idx]),
                    count=int(df_fleet.iloc[10, col_idx])
                ))
            except: pass
    except Exception as e:
        print(f"ERROR loading Fleet: {e}")
        return [], {}, {}, None, ""

    # 2. LOAD DEMAND & AIRPORTS
    try:
        demand_file = "DemandGroup12.xlsx" 
        df_raw = pd.read_excel(demand_file, header=None)
        
        airports = {}
        hub_name = "EDDF"
        
        # SEARCH FOR AIRPORT DATA (Robust search)
        icao_row_idx = -1
        col_start_idx = -1
        for r in range(20):
            for c in range(5):
                val = str(df_raw.iloc[r, c]).strip()
                if "ICAO Code" in val:
                    icao_row_idx = r
                    col_start_idx = c + 1
                    break
            if icao_row_idx != -1: break
            
        codes_row = df_raw.iloc[icao_row_idx, col_start_idx:]
        lats_row  = df_raw.iloc[icao_row_idx + 1, col_start_idx:]
        lons_row  = df_raw.iloc[icao_row_idx + 2, col_start_idx:]
        runways_row = df_raw.iloc[icao_row_idx + 3, col_start_idx:]
        
        valid_codes_set = set()
        
        for col in codes_row.index:
            code_val = codes_row[col]
            if pd.isna(code_val): continue
            code = str(code_val).strip()
            
            try:
                lat = pd.to_numeric(lats_row[col], errors='coerce')
                lon = pd.to_numeric(lons_row[col], errors='coerce')
                runway = pd.to_numeric(runways_row[col], errors='coerce')
                if pd.isna(runway): runway = 2000 
                
                if pd.notna(lat) and pd.notna(lon):
                    airports[code] = Airport(code, runway, 0.0, lat, lon)
                    valid_codes_set.add(code)
            except: pass

        # Distances
        if hub_name in airports:
            hub_lat = airports[hub_name].lat
            hub_lon = airports[hub_name].lon
            for code, apt in airports.items():
                if code != hub_name:
                    apt.dist_from_hub = haversine_distance(hub_lat, hub_lon, apt.lat, apt.lon)
        else:
            print(f"ERROR: Hub {hub_name} not found.")
            return fleet, {}, {}, None, ""

        # FIND MATRIX
        raw_demand = {}
        matrix_header_row = -1
        for r in range(icao_row_idx + 5, len(df_raw)):
            matches = 0
            for c in range(len(df_raw.columns)):
                val = str(df_raw.iloc[r, c]).strip()
                if val in valid_codes_set:
                    matches += 1
            if matches >= 3:
                matrix_header_row = r
                break
        
        matrix_dest_map = {}
        for c in range(len(df_raw.columns)):
            val = str(df_raw.iloc[matrix_header_row, c]).strip()
            if val in valid_codes_set:
                matrix_dest_map[c] = val
                
        for r in range(matrix_header_row + 1, len(df_raw)):
            origin = None
            val0 = str(df_raw.iloc[r, 0]).strip()
            val1 = str(df_raw.iloc[r, 1]).strip() if len(df_raw.columns)>1 else ""
            if val0 in valid_codes_set: origin = val0
            elif val1 in valid_codes_set: origin = val1
            
            if origin:
                for c, dest in matrix_dest_map.items():
                    val = df_raw.iloc[r, c]
                    if pd.notna(val) and val > 0:
                        # Division by 7 for Daily Demand
                        daily_val = val * DEMAND_FACTOR
                        raw_demand[(origin, dest)] = daily_val

        print(f"-> Loaded {len(raw_demand)} demand entries (Daily average).")

    except Exception as e:
        print(f"ERROR loading Demand: {e}")
        return fleet, {}, {}, None, ""

    # 3. Load Coefficients (STRICT LAYOUT FROM USER)
    try:
        # Load raw file (no headers, index 0, 1, 2...)
        df_raw_coef = pd.read_excel("HourCoefficients.xlsx", header=None)
        
        # Slicing based on specific description:
        # Data Rows: 3 to 22 (Indices 2 to 21)
        # ICAO Code: Column C (Index 2)
        # Hour 0 to 23: Column D (Index 3) to AA (Index 26)
        
        # 1. Extract the block [Rows 2:22, Cols 2:27]
        # (End index is exclusive, so 22 gets rows up to 21, 27 gets cols up to 26)
        df_coef = df_raw_coef.iloc[2:22, 2:27].copy()
        
        # 2. Set the first column of this block (Index 2) as the Index (ICAO Codes)
        df_coef = df_coef.set_index(2)
        
        # 3. Rename columns. Current column names are integers 3...26.
        # We want to map Col 3 -> 0, Col 4 -> 1 ... Col 26 -> 23
        col_map = {i: (i - 3) for i in range(3, 27)}
        df_coef = df_coef.rename(columns=col_map)
        
        # 4. Clean index strings
        df_coef.index = df_coef.index.astype(str).str.strip()
        
        print(f"-> Coefficients loaded strictly. Found {len(df_coef)} airports.")
        
    except Exception as e:
        print(f"WARNING: Coef load failed ({e}). Using flat demand.")
        df_coef = pd.DataFrame()
        
    # Sort fleet by seats (Descending) so Big Jets get priority
    fleet.sort(key=lambda x: x.seats, reverse=True)

    return fleet, airports, raw_demand, df_coef, hub_name

# --- DYNAMIC PROGRAMMING SOLVER ---

class DPSolver:
    def __init__(self, ac_type: AircraftType, airports, demand_state, hourly_coef, hub_id):
        self.ac = ac_type
        self.airports = airports
        self.demand_state = demand_state
        self.coef = hourly_coef
        self.hub = hub_id
        self.V = {} 
    
    def get_demand_at_hour(self, origin, dest, hour):
        if (origin, dest) not in self.demand_state:
            return 0
        
        daily_total = self.demand_state[(origin, dest)]
        h = int(hour) % 24
        
        # Default flat profile if not found
        c = 0.04 
        
        # STRICT LOOKUP
        if origin in self.coef.index:
            # We expect columns 0, 1, ... 23 to exist as integers
            try:
                c = self.coef.loc[origin, h]
            except:
                pass # Column 'h' missing? Keep 0.04
        
        return daily_total * c

    def get_recaptured_demand(self, origin, dest, current_step):
        current_hour = current_step // STEPS_PER_HOUR
        total_pax = 0
        for h in [current_hour, current_hour - 1, current_hour - 2]:
            if h < 0: continue 
            pax = self.get_demand_at_hour(origin, dest, h)
            total_pax += pax
        return total_pax

    def solve(self):
        self.V[TOTAL_STEPS] = {}
        for aid in self.airports:
            if aid == self.hub: self.V[TOTAL_STEPS][aid] = (0, None)
            else: self.V[TOTAL_STEPS][aid] = (-float('inf'), None)

        for t in range(TOTAL_STEPS - 1, -1, -1):
            self.V[t] = {}
            for loc in self.airports:
                # WAIT
                best_val = -float('inf')
                best_action = ("WAIT", loc)
                if loc in self.V[t+1]:
                    wait_val = self.V[t+1][loc][0]
                    if wait_val > -float('inf'): best_val = wait_val

                # FLY
                possible_dests = [a for a in self.airports if a != self.hub] if loc == self.hub else [self.hub]

                for dest in possible_dests:
                    dist = self.airports[dest].dist_from_hub if loc == self.hub else self.airports[loc].dist_from_hub
                    
                    if dist > self.ac.max_range: continue
                    if self.airports[dest].runway < self.ac.runway_req: continue

                    steps_needed = self.ac.calc_steps(dist)
                    arrival_time = t + steps_needed

                    if arrival_time <= TOTAL_STEPS:
                        avail_demand = self.get_recaptured_demand(loc, dest, t)
                        # We cap the seat count at 80% of actual capacity
                        safe_seats = int(self.ac.seats * 0.80)
                        pax = min(safe_seats, avail_demand)
                        
                        yield_val = calculate_yield(dist)
                        revenue = pax * yield_val * dist
                        cost = self.ac.calc_trip_cost(dist)
                        leg_profit = revenue - cost
                        
                        future_val = self.V[arrival_time][dest][0]
                        if future_val > -float('inf'):
                            total_val = leg_profit + future_val
                            if total_val > best_val:
                                best_val = total_val
                                best_action = ("FLY", dest, pax, dist, arrival_time)

                self.V[t][loc] = (best_val, best_action)
        return self.V[0][self.hub]

    def reconstruct_path_and_update_demand(self):
        schedule = []
        curr_t = 0
        curr_loc = self.hub
        
        if curr_loc not in self.V[0]: return None, 0
        start_val = self.V[0][self.hub][0]
        # DP consistency check
        if start_val == -float('inf'): return None, 0

        total_profit = 0
        total_block_minutes = 0

        while curr_t < TOTAL_STEPS:
            val, action = self.V[curr_t][curr_loc]
            
            if action[0] == "WAIT":
                curr_t += 1
            elif action[0] == "FLY":
                dest = action[1]
                pax_carried = action[2]
                dist = action[3]
                arrival_t = action[4]
                
                schedule.append({
                    "Origin": curr_loc,
                    "Dest": dest,
                    "Dep_Step": curr_t,
                    "Arr_Step": arrival_t - (math.ceil(self.ac.tat / TIME_STEP_MIN)),
                    "Pax": pax_carried,
                    "Dist": dist
                })
                
                # Update Global Demand
                pax_left_to_remove = pax_carried
                origin = curr_loc
                current_hour = curr_t // STEPS_PER_HOUR
                for h in [current_hour, current_hour-1, current_hour-2]:
                    if h < 0 or pax_left_to_remove <= 0: break
                    available = self.get_demand_at_hour(origin, dest, h)
                    
                    coef = 0.04 
                    # STRICT LOOKUP
                    if origin in self.coef.index:
                        try: coef = self.coef.loc[origin, h]
                        except: pass
                    
                    if coef > 0:
                        to_take = min(available, pax_left_to_remove)
                        deduction = to_take / coef
                        if (origin, dest) in self.demand_state:
                            self.demand_state[(origin, dest)] = max(0, self.demand_state[(origin, dest)] - deduction)
                        pax_left_to_remove -= to_take

                revenue = pax_carried * calculate_yield(dist) * dist
                cost = self.ac.calc_trip_cost(dist)
                total_profit += (revenue - cost)
                flight_time_min = (arrival_t - curr_t) * TIME_STEP_MIN
                total_block_minutes += flight_time_min

                curr_t = arrival_t
                curr_loc = dest
        
        # 1. Deduct Lease Cost (This is the Net Profit IF we fly)
        net_profit = total_profit - self.ac.lease_cost
        
        # 2. Check Constraints
        if total_block_minutes < 360: 
            return None, 0

        if net_profit < 0:
            return None, 0

        # 3. Success
        return schedule, net_profit

# --- MAIN EXECUTION ---

def main():
    fleet, airports, raw_demand, df_coef, hub_name = load_data()

    if not fleet or not airports:
        return

    # Track remaining aircraft counts
    remaining = {ac.name: ac.count for ac in fleet}

    final_report = []
    current_demand = raw_demand.copy()

    aircraft_id_counter = {ac.name: 0 for ac in fleet}

    while True:
        best_profit = 0
        best_ac = None
        best_schedule = None

        # 1. TEST all aircraft types (one copy each)
        for ac_type in fleet:
            if remaining[ac_type.name] <= 0:
                continue

            # IMPORTANT: use a COPY of demand so nothing is committed
            test_demand = current_demand.copy()

            solver = DPSolver(
                ac_type,
                airports,
                test_demand,
                df_coef,
                hub_name
            )

            solver.solve()
            schedule, profit = solver.reconstruct_path_and_update_demand()

            if schedule and profit > best_profit:
                best_profit = profit
                best_ac = ac_type
                best_schedule = schedule

        # 2. STOP if nothing profitable remains
        if best_ac is None:
            break

        # 3. COMMIT the best aircraft (now update real demand)
        solver = DPSolver(
            best_ac,
            airports,
            current_demand,
            df_coef,
            hub_name
        )

        solver.solve()
        schedule, profit = solver.reconstruct_path_and_update_demand()

        aircraft_id_counter[best_ac.name] += 1
        aircraft_label = f"{best_ac.name}_{aircraft_id_counter[best_ac.name]}"

        final_report.append({
            "Aircraft": aircraft_label,
            "Schedule": schedule,
            "Profit": profit
        })

        remaining[best_ac.name] -= 1

        print(
            f"Selected {aircraft_label}: "
            f"Profit = {profit:.2f}, Legs = {len(schedule)}"
        )

    df_long = pd.DataFrame(
        [(o, d, v) for (o, d), v in current_demand.items()],
        columns=["Origin", "Destination", "Demand"]
    )

    df_od = df_long.pivot(
        index="Origin",
        columns="Destination",
        values="Demand"
    ).fillna(0)

    airport_order = [
    "EGLL", "LFPG", "EHAM", "EDDF", "LEMF", "LEBL", "EDDM", "LIRF", "EIDW", "ESSA",
    "LPPT", "EDDT", "EFHK", "EPWA", "EGPH", "LROP", "LGIR", "BIKF", "LICJ", "LPMA"
    ]

    df_od = df_od.reindex(index=airport_order, columns=airport_order)

    output_file = "final_remaining_demand.xlsx"
    df_od.to_excel(output_file)

    print(f"Saved final demand matrix to {output_file}")

    # Optional summary
    total_profit = sum(r["Profit"] for r in final_report)
    print(f"\nTotal Fleet Profit: {total_profit:.2f}")

    
    print("\n\n=== FINAL REPORT ===")
    total_airline_profit = 0
    for item in final_report:
        print(f"Aircraft: {item['Aircraft']} | Profit: {item['Profit']:.2f} EUR")
        total_airline_profit += item['Profit']
        for leg in item['Schedule']:
            dep_time = f"{int((leg['Dep_Step']*TIME_STEP_MIN)//60):02d}:{int((leg['Dep_Step']*TIME_STEP_MIN)%60):02d}"
            arr_time = f"{int((leg['Arr_Step']*TIME_STEP_MIN)//60):02d}:{int((leg['Arr_Step']*TIME_STEP_MIN)%60):02d}"
            print(f"  {dep_time} {leg['Origin']} -> {leg['Dest']} ({arr_time}) | Pax: {int(leg['Pax'])}")
    
    print(f"Total Airline Profit: {total_airline_profit:.2f} EUR")

    #print((math.ceil((leg['Dep_Step']*TIME_STEP_MIN)/60.0)))

if __name__ == "__main__":
    main()