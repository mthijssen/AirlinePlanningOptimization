#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version with per-hour demand state.

- Demand is stored as D_ij^h (hourly demand) instead of a single daily total.
- Recapture uses hours {h, h-1, h-2}.
- After a flight, passengers are subtracted directly from those hourly buckets.

"""

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
        c_t = self.time_cost_param * (dist / self.speed)
        # Fuel Cost
        c_f = (self.fuel_cost_param * FUEL_COST_PER_GALLON / FUEL_FACTOR) * dist
        # Total = Fixed + Time + Fuel
        return self.fixed_cost + c_t + c_f


# --- HELPER FUNCTIONS ---

def calculate_yield(distance):
    # Formula from Appendix B
    if distance == 0:
        return 0
    return 5.9 * (distance ** -0.76) + 0.043


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
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
            except:
                pass
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
            if icao_row_idx != -1:
                break

        codes_row = df_raw.iloc[icao_row_idx, col_start_idx:]
        lats_row = df_raw.iloc[icao_row_idx + 1, col_start_idx:]
        lons_row = df_raw.iloc[icao_row_idx + 2, col_start_idx:]
        runways_row = df_raw.iloc[icao_row_idx + 3, col_start_idx:]

        valid_codes_set = set()

        for col in codes_row.index:
            code_val = codes_row[col]
            if pd.isna(code_val):
                continue
            code = str(code_val).strip()

            try:
                lat = pd.to_numeric(lats_row[col], errors='coerce')
                lon = pd.to_numeric(lons_row[col], errors='coerce')
                runway = pd.to_numeric(runways_row[col], errors='coerce')
                if pd.isna(runway):
                    runway = 2000

                if pd.notna(lat) and pd.notna(lon):
                    airports[code] = Airport(code, runway, 0.0, lat, lon)
                    valid_codes_set.add(code)
            except:
                pass

        # Distances from hub
        if hub_name in airports:
            hub_lat = airports[hub_name].lat
            hub_lon = airports[hub_name].lon
            for code, apt in airports.items():
                if code != hub_name:
                    apt.dist_from_hub = haversine_distance(hub_lat, hub_lon, apt.lat, apt.lon)
        else:
            print(f"ERROR: Hub {hub_name} not found.")
            return fleet, {}, {}, None, ""

        # FIND DEMAND MATRIX
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
            val1 = str(df_raw.iloc[r, 1]).strip() if len(df_raw.columns) > 1 else ""
            if val0 in valid_codes_set:
                origin = val0
            elif val1 in valid_codes_set:
                origin = val1

            if origin:
                for c, dest in matrix_dest_map.items():
                    val = df_raw.iloc[r, c]
                    if pd.notna(val) and val > 0:
                        # DEMAND_FACTOR is already applied here
                        daily_val = val * DEMAND_FACTOR
                        raw_demand[(origin, dest)] = daily_val

        print(f"-> Loaded {len(raw_demand)} demand entries (Daily average).")

    except Exception as e:
        print(f"ERROR loading Demand: {e}")
        return fleet, {}, {}, None, ""

    # 3. Load Coefficients (STRICT LAYOUT FROM USER)
    try:
        df_raw_coef = pd.read_excel("HourCoefficients.xlsx", header=None)

        # Rows 2:22, Cols 2:27
        df_coef = df_raw_coef.iloc[2:22, 2:27].copy()
        df_coef = df_coef.set_index(2)
        col_map = {i: (i - 3) for i in range(3, 27)}
        df_coef = df_coef.rename(columns=col_map)
        df_coef.index = df_coef.index.astype(str).str.strip()

        print(f"-> Coefficients loaded strictly. Found {len(df_coef)} airports.")

    except Exception as e:
        print(f"WARNING: Coef load failed ({e}). Using flat demand profile.")
        df_coef = pd.DataFrame()

    # Sort fleet by seats (Descending) so Big Jets get priority
    fleet.sort(key=lambda x: x.seats, reverse=True)

    return fleet, airports, raw_demand, df_coef, hub_name


# --- NEW: BUILD HOURLY DEMAND STATE FROM DAILY DEMAND + COEFFICIENTS ---

def build_hourly_demand(raw_demand, df_coef):
    """
    Create hourly demand D_ij^h from daily demand D_ij and coefficients α_i^h.

    hourly_demand[(i, j, h)] = D_ij * α_i^h
    If no coefficient is available, use a flat profile with coefficient 0.04.
    """
    hourly_demand = {}
    for (origin, dest), daily_val in raw_demand.items():
        for h in range(24):
            # Default flat coefficient if not found
            coef = 0.04
            if not df_coef.empty and origin in df_coef.index:
                try:
                    coef = df_coef.loc[origin, h]
                except Exception:
                    pass
            # Demand per hour
            demand_h = daily_val * coef
            if demand_h > 0:
                hourly_demand[(origin, dest, h)] = demand_h
    return hourly_demand


# --- DYNAMIC PROGRAMMING SOLVER ---

class DPSolver:
    def __init__(self, ac_type: AircraftType, airports, hourly_demand_state, hub_id):
        self.ac = ac_type
        self.airports = airports
        # hourly_demand_state[(i, j, h)] = remaining demand at hour h
        self.demand_state = hourly_demand_state  # CHANGED: now per-hour
        self.hub = hub_id
        self.V = {}

    def get_demand_at_hour(self, origin, dest, hour):
        """Return remaining demand D_ij^h at hour h."""
        h = int(hour)
        if h < 0 or h > 23:
            return 0
        key = (origin, dest, h)
        return self.demand_state.get(key, 0)

    def get_recaptured_demand(self, origin, dest, current_step):
        """Sum demand from hours {h, h-1, h-2}, skipping negative hours."""
        current_hour = current_step // STEPS_PER_HOUR
        total_pax = 0
        for h in [current_hour, current_hour - 1, current_hour - 2]:
            if h < 0:
                continue
            total_pax += self.get_demand_at_hour(origin, dest, h)
        return total_pax

    def solve(self):
        # Terminal condition
        self.V[TOTAL_STEPS] = {}
        for aid in self.airports:
            if aid == self.hub:
                self.V[TOTAL_STEPS][aid] = (0, None)
            else:
                self.V[TOTAL_STEPS][aid] = (-float('inf'), None)

        # Backward induction
        for t in range(TOTAL_STEPS - 1, -1, -1):
            self.V[t] = {}
            for loc in self.airports:
                # WAIT
                best_val = -float('inf')
                best_action = ("WAIT", loc)
                if loc in self.V[t + 1]:
                    wait_val = self.V[t + 1][loc][0]
                    if wait_val > -float('inf'):
                        best_val = wait_val

                # FLY
                if loc == self.hub:
                    possible_dests = [a for a in self.airports if a != self.hub]
                else:
                    possible_dests = [self.hub]

                for dest in possible_dests:
                    # Distances based on hub-centric distances
                    dist = (self.airports[dest].dist_from_hub
                            if loc == self.hub else
                            self.airports[loc].dist_from_hub)

                    # Range constraint
                    if dist > self.ac.max_range:
                        continue
                    # Runway constraint
                    if self.airports[dest].runway < self.ac.runway_req:
                        continue

                    steps_needed = self.ac.calc_steps(dist)
                    arrival_time = t + steps_needed
                    if arrival_time <= TOTAL_STEPS:
                        avail_demand = self.get_recaptured_demand(loc, dest, t)
                        # Seat cap at 80%
                        safe_seats = int(self.ac.seats * 0.80)
                        pax = min(safe_seats, avail_demand)

                        if pax <= 0:
                            continue

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

        if curr_loc not in self.V[0]:
            return None, 0
        start_val = self.V[0][self.hub][0]
        if start_val == -float('inf'):
            return None, 0

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
                    # Arrival minus turnaround portion in steps
                    "Arr_Step": arrival_t - (math.ceil(self.ac.tat / TIME_STEP_MIN)),
                    "Pax": pax_carried,
                    "Dist": dist
                })

                # --- CHANGED: Update hourly demand directly ---
                origin = curr_loc
                pax_left_to_remove = pax_carried
                current_hour = curr_t // STEPS_PER_HOUR

                # Remove passengers from hours h, h-1, h-2 sequentially
                for h in [current_hour, current_hour - 1, current_hour - 2]:
                    if h < 0 or pax_left_to_remove <= 0:
                        break
                    key = (origin, dest, h)
                    available = self.demand_state.get(key, 0)
                    if available <= 0:
                        continue
                    to_take = min(available, pax_left_to_remove)
                    self.demand_state[key] = available - to_take
                    pax_left_to_remove -= to_take

                # Compute profit for this leg
                revenue = pax_carried * calculate_yield(dist) * dist
                cost = self.ac.calc_trip_cost(dist)
                total_profit += (revenue - cost)

                # Block time in minutes
                flight_time_min = (arrival_t - curr_t) * TIME_STEP_MIN
                total_block_minutes += flight_time_min

                curr_t = arrival_t
                curr_loc = dest

        # Net profit after lease cost
        net_profit = total_profit - self.ac.lease_cost

        # Minimum utilisation: 6 hours block time
        if total_block_minutes < 360:
            return None, 0

        if net_profit < 0:
            return None, 0

        return schedule, net_profit


# --- MAIN EXECUTION ---

def main():
    fleet, airports, raw_demand, df_coef, hub_name = load_data()
    if not fleet or not airports:
        return

    # Build initial hourly demand state
    hourly_demand = build_hourly_demand(raw_demand, df_coef)  # NEW
    current_demand = hourly_demand.copy()  # dict[(i,j,h)] -> remaining demand

    # Track remaining aircraft counts
    remaining = {ac.name: ac.count for ac in fleet}

    final_report = []
    aircraft_id_counter = {ac.name: 0 for ac in fleet}

    total_profit = 0.0

    while True:
        best_profit = 0
        best_ac = None
        best_schedule = None

        # Phase 1: test all aircraft types with remaining copies
        for ac_type in fleet:
            if remaining[ac_type.name] <= 0:
                continue

            # Use a copy of the demand so nothing is committed
            test_demand = current_demand.copy()

            solver = DPSolver(
                ac_type,
                airports,
                test_demand,  # per-hour demand
                hub_name
            )

            solver.solve()
            schedule, profit = solver.reconstruct_path_and_update_demand()

            if schedule and profit > best_profit:
                best_profit = profit
                best_ac = ac_type
                best_schedule = schedule

        # Stop if no profitable schedule
        if best_ac is None:
            break

        # Phase 2: commit best aircraft schedule on real demand
        solver = DPSolver(
            best_ac,
            airports,
            current_demand,
            hub_name
        )

        solver.solve()
        schedule, profit = solver.reconstruct_path_and_update_demand()

        if not schedule:
            # Safety: if for some reason recomputation fails, stop
            break

        aircraft_id_counter[best_ac.name] += 1
        aircraft_label = f"{best_ac.name}_{aircraft_id_counter[best_ac.name]}"

        final_report.append({
            "Aircraft": aircraft_label,
            "Schedule": schedule,
            "Profit": profit
        })

        remaining[best_ac.name] -= 1
        total_profit += profit

        print(
            f"Selected {aircraft_label}: "
            f"Profit = {profit:.2f}, Legs = {len(schedule)}"
        )

    # Export remaining demand (aggregated back to daily per OD for inspection)
    # Sum remaining hourly demand over all hours for each OD
    remaining_daily = {}
    for (o, d, h), v in current_demand.items():
        if v <= 0:
            continue
        key = (o, d)
        remaining_daily[key] = remaining_daily.get(key, 0) + v

    df_long = pd.DataFrame(
        [(o, d, v) for (o, d), v in remaining_daily.items()],
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

    print(f"Saved final remaining demand matrix to {output_file}")

    print(f"\nTotal Fleet Profit: {total_profit:.2f}")

    print("\n\n=== FINAL REPORT ===")
    total_airline_profit = 0
    for item in final_report:
        print(f"Aircraft: {item['Aircraft']} | Profit: {item['Profit']:.2f} EUR")
        total_airline_profit += item['Profit']
        for leg in item['Schedule']:
            dep_time = f"{int((leg['Dep_Step'] * TIME_STEP_MIN) // 60):02d}:{int((leg['Dep_Step'] * TIME_STEP_MIN) % 60):02d}"
            arr_time = f"{int((leg['Arr_Step'] * TIME_STEP_MIN) // 60):02d}:{int((leg['Arr_Step'] * TIME_STEP_MIN) % 60):02d}"
            print(f"  {dep_time} {leg['Origin']} -> {leg['Dest']} ({arr_time}) | Pax: {int(leg['Pax'])}")

    print(f"Total Airline Profit: {total_airline_profit:.2f} EUR")


if __name__ == "__main__":
    main()
