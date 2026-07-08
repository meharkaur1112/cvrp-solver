"""
01_milp_cvrp.py
===============
Capacitated Vehicle Routing Problem (CVRP) — Exact MILP Formulation
--------------------------------------------------------------------
Solves the CVRP as a Mixed-Integer Linear Program (MILP) using
Miller-Tucker-Zemlin (MTZ) subtour elimination constraints.

This script is the first of three approaches benchmarked in this project:
    01_milp_cvrp.py      — Exact MILP formulation (PuLP / CBC)  ← this file
    02_clarke_wright.py  — Clarke-Wright Savings constructive heuristic
    03_pyvrp_solver.py   — PyVRP metaheuristic solver

Problem:
    Given a depot, a fleet of identical vehicles with fixed capacity,
    and a set of customers with known delivery demands, find the
    minimum-distance set of routes that:
        - Serves every customer exactly once
        - Returns every vehicle to the depot after each route
        - Never exceeds any vehicle's weight capacity

Formulation:
    Decision variables:
        x[i,j,k] in {0,1}  —  1 if vehicle k travels arc (i -> j)
        u[i,k]   >= 0      —  cumulative load when vehicle k arrives at i

    Objective:
        Minimise sum of c[i][j] * x[i,j,k] over all arcs and vehicles

    Constraints:
        C1. Every customer visited exactly once (incoming and outgoing arcs)
        C2. Flow conservation at every customer node per vehicle
        C3. Each vehicle leaves and returns to depot at most once
        C4. MTZ subtour elimination + implicit capacity enforcement

    Note on CPLEX Community Edition:
        The Community Edition enforces a hard limit of 1,000 variables.
        Instances exceeding this silently report infeasibility rather than
        raising an explicit error. Cross-validate against CBC if this occurs.

Dataset:
    Solomon C101 benchmark (1987) — industry standard in VRP research.

Distance metric:
    Euclidean distance.

Dependencies:
    pip install pulp matplotlib numpy

Usage:
    Update FILE_PATH and parameters below, then run:
        python 01_milp_cvrp.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import math
import time
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FILE_PATH     = "c101.txt"   # path to Solomon c101.txt
NUM_CUSTOMERS = 10                # keep <= 12 for CBC to solve in reasonable time
NUM_VEHICLES  = 2                 # must satisfy: NUM_VEHICLES * capacity >= total demand
TIME_LIMIT    = 300               # solver time limit in seconds


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARSE THE SOLOMON INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

with open(FILE_PATH, 'r') as f:
    lines = f.readlines()

# Strip blank lines and trailing whitespace before indexing
lines = [l.strip() for l in lines if l.strip()]

instance_name = lines[0]

# Line index 3: "num_vehicles  capacity"
vehicle_line = lines[3].split()
num_vehicles = int(vehicle_line[0])
capacity     = int(vehicle_line[1])

# Lines 6 onwards: one row per node (depot + customers)
nodes = []
for line in lines[6:]:
    parts = line.split()
    if len(parts) < 7:
        continue
    nodes.append({
        'id'          : int(parts[0]),
        'x'           : float(parts[1]),
        'y'           : float(parts[2]),
        'demand'      : float(parts[3]),
        'ready_time'  : float(parts[4]),
        'due_date'    : float(parts[5]),
        'service_time': float(parts[6]),
    })

print(f"Instance     : {instance_name}")
print(f"Capacity     : {capacity}")
print(f"Nodes loaded : {len(nodes)}  (1 depot + {len(nodes)-1} customers)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD SUBSET AND DISTANCE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

# Node 0 is the depot; nodes 1..NUM_CUSTOMERS are the customers we solve for
subset_nodes = [nodes[0]] + nodes[1:NUM_CUSTOMERS + 1]
demand       = [node['demand'] for node in subset_nodes]
total_demand = sum(demand[1:])
subset_size  = len(subset_nodes)

print(f"\nSubset               : 1 depot + {NUM_CUSTOMERS} customers")
print(f"Total demand         : {total_demand}")
print(f"Fleet capacity       : {NUM_VEHICLES} x {capacity} = {NUM_VEHICLES * capacity}")
print(f"Min vehicles needed  : {math.ceil(total_demand / capacity)}")

# Euclidean distance matrix over the subset
sub_dist = [
    [math.sqrt((subset_nodes[i]['x'] - subset_nodes[j]['x'])**2 +
               (subset_nodes[i]['y'] - subset_nodes[j]['y'])**2)
     for j in range(subset_size)]
    for i in range(subset_size)
]

print(f"\nDistance matrix shape : {subset_size} x {subset_size}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. DEFINE INDEX SETS
# ══════════════════════════════════════════════════════════════════════════════

# Vehicles are 1-indexed to avoid collision with depot node index 0
Nodes     = list(range(subset_size))          # [0, 1, ..., n]
Customers = list(range(1, subset_size))        # [1, ..., n]
Vehicles  = list(range(1, NUM_VEHICLES + 1))   # [1, ..., K]


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD THE MILP MODEL
# ══════════════════════════════════════════════════════════════════════════════

# PuLP uses += to add the objective and constraints to the LpProblem object.
# The problem is automatically treated as a MILP because x variables are Binary.
prob = pulp.LpProblem("CVRP_C101_subset", pulp.LpMinimize)

# ── Decision variable: x[i,j,k] = 1 if vehicle k travels arc (i -> j) ────
# Binary — this is what makes the problem a MILP rather than a pure LP.
# Vehicles are 1-indexed to avoid key collision with depot node 0.
x = {
    (i, j, k): pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")
    for i in Nodes for j in Nodes for k in Vehicles if i != j
}

# ── Decision variable: u[i,k] = cumulative load when vehicle k arrives at i ──
# Used in MTZ subtour elimination (constraint C4 below).
# Bounds: demand[i] <= u[i,k] <= capacity  ensures load grows along each route
# and never exceeds vehicle capacity.
u = {
    (i, k): pulp.LpVariable(f"u_{i}_{k}", lowBound=demand[i], upBound=capacity)
    for i in Customers for k in Vehicles
}

print(f"\nx variables  : {len(x)}")
print(f"u variables  : {len(u)}")

# ── Objective: minimise total Euclidean distance travelled ────────────────
prob += pulp.lpSum(
    sub_dist[i][j] * x[i, j, k]
    for i in Nodes for j in Nodes for k in Vehicles if i != j
)

# ── C1: Every customer visited exactly once (one incoming arc total) ──────
for j in Customers:
    prob += pulp.lpSum(
        x[i, j, k] for i in Nodes for k in Vehicles if i != j
    ) == 1

# ── C1b: Every customer departed from exactly once (one outgoing arc total) ─
for i in Customers:
    prob += pulp.lpSum(
        x[i, j, k] for j in Nodes for k in Vehicles if j != i
    ) == 1

# ── C2: Flow conservation — if vehicle k enters node h it must leave h ───
# Ensures routes are continuous; a vehicle cannot arrive somewhere without
# also departing.
for k in Vehicles:
    for h in Customers:
        prob += (
            pulp.lpSum(x[i, h, k] for i in Nodes if i != h) ==
            pulp.lpSum(x[h, j, k] for j in Nodes if j != h)
        )

# ── C3: Each vehicle leaves/returns to depot at most once ─────────────────
# <= 1 (not == 1) allows vehicles to remain idle when fewer routes are needed
for k in Vehicles:
    prob += pulp.lpSum(x[0, j, k] for j in Customers) <= 1
    prob += pulp.lpSum(x[i, 0, k] for i in Customers) <= 1

# ── C4: MTZ subtour elimination + implicit capacity enforcement ───────────
# If x[i,j,k]=1 then u[j,k] >= u[i,k] + demand[j].
# This forces cumulative load to increase monotonically along each route,
# simultaneously preventing subtours and enforcing capacity.
# Q acts as a big-M coefficient — a known limitation that weakens the LP
# relaxation and slows branch-and-bound at larger problem sizes.
for k in Vehicles:
    for i in Customers:
        for j in Customers:
            if i != j:
                prob += (
                    u[i, k] - u[j, k] + capacity * x[i, j, k]
                    <= capacity - demand[j]
                )

print(f"Constraints  : {len(prob.constraints)}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SOLVE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSolving with CBC  (time limit: {TIME_LIMIT}s)...")
print("-" * 60)

t0 = time.time()
pulp.PULP_CBC_CMD(msg=True, timeLimit=TIME_LIMIT).solve(prob)
elapsed = time.time() - t0

print("-" * 60)
print(f"Status      : {pulp.LpStatus[prob.status]}")
print(f"Objective   : {pulp.value(prob.objective):.4f}")
print(f"Solve time  : {elapsed:.2f} seconds")


# ══════════════════════════════════════════════════════════════════════════════
# 7. EXTRACT AND PRINT ROUTES
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Route Details ───────────────────────────────────────")

routes       = []
all_visited  = []
manual_total = 0.0

for k in Vehicles:
    # Find the arc leaving the depot for this vehicle
    for j in Customers:
        val = pulp.value(x[0, j, k])
        if val is not None and round(val) == 1:
            route = [0, j]
            # Follow the chain of active arcs back to the depot
            while route[-1] != 0:
                current = route[-1]
                for jj in Nodes:
                    if jj != current:
                        vv = pulp.value(x[current, jj, k])
                        if vv is not None and round(vv) == 1:
                            route.append(jj)
                            break
            routes.append((k, route))

for k, route in routes:
    load = sum(demand[node] for node in route if node != 0)
    dist = sum(sub_dist[route[i]][route[i+1]] for i in range(len(route)-1))
    manual_total += dist
    all_visited.extend([node for node in route if node != 0])

    labels = " -> ".join("Depot" if n == 0 else f"C{n}" for n in route)
    print(f"\n  Vehicle {k}: {labels}")
    print(f"  Load     : {load:.0f} / {capacity}")
    print(f"  Distance : {dist:.4f}")

print(f"\n  Total distance  : {manual_total:.4f}")
print(f"  Vehicles used   : {len(routes)} / {NUM_VEHICLES}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

expected = set(range(1, NUM_CUSTOMERS + 1))
visited  = set(all_visited)
all_once = (visited == expected and len(all_visited) == len(set(all_visited)))
cap_ok   = all(
    sum(demand[n] for n in route if n != 0) <= capacity
    for _, route in routes
)

print(f"\n── Validation ──────────────────────────────────────────")
print(f"  All customers visited exactly once? {all_once}")
print(f"  All routes within capacity?         {cap_ok}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

palette = ["#E63946", "#2A9D8F", "#E9C46A", "#F4A261", "#264653"]
coords  = [(n['x'], n['y']) for n in subset_nodes]

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#F8F9FA")

for k, route in routes:
    col = palette[(k - 1) % len(palette)]
    for step in range(len(route) - 1):
        a, b = coords[route[step]], coords[route[step + 1]]
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.2))

for i in range(1, len(subset_nodes)):
    ax.scatter(*coords[i], s=180, color="#264653", zorder=5)
    ax.annotate(f"C{i}", coords[i], textcoords="offset points",
                xytext=(6, 5), fontsize=8, fontweight="bold", color="#264653")

ax.scatter(*coords[0], s=350, marker="*", color="#E63946", zorder=6)
ax.annotate("Depot", coords[0], textcoords="offset points",
            xytext=(8, 6), fontsize=10, fontweight="bold", color="#E63946")

patches = [mpatches.Patch(color=palette[(k-1) % len(palette)],
           label=f"Vehicle {k}") for k, _ in routes]
ax.legend(handles=patches, loc="upper right", fontsize=10)

ax.set_title(
    f"CVRP — MILP Solution (CBC)  |  cost = {pulp.value(prob.objective):.4f}  |  "
    f"{NUM_CUSTOMERS} customers",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("outputs/milp_routes.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nRoute map saved -> outputs/milp_routes.png")
