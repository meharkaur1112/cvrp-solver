"""
03_pyvrp_solver.py
==================
Capacitated Vehicle Routing Problem (CVRP) — PyVRP Metaheuristic Solver
------------------------------------------------------------------------
Solves the CVRP using PyVRP, a state-of-the-art vehicle routing solver
based on iterated local search (published in INFORMS Journal on Computing, 2024).

This script is the third of three approaches benchmarked in this project:
    01_milp_cvrp.py      — Exact MILP formulation (PuLP / CBC)
    02_clarke_wright.py  — Clarke-Wright Savings constructive heuristic
    03_pyvrp_solver.py   — PyVRP metaheuristic solver  ← this file

Problem:
    Given a depot, a fleet of identical vehicles with fixed capacity,
    and a set of customers with known delivery demands, find the
    minimum-distance set of routes that:
        - Serves every customer exactly once
        - Returns every vehicle to the depot after each route
        - Never exceeds any vehicle's weight capacity

How PyVRP differs from the MILP in 01_milp_cvrp.py:
    - No explicit x[i,j,k] binary arc variables
    - No MTZ subtour elimination constraints
    - No u[i,k] cumulative load variables
    - Uses iterated local search rather than branch-and-bound
    - Results are near-optimal rather than proven optimal
    - Scales to 100+ customers in seconds vs hours for the exact MILP

Dataset:
    Solomon C101 benchmark (1987) — industry standard in VRP research.
    Time window fields are parsed but ignored (pure CVRP, not VRPTW).

Distance metric:
    Euclidean distance, rounded to the nearest integer (required by PyVRP's
    internal integer arithmetic). Edges must be explicitly declared —
    PyVRP does not auto-compute distances from coordinates.

Dependencies:
    pip install pyvrp matplotlib numpy

Usage:
    Update FILE_PATH and parameters below, then run:
        python 03_pyvrp_solver.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import math
import time
import matplotlib.pyplot as plt
from pyvrp import Model
from pyvrp.stop import MaxRuntime


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FILE_PATH     = "data/c101.txt"
NUM_CUSTOMERS = 25
NUM_VEHICLES  = 5
RUNTIME_SEC   = 40    # PyVRP search budget — keeps improving until time is up


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
        'ready_time'  : float(parts[4]),   # parsed but not used (pure CVRP)
        'due_date'    : float(parts[5]),   # parsed but not used (pure CVRP)
        'service_time': float(parts[6]),   # parsed but not used (pure CVRP)
    })

print(f"Instance     : {instance_name}")
print(f"Capacity     : {capacity}")
print(f"Nodes loaded : {len(nodes)}  (1 depot + {len(nodes)-1} customers)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD SUBSET
# ══════════════════════════════════════════════════════════════════════════════

# Node 0 is the depot; nodes 1..NUM_CUSTOMERS are the customers we solve for
subset_nodes = [nodes[0]] + nodes[1:NUM_CUSTOMERS + 1]
demand       = [node['demand'] for node in subset_nodes]
total_demand = sum(demand[1:])

print(f"\nSubset               : 1 depot + {NUM_CUSTOMERS} customers")
print(f"Total demand         : {total_demand}")
print(f"Fleet capacity       : {NUM_VEHICLES} x {capacity} = {NUM_VEHICLES * capacity}")
print(f"Min vehicles needed  : {math.ceil(total_demand / capacity)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD THE PYVRP MODEL
# ══════════════════════════════════════════════════════════════════════════════

m = Model()

# Vehicle type must be declared before depot/clients (PyVRP convention)
m.add_vehicle_type(num_available=NUM_VEHICLES, capacity=int(capacity))

# Add the depot (node 0)
depot = m.add_depot(
    x=int(subset_nodes[0]['x']),
    y=int(subset_nodes[0]['y'])
)

# Add one client per customer node
clients = [
    m.add_client(
        x=int(subset_nodes[i]['x']),
        y=int(subset_nodes[i]['y']),
        delivery=int(demand[i])
    )
    for i in range(1, len(subset_nodes))
]

# Add edges with Euclidean distance.
# PyVRP requires every edge to be explicitly declared — missing edges
# default to MAX_VALUE (effectively unusable). Distances must be integers.
for frm in m.locations:
    for to in m.locations:
        if frm != to:
            dist = int(round(math.sqrt((frm.x - to.x)**2 + (frm.y - to.y)**2)))
            m.add_edge(frm, to, distance=dist)

print(f"\nModel built  : 1 depot, {len(clients)} clients, {NUM_VEHICLES} vehicles")
print(f"Total edges  : {len(m.locations) * (len(m.locations) - 1)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SOLVE
# ══════════════════════════════════════════════════════════════════════════════

# MaxRuntime sets a wall-clock budget — PyVRP keeps improving until time runs out.
# Unlike the MILP, it does not guarantee optimality but finds strong solutions fast.
print(f"\nSolving with PyVRP  (budget: {RUNTIME_SEC}s)...")
print("-" * 60)

t0     = time.time()
result = m.solve(stop=MaxRuntime(RUNTIME_SEC), display=True)
elapsed = time.time() - t0

print("-" * 60)
print(f"Solve time      : {elapsed:.2f} seconds")
print(f"Feasible        : {result.is_feasible()}")
print(f"Best cost found : {result.cost()}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PRINT ROUTE DETAILS
# ══════════════════════════════════════════════════════════════════════════════

solution = result.best
routes   = solution.routes()

print("\n── Route Details ───────────────────────────────────────")

all_visited       = []
manual_total_dist = 0

for idx, route in enumerate(routes):
    visits = route.visits()   # 0-based client indices, maps to subset_nodes positions

    route_demand = sum(subset_nodes[c]["demand"] for c in visits)
    route_dist   = route.distance()

    manual_total_dist += route_dist
    all_visited.extend(visits)

    labels = " -> ".join(["Depot"] + [f"C{c}" for c in visits] + ["Depot"])
    print(f"\n  Vehicle {idx+1}: {labels}")
    print(f"  Load     : {route_demand:.0f} / {capacity}")
    print(f"  Distance : {route_dist}")

print(f"\n  PyVRP reported cost      : {result.cost()}")
print(f"  Manually summed distance : {manual_total_dist}")
print(f"  Match?                   : {result.cost() == manual_total_dist}")
print(f"  Vehicles used            : {len(routes)} / {NUM_VEHICLES}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

expected  = set(range(1, NUM_CUSTOMERS + 1))
visited   = set(all_visited)
all_once  = (visited == expected and len(all_visited) == len(set(all_visited)))

capacity_ok = True
for idx, route in enumerate(routes):
    visits       = route.visits()
    route_demand = sum(subset_nodes[c]["demand"] for c in visits)
    if route_demand > capacity:
        capacity_ok = False
        print(f"  Vehicle {idx+1} exceeds capacity: {route_demand} > {capacity}")

print(f"\n── Validation ──────────────────────────────────────────")
print(f"  All customers visited exactly once? {all_once}")
print(f"  All routes within capacity?         {capacity_ok}")
print(f"  PyVRP feasibility check?            {result.is_feasible()}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

colours = ["#E63946", "#2A9D8F", "#E9C46A", "#F4A261", "#264653",
           "#A8DADC", "#457B9D", "#1D3557", "#F1FAEE", "#E9C46A"]
coords  = [(n['x'], n['y']) for n in subset_nodes]

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#F8F9FA")

for idx, route in enumerate(routes):
    visits = route.visits()
    path   = [0] + list(visits) + [0]
    col    = colours[idx % len(colours)]
    for step in range(len(path) - 1):
        a, b = coords[path[step]], coords[path[step + 1]]
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="->", color=col, lw=2))

for i in range(1, len(subset_nodes)):
    ax.scatter(*coords[i], s=180, color="#264653", zorder=5)
    ax.annotate(f"C{i}", coords[i], textcoords="offset points",
                xytext=(6, 5), fontsize=8, fontweight="bold", color="#264653")

ax.scatter(*coords[0], s=350, marker="*", color="#E63946", zorder=6)
ax.annotate("Depot", coords[0], textcoords="offset points",
            xytext=(8, 6), fontsize=10, fontweight="bold", color="#E63946")

ax.set_title(
    f"CVRP — PyVRP Solution  |  cost = {result.cost()}  |  "
    f"{len(routes)} routes  |  {NUM_CUSTOMERS} customers",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("outputs/pyvrp_routes.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nRoute map saved -> outputs/pyvrp_routes.png")
