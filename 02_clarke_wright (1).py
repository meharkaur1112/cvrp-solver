"""
02_clarke_wright.py
===================
Capacitated Vehicle Routing Problem (CVRP) — Clarke-Wright Savings Heuristic
-----------------------------------------------------------------------------
Solves the CVRP using the Clarke-Wright Savings algorithm (Clarke & Wright, 1964),
a classical constructive heuristic that builds routes by greedily merging
individual customer spokes into shared vehicle routes.

This script is the second of three approaches benchmarked in this project:
    01_milp_cvrp.py      — Exact MILP formulation (PuLP / CBC)
    02_clarke_wright.py  — Clarke-Wright Savings heuristic  ← this file
    03_pyvrp_solver.py   — PyVRP metaheuristic solver

Algorithm:
    1. Start with every customer on its own dedicated spoke route
       (depot -> Ci -> depot). Always feasible, maximally wasteful.

    2. Compute the saving S[i,j] for merging any two routes:
           S[i,j] = dist(depot, i) + dist(depot, j) - dist(i, j)
       Large saving = customers far from depot but close to each other.

    3. Sort all (i,j) pairs by saving descending.

    4. Greedily merge routes when all feasibility checks pass:
           a. i and j are not already on the same route
           b. i is the last customer in its route
           c. j is the first customer in its route
           d. Merged demand does not exceed vehicle capacity

    This solution is also used as a warm-start incumbent for the MILP in
    01_milp_cvrp.py, tightening the branch-and-bound upper bound from node 0.

Key result (25 customers, capacity 200):
    Total distance : 196.0153
    Vehicles used  : 3 / 5  (matches mathematical minimum)
    Gap vs PyVRP   : 4.82%
    Runtime        : milliseconds

Dependencies:
    pip install matplotlib numpy

Usage:
    Update FILE_PATH and parameters below, then run:
        python 02_clarke_wright.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FILE_PATH     = "data/c101.txt"
NUM_CUSTOMERS = 25
NUM_VEHICLES  = 5     # maximum vehicles available


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARSE THE SOLOMON INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

with open(FILE_PATH, 'r') as f:
    lines = f.readlines()

lines = [l.strip() for l in lines if l.strip()]

instance_name = lines[0]

vehicle_line = lines[3].split()
num_vehicles = int(vehicle_line[0])
capacity     = int(vehicle_line[1])

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
# 3. BUILD SUBSET
# ══════════════════════════════════════════════════════════════════════════════

subset_nodes = [nodes[0]] + nodes[1:NUM_CUSTOMERS + 1]
demand       = [node['demand'] for node in subset_nodes]
total_demand = sum(demand[1:])
n            = NUM_CUSTOMERS

print(f"\nSubset               : 1 depot + {n} customers")
print(f"Total demand         : {total_demand}")
print(f"Min vehicles needed  : {math.ceil(total_demand / capacity)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISTANCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def dist(i, j):
    """Euclidean distance between nodes i and j by index in subset_nodes."""
    return math.sqrt(
        (subset_nodes[i]['x'] - subset_nodes[j]['x'])**2 +
        (subset_nodes[i]['y'] - subset_nodes[j]['y'])**2
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE SAVINGS
# ══════════════════════════════════════════════════════════════════════════════

# For every ordered pair (i, j) of customers, compute:
#   S[i,j] = dist(depot, i) + dist(depot, j) - dist(i, j)
# This quantifies how much distance is saved by linking i directly to j
# instead of returning to the depot between them.
savings = []
for i in range(1, n + 1):
    for j in range(1, n + 1):
        if i != j:
            s = dist(0, i) + dist(0, j) - dist(i, j)
            savings.append((s, i, j))

# Sort descending — highest-value merges are attempted first
savings.sort(reverse=True, key=lambda triple: triple[0])

print(f"\nSavings computed : {len(savings)} pairs")
print(f"Top saving       : {savings[0][0]:.4f}  (C{savings[0][1]} <-> C{savings[0][2]})")


# ══════════════════════════════════════════════════════════════════════════════
# 6. GREEDY MERGE
# ══════════════════════════════════════════════════════════════════════════════

t0 = time.time()

# Initialise: every customer on its own dedicated route (spoke solution)
# routes    — route_id -> ordered list of customer nodes on that route
# route_of  — customer -> its current route_id
# route_dem — route_id -> total demand on that route
# merged    — set of route_ids that have been absorbed into another route
routes    = {i: [i] for i in range(1, n + 1)}
route_of  = {i: i   for i in range(1, n + 1)}
route_dem = {i: demand[i] for i in range(1, n + 1)}
merged    = set()

for s, i, j in savings:

    # No beneficial merges remain once savings turn non-positive
    if s <= 0:
        break

    ri = route_of[i]
    rj = route_of[j]

    # Feasibility check 1: same route would create a closed subtour
    if ri == rj:
        continue

    # Feasibility check 2: i must be the last node in its route
    # (directly before the depot return leg) — only endpoints can be linked
    if routes[ri][-1] != i:
        continue

    # Feasibility check 3: j must be the first node in its route
    # (directly after the depot departure leg)
    if routes[rj][0] != j:
        continue

    # Feasibility check 4: combined demand must not exceed vehicle capacity
    if route_dem[ri] + route_dem[rj] > capacity:
        continue

    # All checks passed — merge route rj into route ri
    for c in routes[rj]:
        route_of[c] = ri

    routes[ri]    = routes[ri] + routes[rj]
    route_dem[ri] += route_dem[rj]
    merged.add(rj)

# Extract final routes (discard absorbed route slots)
final_routes = [
    routes[rid]
    for rid in sorted(routes.keys())
    if rid not in merged
]

elapsed = time.time() - t0
print(f"Solve time       : {elapsed*1000:.2f} ms")
print(f"Routes found     : {len(final_routes)} / {NUM_VEHICLES} vehicles")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRINT ROUTE DETAILS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Clarke-Wright Solution ─────────────────────────────────")

total_distance = 0.0
all_visited    = []

for idx, route in enumerate(final_routes):
    # Full route distance: depot -> first customer -> ... -> last customer -> depot
    route_dist = (
        dist(0, route[0]) +
        sum(dist(route[k], route[k+1]) for k in range(len(route)-1)) +
        dist(route[-1], 0)
    )
    route_demand = sum(demand[c] for c in route)
    total_distance += route_dist
    all_visited.extend(route)

    labels = " -> ".join(["Depot"] + [f"C{c}" for c in route] + ["Depot"])
    print(f"\n  Vehicle {idx+1}: {labels}")
    print(f"  Load     : {route_demand:.0f} / {capacity}")
    print(f"  Distance : {route_dist:.4f}")

print(f"\n  Total distance  : {total_distance:.4f}")
print(f"  Vehicles used   : {len(final_routes)} / {NUM_VEHICLES}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

expected = set(range(1, n + 1))
visited  = set(all_visited)
all_once = (visited == expected and len(all_visited) == len(set(all_visited)))
cap_ok   = all(sum(demand[c] for c in r) <= capacity for r in final_routes)

print(f"\n── Validation ──────────────────────────────────────────")
print(f"  All customers visited exactly once? {all_once}")
print(f"  All routes within capacity?         {cap_ok}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

palette = ["#E63946", "#2A9D8F", "#E9C46A", "#F4A261", "#264653"]
coords  = [(n_['x'], n_['y']) for n_ in subset_nodes]

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#F8F9FA")

for idx, route in enumerate(final_routes):
    col  = palette[idx % len(palette)]
    path = [0] + route + [0]
    for step in range(len(path) - 1):
        a, b = coords[path[step]], coords[path[step + 1]]
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.2))

for i in range(1, len(subset_nodes)):
    ax.scatter(*coords[i], s=180, color="#264653", zorder=5)
    ax.annotate(f"C{i}", coords[i], textcoords="offset points",
                xytext=(6, 5), fontsize=8, fontweight="bold", color="#264653")

ax.scatter(*coords[0], s=350, marker="*", color="#E63946", zorder=6)
ax.annotate("Depot", coords[0], textcoords="offset points",
            xytext=(8, 6), fontsize=10, fontweight="bold", color="#E63946")

patches = [mpatches.Patch(color=palette[i % len(palette)],
           label=f"Vehicle {i+1}") for i in range(len(final_routes))]
ax.legend(handles=patches, loc="upper right", fontsize=10)

ax.set_title(
    f"CVRP — Clarke-Wright Solution  |  cost = {total_distance:.4f}  |  "
    f"{len(final_routes)} routes  |  {NUM_CUSTOMERS} customers",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("outputs/cw_routes.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nRoute map saved -> outputs/cw_routes.png")
