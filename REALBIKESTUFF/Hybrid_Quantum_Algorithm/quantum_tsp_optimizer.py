"""
quantum_tsp_optimizer.py

This file loads reduced cluster data from the bikeshare optimization pipeline,
filters out clusters that do not provide a significant benefit, computes an optimal 
route (using a brute-force TSP solver as a placeholder for a quantum-inspired optimization)
among the remaining cluster centers, and visualizes the path.

Assumptions for travel cost estimation:
 - Average speed is 25 km/h (including stops). TODO: switch with actual path time, google api
 - Driver wage is $17.50 per hour.
 - Wear and tear cost is $0.15 per km.
"""

import os
import json
import math
import itertools
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------
# Utility Functions
# ----------------------------
def calculate_distance(coord1, coord2):
    """
    Approximate the distance (in km) between two (lat, lon) points using a simple Euclidean formula.
    (1° latitude ≈ 111 km; 1° longitude ≈ 81 km at Toronto's latitude)
    """
    lat_diff = (coord1[0] - coord2[0]) * 111
    lon_diff = (coord1[1] - coord2[1]) * 81
    return math.sqrt(lat_diff**2 + lon_diff**2)

def load_reduced_clusters(filename="reduced_clusters.json"):
    """
    Load reduced clusters JSON from the exports directory.
    """
    EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    filepath = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Reduced clusters file not found at {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("clusters", [])

# ----------------------------
# TSP Solver (Brute-Force Quantum-Inspired)
# ----------------------------
def solve_tsp_bruteforce(points):
    """
    Solve the Traveling Salesman Problem for a small set of points using brute force.
    'points' is a list of (lat, lon) tuples.
    Returns the best route (list of indices) and the minimal total distance.
    """
    n = len(points)
    if n == 0:
        return [], 0
    best_route = None
    best_distance = float("inf")
    # Fix starting point (index 0) and try all permutations for the remaining points
    for perm in itertools.permutations(range(1, n)):
        route = [0] + list(perm)
        total_distance = 0
        for i in range(n - 1):
            total_distance += calculate_distance(points[route[i]], points[route[i + 1]])
        # Complete the cycle by returning to the starting point
        total_distance += calculate_distance(points[route[-1]], points[route[0]])
        if total_distance < best_distance:
            best_distance = total_distance
            best_route = route
    return best_route, best_distance

# ----------------------------
# Visualization Function
# ----------------------------
def visualize_route(points, route, total_distance, gross_benefit, save_filename="tsp_route.png"):
    """
    Visualize the TSP route over the points using networkx.
    'points' is a list of (lat, lon) tuples.
    'route' is a list of indices indicating the visiting order.
    
    Travel cost estimation parameters:
      - Average speed: 50 km/h
      - Driver wage: $17.50 per hour
      - Wear and tear: $0.15 per km
      
    Net benefit is computed as: gross_benefit - travel_cost.
    """
    driver_wage = 17.50  # per hour
    average_speed = 50   # km/h
    wear_cost_rate = 0.15  # per km

    travel_time_hours = total_distance / average_speed
    travel_cost = travel_time_hours * driver_wage + total_distance * wear_cost_rate
    net_benefit = gross_benefit - travel_cost

    # Build graph for visualization
    G = nx.DiGraph()
    labels = {}
    for i, (lat, lon) in enumerate(points):
        G.add_node(i, pos=(lon, lat))  # using (lon, lat) for a natural x-y layout
        labels[i] = f"{i}"
    
    # Create edges for the route (closing the cycle)
    route_cycle = route + [route[0]]
    edges = [(route_cycle[i], route_cycle[i+1]) for i in range(len(route_cycle)-1)]
    G.add_edges_from(edges)
    
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='->', arrowsize=15)
    
    plt.title(
        f"TSP Route for Rebalancing Bikes\n"
        f"Total Distance: {total_distance:.2f} km\n"
        f"Gross Benefit: ${gross_benefit:.2f}\n"
        f"Travel Cost: ${travel_cost:.2f} (Time: {travel_time_hours:.2f} h)\n"
        f"Net Benefit: ${net_benefit:.2f}"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR, exist_ok=True)
    save_path = os.path.join(EXPORT_DIR, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Route visualization saved to {save_path}")

# ----------------------------
# Main Function
# ----------------------------
def main():
    try:
        clusters = load_reduced_clusters()
    except FileNotFoundError as e:
        print(e)
        return

    if not clusters:
        print("No clusters loaded from the file.")
        return

    # Filter out clusters with negligible benefit.
    # Here we assume a cluster is "worthwhile" if its total_payout_benefit is above a threshold.
    benefit_threshold = 1.0  # Adjust this threshold as needed
    filtered_clusters = [c for c in clusters if c.get("total_payout_benefit", 0) > benefit_threshold]

    if not filtered_clusters:
        print("No clusters with sufficient benefit found.")
        return

    # Extract centers and corresponding benefits from filtered clusters.
    points = []
    benefits = []
    for cluster in filtered_clusters:
        center = cluster.get("center")
        benefit = cluster.get("total_payout_benefit", 0)
        if center and isinstance(center, list) and len(center) == 2:
            points.append(tuple(center))
            benefits.append(benefit)
    
    if len(points) < 2:
        print("Not enough cluster centers to compute a route after filtering.")
        return

    print("Computing TSP route over the following cluster centers (after filtering):")
    for i, p in enumerate(points):
        print(f"Cluster {i}: {p}, Benefit: ${benefits[i]:.2f}")

    # Solve the TSP using brute force
    route, total_distance = solve_tsp_bruteforce(points)
    if route is None:
        print("Failed to compute a TSP route.")
        return

    # Compute the gross total benefit along the route.
    gross_benefit = sum(benefits[i] for i in route)
    print("\nOptimal route (visiting order of cluster indices):", route)
    print(f"Total travel distance: {total_distance:.2f} km")
    print(f"Gross Benefit along the route: ${gross_benefit:.2f}")

    # Visualize the route with travel cost and net benefit
    visualize_route(points, route, total_distance, gross_benefit)

if __name__ == "__main__":
    main()
