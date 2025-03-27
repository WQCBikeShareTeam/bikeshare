#!/usr/bin/env python3
"""
Cluster Route Optimizer using SciPy

This program loads real bikeshare data from exported JSON files and uses
scipy optimization to find the optimal route between clusters for bike redistribution.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple
from datetime import datetime
from scipy.optimize import minimize, LinearConstraint, Bounds
import argparse

# cost scale, how much cost function affects
COST_SCALE = 0.1


def load_json_data(filename: str, directory: str = "exports") -> dict:
    """Load JSON data from file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, directory, filename)
    
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}


def setup_scipy_optimization(clusters, cost_matrix):
    """
    Set up the optimization problem for SciPy
    
    Args:
        clusters: List of cluster data with benefits and bike adjustments
        cost_matrix: Travel costs between clusters
        
    Returns:
        Dictionary with problem components for SciPy optimize
    """
    n_clusters = len(clusters)
    
    # Extract payout benefits and required bikes
    benefits = np.array([cluster['total_payout_benefit'] for cluster in clusters])
    bike_adjustments = np.array([abs(cluster['total_adjustment']) for cluster in clusters])
    
    # Create cost matrix as numpy array (assuming symmetric costs)
    costs = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                cluster_i = f"Cluster_{i}"
                cluster_j = f"Cluster_{j}"
                if cluster_i in cost_matrix and cluster_j in cost_matrix[cluster_i]:
                    costs[i, j] = cost_matrix[cluster_i][cluster_j]
    
    # Define the objective function: maximize benefits - costs
    # Since SciPy minimizes, we'll negate the objective
    def objective(x):
        # x is a binary vector, 1 if cluster is visited, 0 otherwise
        benefit_term = np.dot(benefits, x)
        
        cost_term = 0
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    cost_term += costs[i, j] * x[i] * x[j]
        
        # Apply the cost scale factor
        return -(benefit_term - COST_SCALE * cost_term)

    
    # Initial point - all clusters unselected
    x0 = np.zeros(n_clusters)
    
    # Define truck capacity constraints
    # 1. Total bikes <= truck capacity
    # 2. Total bikes >= 0 (implicitly satisfied)
    A = bike_adjustments.reshape(1, -1)  # Coefficient matrix for constraints
    lb = np.array([0])      # Lower bound
    ub = np.array([80])     # Upper bound (truck capacity)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Define bounds for each variable (binary: 0 or 1)
    variable_bounds = Bounds(np.zeros(n_clusters), np.ones(n_clusters))
    
    return {
        'objective': objective,
        'x0': x0,
        'constraints': constraints,
        'bounds': variable_bounds,
        'n_clusters': n_clusters,
        'benefits': benefits,
        'costs': costs,
        'bike_adjustments': bike_adjustments
    }


def solve_binary_optimization(problem):
    """
    Solve the binary optimization problem using SciPy
    
    Args:
        problem: Dictionary with problem components from setup_scipy_optimization
        
    Returns:
        Dictionary with solution results
    """
    # Since SciPy doesn't directly handle binary constraints, we'll use a trick:
    # 1. Solve the continuous relaxation (variables between 0 and 1)
    # 2. Round the result to binary values
    # 3. Check which binary solutions are feasible and pick the best
    
    print("Solving optimization problem with SciPy...")
    
    # Solve relaxed problem (continuous variables between 0 and 1)
    start_time = time.time()
    result = minimize(
        problem['objective'],
        problem['x0'],
        method='SLSQP',  
        bounds=problem['bounds'],
        constraints=problem['constraints'],
        options={'maxiter': 1000, 'disp': True}
    )
    solve_time = time.time() - start_time
    
    print(f"Optimization completed in {solve_time:.4f} seconds")
    print(f"Success: {result.success}")
    print(f"Status: {result.message}")
    print(f"Function evaluations: {result.nfev}")
    
    # If the relaxed solution isn't already binary (exactly 0 or 1),
    # we need to find the best binary solution
    x_relaxed = result.x
    print(f"Relaxed solution: {x_relaxed}")
    
    # Check if solution already binary (with tolerance)
    is_binary = all(abs(x - round(x)) < 1e-6 for x in x_relaxed)
    
    if is_binary:
        # If already binary, just round to exact 0/1 values
        x_binary = np.round(x_relaxed).astype(int)
        obj_value = -problem['objective'](x_binary)  # Negate again to get true value
        
        selected_clusters = [f"Cluster_{i}" for i, selected in enumerate(x_binary) if selected == 1]
        
        return {
            'selected_clusters': selected_clusters,
            'x': x_binary,
            'objective_value': obj_value,
            'solve_time': solve_time,
            'status': 'Relaxed solution is binary'
        }
    else:
        # We need to check near-binary solutions
        print("Relaxed solution is not binary. Finding best binary solution...")
        
        # Sort variables by their values (closest to 1 first)
        sorted_indices = np.argsort(-x_relaxed)
        
        best_binary = None
        best_objective = float('-inf')
        total_bikes = problem['bike_adjustments']
        truck_capacity = 80
        
        # Try different combinations, starting with most promising variables
        for n_selected in range(problem['n_clusters'] + 1):
            # Select the top n_selected variables
            binary_solution = np.zeros(problem['n_clusters'])
            selected_indices = sorted_indices[:n_selected]
            binary_solution[selected_indices] = 1
            
            # Check if feasible (within bike capacity)
            bikes_needed = np.sum(total_bikes * binary_solution)
            if bikes_needed <= truck_capacity:
                # Calculate objective value
                obj_value = -problem['objective'](binary_solution)  # Negate to get true value
                
                if obj_value > best_objective:
                    best_objective = obj_value
                    best_binary = binary_solution.copy()
        
        if best_binary is not None:
            selected_clusters = [f"Cluster_{i}" for i, selected in enumerate(best_binary) if selected == 1]
            
            return {
                'selected_clusters': selected_clusters,
                'x': best_binary,
                'objective_value': best_objective,
                'solve_time': solve_time,
                'status': 'Found best binary solution'
            }
        else:
            return {
                'selected_clusters': [],
                'x': np.zeros(problem['n_clusters']),
                'objective_value': 0,
                'solve_time': solve_time,
                'status': 'No feasible binary solution found'
            }


def calculate_route_metrics(selected_clusters, clusters, cost_matrix):
    """
    Calculate metrics for the selected route
    
    Args:
        selected_clusters: List of selected cluster names
        clusters: List of cluster data
        cost_matrix: Travel costs between clusters
        
    Returns:
        Dictionary with calculated metrics
    """
    total_payout = 0
    total_bikes = 0
    total_stations = 0
    
    # Calculate payouts and bike adjustments
    for cluster_name in selected_clusters:
        # Extract cluster index from name
        cluster_idx = int(cluster_name.split('_')[1])
        
        # Add payout benefit
        total_payout += clusters[cluster_idx]['total_payout_benefit']
        
        # Add bike adjustments
        total_bikes += abs(clusters[cluster_idx]['total_adjustment'])
        
        # Count stations
        total_stations += len(clusters[cluster_idx]['stations'])
    
    # Calculate travel costs
    total_cost = 0
    for i, cluster_from in enumerate(selected_clusters):
        for j, cluster_to in enumerate(selected_clusters):
            if i != j:
                if cluster_from in cost_matrix and cluster_to in cost_matrix[cluster_from]:
                    total_cost += cost_matrix[cluster_from][cluster_to]
    
    # Since we double-counted each path (A->B and B->A), divide by 2
    total_cost /= 2
    
    return {
        'total_payout': total_payout,
        'total_cost': total_cost,
        'net_benefit': total_payout - total_cost,
        'total_bikes': total_bikes,
        'stations_count': total_stations
    }


def visualize_route(selected_clusters, clusters, cost_matrix, title="Best Route", filename="best_route.png"):
    """
    Visualize the best route (optimal ordering) among the selected clusters.
    This function computes a TSP route (using a simple heuristic) and draws only that path.
    
    Args:
        selected_clusters: List of selected cluster names (e.g. ['Cluster_1', 'Cluster_7'])
        clusters: List of all clusters with their data
        cost_matrix: Dictionary with travel costs between clusters
        title: Title for the visualization
        filename: Filename to save the visualization
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import os

    # Create a complete graph over the selected clusters.
    # We'll label nodes using their names.
    G = nx.complete_graph(len(selected_clusters))
    mapping = {i: selected_clusters[i] for i in range(len(selected_clusters))}
    G = nx.relabel_nodes(G, mapping)

    # Set positions for nodes from the clusters' center coordinates.
    pos = {}
    for cluster_name in selected_clusters:
        cluster_idx = int(cluster_name.split('_')[1])
        pos[cluster_name] = tuple(clusters[cluster_idx]['center'])
    
    # Set edge weights in the complete graph using the cost matrix.
    for i in range(len(selected_clusters)):
        for j in range(i + 1, len(selected_clusters)):
            n1 = selected_clusters[i]
            n2 = selected_clusters[j]
            weight = cost_matrix.get(n1, {}).get(n2, 0)
            G[n1][n2]['weight'] = weight
            G[n2][n1]['weight'] = weight

    # Compute a TSP route (this returns a cycle).
    try:
        tsp_route = nx.approximation.traveling_salesman_problem(G, weight='weight', cycle=True)
        # Remove the last node if it repeats the first to get a simple path.
        if tsp_route[0] == tsp_route[-1]:
            tsp_route = tsp_route[:-1]
    except Exception as e:
        print(f"Error computing TSP route: {e}")
        tsp_route = selected_clusters  # Fallback: use the order of selection

    # Create a directed graph H to represent the path.
    H = nx.DiGraph()
    for i in range(len(tsp_route) - 1):
        n1 = tsp_route[i]
        n2 = tsp_route[i+1]
        weight = cost_matrix.get(n1, {}).get(n2, 0)
        H.add_edge(n1, n2, weight=weight)
    for node in tsp_route:
        H.add_node(node, pos=pos[node])
    
    # Draw the route.
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(H, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(H, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(H, pos, arrowstyle='->', arrowsize=15, edge_color='gray', width=2)
    edge_labels = nx.get_edge_attributes(H, 'weight')
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    
    # Save the figure.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    export_dir = os.path.join(base_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)
    plt.savefig(os.path.join(export_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Best route visualization saved to {filename}")
    plt.close()


def main():
    """Main function to run the optimization"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize bike redistribution route using SciPy')
    parser.add_argument('--reduced-clusters', default='reduced_clusters.json', help='Reduced clusters JSON file')
    parser.add_argument('--travel-costs', default='cluster_travel_costs.json', help='Travel costs JSON file')
    
    args = parser.parse_args()
    
    print("=== Bikeshare Cluster Route Optimizer (SciPy) ===")
    print(f"Loading data from {args.reduced_clusters} and {args.travel_costs}")
    
    # Load clusters data
    clusters_data = load_json_data(args.reduced_clusters)
    clusters = []
    
    # Extract clusters from the JSON data
    if 'clusters' in clusters_data:
        clusters = clusters_data['clusters']
    
    # Load travel costs data
    travel_costs_data = load_json_data(args.travel_costs)
    cost_matrix = {}
    
    # Extract cost matrix
    if 'cost_matrix' in travel_costs_data:
        cost_matrix = travel_costs_data['cost_matrix']
    
    print(f"Loaded {len(clusters)} clusters")
    print(f"Loaded cost matrix with {len(cost_matrix)} entries")
    
    # Print a summary of the clusters
    print("\nCluster Summary:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {len(cluster['stations'])} stations")
        print(f"  Total adjustment: {cluster['total_adjustment']} bikes")
        print(f"  Total benefit: ${cluster['total_payout_benefit']:.2f}")
        print(f"  Center location: {cluster['center']}")
        print("-" * 30)
    
    # Set up and solve the optimization problem with SciPy
    print("\n=== Running SciPy Optimization ===")
    
    # Set up the problem
    start_time = time.time()
    problem = setup_scipy_optimization(clusters, cost_matrix)
    setup_time = time.time() - start_time
    
    print(f"Problem setup complete in {setup_time:.4f} seconds")
    print(f"Number of clusters: {problem['n_clusters']}")
    print(f"Benefits: {problem['benefits']}")
    print(f"Bike adjustments: {problem['bike_adjustments']}")
    
    # Solve the problem
    solution = solve_binary_optimization(problem)
    
    # Display results
    if solution['selected_clusters']:
        print(f"\nSelected clusters: {', '.join(solution['selected_clusters'])}")
        print(f"Objective value: {solution['objective_value']:.4f}")
        
        # Calculate route metrics
        metrics = calculate_route_metrics(solution['selected_clusters'], clusters, cost_matrix)
        print(f"Total payout: ${metrics['total_payout']:.2f}")
        print(f"Total travel cost: ${metrics['total_cost']:.2f}")
        print(f"Net benefit: ${metrics['net_benefit']:.2f}")
        print(f"Total bikes to adjust: {metrics['total_bikes']}")
        print(f"Total stations: {metrics['stations_count']}")
        
        # Visualize the route
        visualize_route(
            solution['selected_clusters'],
            clusters,
            cost_matrix,
            "SciPy Optimizer Route",
            "scipy_optimizer_route.png"
        )
    else:
        print("No solution found")
    
    print("\nOptimization complete")


if __name__ == "__main__":
    main() 