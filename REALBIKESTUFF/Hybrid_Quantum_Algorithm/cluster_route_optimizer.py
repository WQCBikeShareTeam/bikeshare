#!/usr/bin/env python3
"""
Cluster Route Optimizer

This program loads real bikeshare data from exported JSON files,
sets up the quadratic program for optimization and solves it using
available open-source solvers.
"""

import os
import json
import time
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple
from datetime import datetime

# Import Qiskit optimization libraries
try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer, SlsqpOptimizer, CobylaOptimizer
    QISKIT_AVAILABLE = True
except ImportError:
    print("Warning: Qiskit optimization libraries not found. Optimization will be disabled.")
    QISKIT_AVAILABLE = False


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


def setup_truck_route_optimization(clusters, cost_matrix):
    """
    Set up the truck route optimization problem using Qiskit QuadraticProgram
    
    Args:
        clusters: List of cluster data containing benefits and bike adjustments
        cost_matrix: Travel costs between clusters
        
    Returns:
        Qiskit QuadraticProgram instance
    """
    if not QISKIT_AVAILABLE:
        print("Cannot create QuadraticProgram - Qiskit optimization not available")
        return None
        
    # Initialize the optimization problem
    problem = QuadraticProgram("Cluster Route Optimization")
    
    # Create a cluster name list
    cluster_names = [f"Cluster_{i}" for i in range(len(clusters))]
    
    # Create dictionaries for payouts and required bikes
    payouts = {}
    required_bikes = {}
    
    for i, cluster in enumerate(clusters):
        cluster_name = f"Cluster_{i}"
        payouts[cluster_name] = cluster['total_payout_benefit']
        # Use absolute value for required bikes (treat both additions and removals as trips)
        required_bikes[cluster_name] = abs(cluster['total_adjustment'])
    
    # Add binary variables for each cluster (1 if visited, 0 if not)
    for cluster_name in cluster_names:
        problem.binary_var(name=cluster_name)
    
    # Add objective function: maximize profit (payout - cost)
    linear_terms = {}
    quadratic_terms = {}
    
    # Calculate linear terms for payouts
    for cluster_name in cluster_names:
        linear_terms[cluster_name] = payouts[cluster_name]
    
    # Calculate quadratic terms for costs between clusters
    for i, cluster_from in enumerate(cluster_names):
        for j, cluster_to in enumerate(cluster_names):
            if i != j:  # Don't include cost from a cluster to itself
                # Get the cost between these clusters
                if cluster_from in cost_matrix and cluster_to in cost_matrix[cluster_from]:
                    cost = cost_matrix[cluster_from][cluster_to]
                    # Add as negative cost (penalty) in the quadratic term
                    quadratic_terms[(cluster_from, cluster_to)] = -cost
    
    # Set the objective function
    problem.maximize(
        constant=0,
        linear=linear_terms,
        quadratic=quadratic_terms
    )
    
    # Truck constraints
    truck_capacity = 80  # Maximum bike capacity
    truck_min_bikes = 0  # Minimum bikes to handle
    
    # Add constraints for required bikes
    problem.linear_constraint(
        linear=required_bikes,
        sense='>=',
        rhs=truck_min_bikes,
        name='min_bikes'
    )
    
    problem.linear_constraint(
        linear=required_bikes,
        sense='<=',
        rhs=truck_capacity,
        name='max_bikes'
    )
    
    return problem


def solve_quadratic_program(problem):
    """
    Solve the quadratic program using available open-source solvers
    
    Args:
        problem: Quadratic program to solve
        
    Returns:
        Solution with selected clusters and objective value
    """
    if not QISKIT_AVAILABLE:
        print("Cannot solve - Qiskit optimization not available")
        return None
    
    print("Attempting to solve the quadratic program...")
    
    # Try different solvers in order of preference
    solvers = []
    
    # 1. NumPyMinimumEigensolver - classical exact solver
    try:
        exact_solver = NumPyMinimumEigensolver()
        eigen_optimizer = MinimumEigenOptimizer(exact_solver)
        solvers.append(("NumPyMinimumEigensolver", eigen_optimizer))
    except Exception as e:
        print(f"Could not initialize NumPyMinimumEigensolver: {e}")
    
    # 2. COBYLA - classical approximate solver
    try:
        cobyla_optimizer = CobylaOptimizer()
        solvers.append(("COBYLA", cobyla_optimizer))
    except Exception as e:
        print(f"Could not initialize CobylaOptimizer: {e}")
    
    # 3. SLSQP - classical approximate solver
    try:
        slsqp_optimizer = SlsqpOptimizer()
        solvers.append(("SLSQP", slsqp_optimizer))
    except Exception as e:
        print(f"Could not initialize SlsqpOptimizer: {e}")
    
    # Try solvers one by one
    for solver_name, solver in solvers:
        print(f"Attempting to solve with {solver_name}...")
        try:
            start_time = time.time()
            result = solver.solve(problem)
            solve_time = time.time() - start_time
            
            print(f"{solver_name} solution found in {solve_time:.4f} seconds")
            print(f"Objective value: {result.fval}")
            
            # Extract selected clusters
            selected_clusters = []
            for i, x in enumerate(result.x):
                if x > 0.5:  # Binary threshold
                    var_name = problem.variables[i].name
                    selected_clusters.append(var_name)
            
            return {
                'selected_clusters': selected_clusters,
                'objective_value': result.fval,
                'solver': solver_name
            }
            
        except Exception as e:
            print(f"Error with {solver_name}: {e}")
    
    print("All solvers failed. No solution found.")
    return None


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
    total_cost = 0
    total_bikes = 0
    total_stations = 0
    
    # Calculate total payout and bikes
    for cluster_name in selected_clusters:
        cluster_idx = int(cluster_name.split('_')[1])
        cluster = clusters[cluster_idx]
        
        total_payout += cluster['total_payout_benefit']
        total_bikes += abs(cluster['total_adjustment'])
        total_stations += len(cluster['stations'])
    
    # Calculate total travel cost
    for i, cluster_from in enumerate(selected_clusters):
        for j, cluster_to in enumerate(selected_clusters):
            if i != j:  # Don't count self-edges
                if cluster_from in cost_matrix and cluster_to in cost_matrix[cluster_from]:
                    total_cost += cost_matrix[cluster_from][cluster_to]
    
    # Divide by 2 because we counted each edge twice
    total_cost /= 2
    
    return {
        'total_payout': total_payout,
        'total_cost': total_cost,
        'net_benefit': total_payout - total_cost,
        'total_bikes': total_bikes,
        'stations_count': total_stations
    }


def visualize_route(selected_clusters, clusters, cost_matrix, title="Optimizer Route", filename="route.png"):
    """
    Visualize the selected route as a network graph
    
    Args:
        selected_clusters: List of selected cluster names
        clusters: List of cluster data
        cost_matrix: Travel costs between clusters
        title: Title for the visualization
        filename: Filename to save the visualization
    """
    # Create a graph
    G = nx.Graph()
    
    # Add nodes for selected clusters
    for cluster_name in selected_clusters:
        cluster_idx = int(cluster_name.split('_')[1])
        cluster = clusters[cluster_idx]
        
        # Add node with attributes
        G.add_node(
            cluster_name, 
            pos=tuple(cluster['center']), 
            bikes=abs(cluster['total_adjustment']),
            benefit=cluster['total_payout_benefit']
        )
    
    # Add edges between all selected clusters with costs as weights
    for i, cluster_from in enumerate(selected_clusters):
        for j, cluster_to in enumerate(selected_clusters):
            if i < j:  # Only add each edge once
                if cluster_from in cost_matrix and cluster_to in cost_matrix[cluster_from]:
                    cost = cost_matrix[cluster_from][cluster_to]
                    G.add_edge(cluster_from, cluster_to, weight=cost)
    
    # If there are no edges (only one node), add a self-loop
    if len(G.edges) == 0 and len(G.nodes) == 1:
        node = list(G.nodes)[0]
        G.add_edge(node, node, weight=0)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Get positions from node attributes
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes with size proportional to bikes and color to benefit
    node_sizes = [G.nodes[n]['bikes'] * 100 for n in G.nodes]
    node_colors = [G.nodes[n]['benefit'] for n in G.nodes]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis)
    
    # Draw edges with width proportional to inverse of cost
    edge_widths = [1 / (G.edges[e]['weight'] + 0.1) * 2 for e in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
    
    # Draw node labels
    labels = {}
    for n in G.nodes:
        cluster_idx = int(n.split('_')[1])
        bikes = abs(clusters[cluster_idx]['total_adjustment'])
        benefit = clusters[cluster_idx]['total_payout_benefit']
        labels[n] = f"{n}\n{bikes} bikes\n${benefit:.2f}"
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Draw edge labels
    edge_labels = {e: f"${G.edges[e]['weight']:.2f}" for e in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    
    # Create exports directory if it doesn't exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    export_dir = os.path.join(base_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Save figure
    plt.savefig(os.path.join(export_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Route visualization saved to {filename}")
    plt.close()


def main():
    """Main function to run the optimization"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize bike redistribution route')
    parser.add_argument('--reduced-clusters', default='reduced_clusters.json', help='Reduced clusters JSON file')
    parser.add_argument('--travel-costs', default='cluster_travel_costs.json', help='Travel costs JSON file')
    parser.add_argument('--setup-only', action='store_true', help='Only set up the problem without solving')
    
    args = parser.parse_args()
    
    print("=== Bikeshare Cluster Route Optimizer ===")
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
    
    # Set up the quadratic program
    print("\n=== Setting Up Quadratic Program ===")
    start_time = time.time()
    qiskit_problem = setup_truck_route_optimization(clusters, cost_matrix)
    setup_time = time.time() - start_time
    
    if qiskit_problem:
        print(f"Problem setup complete in {setup_time:.4f} seconds")
        print("Problem details:")
        print(qiskit_problem.prettyprint())
        
        # Export the problem to a file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        export_dir = os.path.join(base_dir, "exports")
        os.makedirs(export_dir, exist_ok=True)
        problem_path = os.path.join(export_dir, "optimization_problem.txt")
        
        with open(problem_path, 'w') as f:
            f.write(qiskit_problem.prettyprint())
        print(f"Exported problem formulation to {problem_path}")
        
        # Solve the problem if not in setup-only mode
        if not args.setup_only:
            print("\n=== Solving Optimization Problem ===")
            solution = solve_quadratic_program(qiskit_problem)
            
            if solution and solution['selected_clusters']:
                print(f"\nSelected clusters: {', '.join(solution['selected_clusters'])}")
                print(f"Objective value: {solution['objective_value']:.4f}")
                print(f"Solver used: {solution['solver']}")
                
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
                    f"{solution['solver']} Optimizer Route",
                    f"{solution['solver'].lower()}_optimizer_route.png"
                )
            else:
                print("No solution found or no clusters selected.")
                print("Try using the scipy_route_optimizer.py script as an alternative.")
        else:
            print("\nSetup-only mode. Skipping optimization.")
            print("To solve this problem, run without the --setup-only flag or use scipy_route_optimizer.py.")
    else:
        print("Failed to set up the optimization problem. Please check if Qiskit optimization is installed.")
    
    print("\nProcess complete")


if __name__ == "__main__":
    main() 