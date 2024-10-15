import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import Counter
import numpy as np
import powerlaw

def edges_in_degree_range(G, min_degree, max_degree):
    # Find nodes with degrees in the specified range
    nodes_in_range = [node for node, degree in G.degree() if min_degree <= degree <= max_degree]
    
    # Create a subgraph with only these nodes
    subgraph = G.subgraph(nodes_in_range)
    
    # Count the number of edges in this subgraph
    return subgraph.number_of_edges()

def assign_value(p):
    outcomes = [1, 2]
    probabilities = [p, 1 - p]

    x = random.choices(outcomes, probabilities)[0]
    return x


graph = nx.Graph()

graph.add_nodes_from([i for i in range(0, 4039)])
i = 0
with open('large_twitch_edges.csv', 'r') as file:
    for line in file:
        if i == 0:
            i = 1
            continue

        nodes = line.split(",")
        #print(nodes[0], nodes[1])
        graph.add_edge(int(nodes[0]), int(nodes[1]))

print("acabou")

print(nx.average_clustering(graph))
print(nx.transitivity(graph))
print(nx.average_shortest_path_length(graph))


degrees = [degree for _, degree in graph.degree()]

bin_edges = np.arange(0, max(degrees) + 100, 100)

hist, bin_edges = np.histogram(degrees, bins=bin_edges)

# Print the count of nodes in each interval
for i in range(len(hist)):
    print(f"Nodes with degrees between {bin_edges[i]} and {bin_edges[i+1]}: {hist[i]}")


# Plot the degree distribution with intervals
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Degree Distribution with Intervals')
plt.show()

degrees = [degree for _, degree in graph.degree()]

# Fit the degree distribution to a power law
fit = powerlaw.Fit(degrees)

# Get the power-law exponent
alpha = fit.power_law.alpha
print(f"Estimated power-law exponent: {alpha}")
print(fit.power_law.KS())

min_degree = 0
max_degree = 100

degree_ranges = [(0, 100), (100, 200), (200, 300), (300, 1000)]

for min_degree, max_degree in degree_ranges:
    num_edges_in_range = edges_in_degree_range(graph, min_degree, max_degree)
    fraction = num_edges_in_range / graph.number_of_edges()
    print(f"Fraction of edges in degree range [{min_degree}, {max_degree}]: {fraction:.4f}")