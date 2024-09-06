import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import Counter


def assign_value(p):
    outcomes = [1, 2]
    probabilities = [p, 1 - p]

    x = random.choices(outcomes, probabilities)[0]
    return x


def random_steps(graph: nx.Graph, start: int, step_size: int) -> int:

    current = start

    for _ in range(step_size):
        neighbors = list(graph.neighbors(current))
        if len(neighbors) > 0:
            current = neighbors[random.randint(0, len(neighbors)-1)]

    return current


def random_walk_sequence(graph: nx.Graph, number_of_networks: int, probability: float) -> nx.Graph:
    
    nodes = graph.number_of_nodes()

    

    for _ in range(0, number_of_networks):
        
        start = random.randint(1, nodes-1)
        current = start

        # novo vértice
        graph.add_node(nodes)

        
        marked = [start]

        step_size = assign_value(probability)

        for _ in range(9): 
            current = random_steps(graph, current, step_size)

            marked.append(current)


        for v in marked:
            graph.add_edge(v, nodes)

        # nx.draw(G)
        # plt.show()

        nodes += 1

    
    return graph





# Create an undirected graph



#G = random_walk_sequence(G, 20000)
x = []
y = []
z = []
w = []

for i in range(10):
    m = i / 10
    print(i)
    G = nx.Graph()
    #print(G.edges())

    # Add nodes
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Add edges
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)])

    G = random_walk_sequence(G, 4000, m)
    #print(G.edges())


    print("GRAFO FEITO")

    # degrees = [degree for node, degree in G.degree()]

    # # Step 2: Count the frequency of each degree
    # degree_count = Counter(degrees)

    # # Get the degree values and their corresponding frequencies
    # degree_values, degree_frequencies = zip(*degree_count.items())

    # # Step 3: Plot the degree distribution
    # plt.bar(degree_values, degree_frequencies)
    # plt.xlabel('Degree')
    # plt.ylabel('Number of Nodes')
    # plt.title('Degree Distribution')
    # plt.show()
    
    
    x.append(m)
    y.append(nx.average_clustering(G))
    z.append(nx.transitivity(G))
    w.append(nx.average_shortest_path_length(G))

print(y)
print(z)
plt.plot(x, y)
plt.show()

plt.plot(x, z)
plt.show()

plt.plot(x, w)
plt.show()

# clustering_coeff = nx.average_clustering(G)
# print("Average Clustering Coefficient:", clustering_coeff)

# # degree_centrality = nx.degree_centrality(G)
# # print("Degree Centrality:", degree_centrality)

# global_clustering_coefficient = nx.transitivity(G)
# print("Global Clustering Coefficient:", global_clustering_coefficient)

#pos = nx.spring_layout(G, iterations=10)
#nx.draw(G, pos, node_size=5, with_labels=False)
#plt.show()
