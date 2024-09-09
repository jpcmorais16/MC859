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


def random_walk_sequence(graph: nx.Graph, number_of_nodes: int, probability: float, n_marked: int) -> nx.Graph:
    
    nodes = graph.number_of_nodes()

    for _ in range(0, number_of_nodes):
        
        start = random.randint(1, nodes-1)
        current = start

        graph.add_node(nodes)
    
        marked = [start]

        step_size = assign_value(probability)

        for _ in range(n_marked - 1): 
            current = random_steps(graph, current, step_size)

            marked.append(current)


        for v in marked:
            graph.add_edge(v, nodes)

        nodes += 1

    
    return graph



x = []
y = []
z = []
w = []

for i in range(10):
    m = i / 10

    print(i)

    G = nx.Graph()

    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)])

    G = random_walk_sequence(G, 4000, m, 2)

    
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

