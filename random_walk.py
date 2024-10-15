import networkx as nx
import random
import numpy as np
import powerlaw
import time
from scipy.stats import pearsonr
from collections import Counter
from collections import deque


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


def single_shortest_path_length(G, source, cutoff):
   
    adj = G._adj
    firstlevel = [source]
    seen = set(firstlevel)
    nextlevel = firstlevel
    level = 0
    n = len(adj)

    while nextlevel and cutoff > level:
        level += 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in adj[v]:
                if w not in seen:
                    if level >= cutoff:
                        return source, w

                    seen.add(w)
                    nextlevel.append(w)
            if len(seen) == n:
                return


def find_pair_with_distance_greater_than_5(graph):
    # Iterate over each node in the graph
    for node in graph.nodes():
        # Perform a breadth-first search (BFS) from this node
        result = single_shortest_path_length(graph, node, 5)

        if result is None:
            continue
        
        return result
    return None


def random_steps(graph: nx.Graph, start: int, step_size: int) -> int:

    current = start

    for _ in range(step_size):
        neighbors = list(graph.neighbors(current))
        if len(neighbors) > 0:
            current = neighbors[random.randint(0, len(neighbors)-1)]

    return current


# random walk -> add edges around hubs after half total nodes -> add edges between low degree neighbors of hubs after half total nodes

def random_walk_sequence_v1(graph: nx.Graph, number_of_nodes: int, probability_step_length_one: float,
                          n_marked: float, max_top_nodes: float, edges_added_per_walk_per_top_node: float, n_distant_edges:float) -> nx.Graph:
    
    n_nodes = graph.number_of_nodes()

    for _ in range(0, number_of_nodes):
        
        start = random.randint(1, n_nodes-1)
        current = start

        graph.add_node(n_nodes)
    
        marked = [start]

        step_size = assign_value(probability_step_length_one)

        for _ in range(int(n_marked)-1): 
            current = random_steps(graph, current, step_size)
        
            marked.append(current)

        if assign_value(n_marked - int(n_marked)) == 1:
            current = random_steps(graph, current, step_size)
        
            marked.append(current)


        for v in marked:
            graph.add_edge(v, n_nodes)

        n_nodes += 1


        if n_nodes > number_of_nodes/2:

            sorted_nodes_by_degree = sorted(graph.degree(), key=lambda x: x[1], reverse=True)

            top_x = int(max_top_nodes)

            if assign_value(max_top_nodes - int(max_top_nodes)) == 1:
                top_x += 1

            top_nodes = sorted_nodes_by_degree[:top_x]

            for node in top_nodes:
                
                for _ in range(int(edges_added_per_walk_per_top_node)):

                    neighbors = list(graph.neighbors(node[0]))
                    for u in neighbors:
                    
                        v = neighbors[random.randint(0, len(neighbors) - 1)]
                        
                        if nx.shortest_path_length(graph, u, v) == 2:
                            graph.add_edge(v, u)
                            break

                if assign_value(edges_added_per_walk_per_top_node - int(edges_added_per_walk_per_top_node)) == 1:
                    
                    neighbors = list(graph.neighbors(node[0]))
                    for u in neighbors:
                    
                        v = neighbors[random.randint(0, len(neighbors) - 1)]
                        
                        if nx.shortest_path_length(graph, u, v) == 2:
                            graph.add_edge(v, u)
                            break      


            for _ in range(int(n_distant_edges)):
                
                while True:
                    top_nodes = sorted_nodes_by_degree[:n_nodes//7]
                    
                    hub1 = top_nodes[random.randint(0, len(top_nodes)-1)]
                    hub2 = top_nodes[random.randint(0, len(top_nodes)-1)]

                    if hub1 != hub2:

                        u = sorted(graph.neighbors(hub1[0]), key=lambda x: len(list(graph.neighbors(x))))[0]
                        v = sorted(graph.neighbors(hub2[0]), key=lambda x: len(list(graph.neighbors(x))))[0]

                        if nx.shortest_path_length(graph, u, v) < 3:
                            continue

                        #print(u, len(list(graph.neighbors(u))), v, len(list(graph.neighbors(v))))

                        graph.add_edge(u, v)

                        break     
        
    return graph

def random_walk_sequence_v2(graph: nx.Graph, number_of_nodes: int, probability_step_length_one: float,
                          n_marked: float, n_random_edges: int, decay_coefficient: float) -> nx.Graph:
    
    n_nodes = graph.number_of_nodes()

    for _ in range(0, number_of_nodes):
        
        start = random.randint(1, n_nodes-1)
        current = start

        graph.add_node(n_nodes)
    
        marked = [start]

        step_size = assign_value(probability_step_length_one)

        for _ in range(int(n_marked)-1): 
            current = random_steps(graph, current, step_size)
        
            marked.append(current)

        if assign_value(n_marked - int(n_marked)) == 1:
            current = random_steps(graph, current, step_size)
        
            marked.append(current)


        for v in marked:
            graph.add_edge(v, n_nodes)

        n_nodes += 1


        for _ in range(n_random_edges):
            
            while True:

                u = random.randint(1, n_nodes-1)
                v = random.randint(1, n_nodes-1)

                distance = nx.shortest_path_length(graph, u, v)

                if distance > 0 and assign_value(1/distance**decay_coefficient) == 1:
                    graph.add_edge(u, v)
                    break

        
    return graph


def random_walk_sequence(graph: nx.Graph, number_of_nodes: int, probability_step_length_one: float,
                          n_marked: float, n_random_edges: int, decay_coefficient: float) -> nx.Graph:
    
    n_nodes = graph.number_of_nodes()

    for _ in range(0, number_of_nodes):
        
        start = random.randint(1, n_nodes-1)
        current = start

        graph.add_node(n_nodes)
    
        marked = [start]

        step_size = assign_value(probability_step_length_one)

        for _ in range(int(n_marked)-1): 
            current = random_steps(graph, current, step_size)
        
            marked.append(current)

        if assign_value(n_marked - int(n_marked)) == 1:
            current = random_steps(graph, current, step_size)
        
            marked.append(current)


        for v in marked:
            graph.add_edge(v, n_nodes)

        n_nodes += 1


        s = 1

        average_degree = (graph.number_of_edges() * 2)/graph.number_of_nodes()

        k = 0
        diameter = 0

        print(average_degree)
        print(graph.number_of_nodes())

        while s + average_degree * (average_degree-1)**k  < graph.number_of_nodes():

            s += average_degree * (average_degree-1)**k 
            k += 1
            diameter += 2

        
        
        print("diameter: ", diameter)


        inverse_sum = 0

        for i in range(2, diameter + 1):
            inverse_sum += 1/(i**2)


        normal_constant = 1/inverse_sum

        ranges = []
        current_inverse_sum = 0
        for i in range(2, diameter+1):
            current_inverse_sum += 1/(i**2)
            ranges.append(normal_constant* current_inverse_sum)



        random_variable = random.randint(0, 101)/100
        number_of_steps = 0

        while number_of_steps < len(ranges) and ranges[number_of_steps] < random_variable:
            number_of_steps += 1

        number_of_steps += 2
        print("normal constant: ", normal_constant)
        print("random variable: ", random_variable)
        print("number of steps: ", number_of_steps)
        print("ranges: ", ranges)

        # bfs 
        source = random.randint(1, n_nodes-1)

        marked = []
        
        queue = deque([source])
        levels = {source: 0}
        current_level = 0

        target = source

        while current_level < number_of_steps:

            if len(queue) == 0:
                break

            node = queue.popleft()
            current_level = levels[node]
            #print("level: ", current_level, number_of_steps, node)
            
            # Explore all the neighbors of the current node
            for neighbor in graph.neighbors(node):
                if neighbor not in levels:  # If the neighbor hasn't been visited
                    queue.append(neighbor)
                    levels[neighbor] = current_level + 1  # Mark the level of the neighbor

                    if levels[neighbor] == number_of_steps:
                        target = neighbor
                        break


        #print("distance: ", nx.shortest_path_length(graph, source, target))
        graph.add_edge(source, target)
        print(source, target)
        
        print("prob: ", random_variable)
        print("steps: ", number_of_steps)



    return graph


def create_point(filepath: str, n_nodes: int, probability_step_length_one: float,
                  n_marked: float, n_random_edges: int, decay_coefficient: float):
    
    start_time = time.time()  # Record the start time

    
    G = nx.Graph()

    G = nx.cycle_graph(10)

    G = random_walk_sequence(G, n_nodes, probability_step_length_one, n_marked, n_random_edges, decay_coefficient)

    #G = random_walk_sequence_v1(G, n_nodes, probability_step_length_one, n_marked, 5, 1, n_random_edges)

    degrees = [degree for _, degree in G.degree()]

    bin_edges = np.arange(0, max(degrees) + 100, 100)

    hist, bin_edges = np.histogram(degrees, bins=bin_edges)

    # Print the count of nodes in each interval
    for i in range(len(hist)):
        print(f"Nodes with degrees between {bin_edges[i]} and {bin_edges[i+1]}: {hist[i]}")


    # Fit the degree distribution to a power law
    fit = powerlaw.Fit(degrees)

    alpha = fit.power_law.alpha
    

    average_clust = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    path_time = time.time()
    average_shortest_path = nx.average_shortest_path_length(G)
    path_time_end = time.time()
    print(f"\n\nTime spent: {(path_time_end - path_time):.4f} seconds\n\n")

    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n\nTime spent: {elapsed_time:.4f} seconds\n\n")


    print("average clustering: ", average_clust)
    print("transitivity: ", transitivity)
    print("average shortest path length: ", average_shortest_path)
    print(f"Estimated power-law exponent: {alpha}")
    print("KS: ", fit.power_law.KS())
    print("number of edges: ", G.number_of_edges())


    degree_ranges = [(0, 100), (100, 200), (200, 300), (300, 1000)]

    for min_degree, max_degree in degree_ranges:
        num_edges_in_range = edges_in_degree_range(G, min_degree, max_degree)
        fraction = num_edges_in_range / G.number_of_edges()
        print(f"Fraction of edges in degree range [{min_degree}, {max_degree}]: {fraction:.4f}")

    
    degrees = [degree for node, degree in G.degree()]

    # Step 2: Count the occurrences of each degree (degree distribution)
    degree_counts = Counter(degrees)

    # Extract degrees and their corresponding counts
    unique_degrees = np.array(list(degree_counts.keys()))
    counts = np.array(list(degree_counts.values()))

    # Step 3: Apply logarithm (using natural log)
    log_degrees = np.log(unique_degrees)
    log_counts = np.log(counts)

    # Step 4: Calculate the Pearson correlation coefficient
    r, _ = pearsonr(log_degrees, log_counts)

    

    with open(filepath, 'a') as file:

        file.write(str(probability_step_length_one) + " " + str(n_nodes) + " " + str(n_marked) + " " + str(n_random_edges) + " " + str(decay_coefficient) + "\n")
        
        file.write(f"probability_step_length_one: {probability_step_length_one}\n")
        #file.write(f"max_top_nodes: {max_top_nodes}\n")
        file.write(f"n_marked: {n_marked}\n")
        #file.write(f"distant edges: {n_distant_edges}\n")
        file.write(f"n nodes: {n_nodes}\n")
        file.write(f"n random edges: {n_random_edges}\n")
        file.write(f"decay coefficient: {decay_coefficient}\n")
        #file.write(f"edges_added_per_walk_per_top_node: {edges_added_per_walk_per_top_node}\n")
        file.write("average clustering: {0}\n".format(average_clust))
        file.write("transitivity: {0}\n".format(transitivity))
        file.write("average shortest path length: {0}\n".format(average_shortest_path))
        file.write("number of edges: {0}\n".format(G.number_of_edges()))
        file.write("power-law exponent: {0}\n".format(alpha))
        file.write("r coefficient: {0}\n".format(r))
        file.write("KS: {0}\n\n\n".format(fit.power_law.KS()))


if __name__ == "__main__":

    create_point("algoritmo_ruben.txt", n_nodes = 100000, probability_step_length_one=1, n_marked=5, n_random_edges=1, decay_coefficient=2)
    #create_point("algoritmo_v1.txt", n_nodes = 50000, probability_step_length_one=1, n_marked=5, n_random_edges=1, decay_coefficient=2)