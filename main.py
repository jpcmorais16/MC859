import multiprocessing
import os
import numpy as np
import random_walk as rw

def run_simulation(filename, prob_start, prob_end):
    for probability_step_length_one in np.arange(prob_start, prob_end, 0.1):   
        for n_nodes in range(1000, 101001, 10000):
            for n_marked in range(2, 33, 5):
                for n_random_edges in range(0, 6):
                    for decay_coefficient in np.arange(.0, 3.1, .5):
                        rw.create_point(filename, probability_step_length_one=probability_step_length_one,
                                        n_nodes=n_nodes, n_marked=n_marked, n_random_edges=n_random_edges, decay_coefficient=decay_coefficient)

if __name__ == "__main__":
    # Define the range for each process (from 0.1 to 1.0, split into 10 equal parts)
    ranges = [
        ('file1.txt', 0.1, 0.2),
        ('file2.txt', 0.2, 0.3),
        ('file3.txt', 0.3, 0.4),
        ('file4.txt', 0.4, 0.5),
        ('file5.txt', 0.5, 0.6),
        ('file6.txt', 0.6, 0.7),
        ('file7.txt', 0.7, 0.8),
        ('file8.txt', 0.8, 0.9),
        ('file9.txt', 0.9, 1.0),
        ('file10.txt', 0, 0.1)
    ]

    # Create and start processes
    processes = []
    for filename, prob_start, prob_end in ranges:
        pr = multiprocessing.Process(target=run_simulation, args=(filename, prob_start, prob_end))
        pr.start()
        processes.append(pr)

    # Wait for all child processes to finish
    for pr in processes:
        pr.join()
