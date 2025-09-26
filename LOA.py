import time
import numpy as np


# Lyrebird Optimization Algorithm (LOA)
def LOA(lyrebirds, fobj, lb, ub, max_iterations):
    num_variables, num_lyrebirds = lyrebirds.shape[0], lyrebirds.shape[1]
    sigma = 0.1
    crossover_rate = 0.8
    mutation_rate = 0.1
    best_solution = np.zeros((1, num_lyrebirds))
    best_fitness = float('inf')
    Convergence_curve = np.zeros((max_iterations, 1))

    t = 0
    ct = time.time()
    # Main loop
    for iteration in range(max_iterations):
        # Evaluate fitness
        fitness = fobj(lyrebirds[:])

        # Sort by fitness
        sorted_indices = np.argsort(fitness)
        lyrebirds = lyrebirds[sorted_indices]
        fitness = fitness[sorted_indices]

        # Keep top half
        lyrebirds = lyrebirds[:num_lyrebirds // 2]

        # Crossover
        num_crossovers = round(crossover_rate * (num_lyrebirds // 2))
        offspring = []
        for _ in range(num_crossovers):
            p1, p2 = lyrebirds[np.random.randint(0, num_lyrebirds // 2, 2)]
            alpha = np.random.rand()
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = (1 - alpha) * p1 + alpha * p2
            offspring.extend([child1, child2])

        # Add offspring to population
        lyrebirds = np.vstack([lyrebirds, offspring])

        # Mutation
        num_mutations = round(mutation_rate * num_lyrebirds)
        for _ in range(num_mutations):
            idx = np.random.randint(len(lyrebirds))
            mutation = sigma * np.random.randn(num_variables)
            lyrebirds[idx] = np.clip(lyrebirds[idx] + mutation, lb, ub)

        # Evaluate new fitness
        fitness = fobj(lyrebirds[:])
        sorted_indices = np.argsort(fitness)
        lyrebirds = lyrebirds[sorted_indices]
        fitness = fitness[sorted_indices]

        # Print best solution
        best_solution = lyrebirds[0]
        best_fitness = fitness[0]
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[max_iterations - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct
