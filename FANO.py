import time
import numpy as np


# Far and Near Optimization (FANO)
def FANO(population, objective_func, lb, ub, Max_iter):
    num_particles, dim = population.shape[0], population.shape[1]
    minimize=True
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmin(fitness) if minimize else np.argmax(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]
    Convergence_curve = np.zeros((Max_iter, 1))

    ct = time.time()
    # Step 5. For t = 1 to T
    for t in range(Max_iter):
        # Step 6. For i = 1 to N
        for i in range(num_particles):
            # Step 7. Calculate Di using Eq. (4)
            distances = np.linalg.norm(population[i] - population, axis=1)
            distances[i] = np.inf  # Exclude self

            # Step 8. Determine FMi and NMi
            farthest_idx = np.argmax(distances)
            nearest_idx = np.argmin(distances)

            # Step 9. Phase 1: Moving to the farthest member (Exploration)
            # Step 10. Calculate new position of ith population member using Eq. (5)
            I = np.random.choice([1, 2])
            r1 = np.random.rand(dim)
            xp1 = population[i] + r1 * (population[farthest_idx] - I * population[i])
            xp1 = np.clip(xp1, lb, ub)
            fp1 = objective_func(xp1)

            # Step 11. Update ith population member using Eq. (6)
            if (minimize and fp1 <= fitness[i]) or (not minimize and fp1 >= fitness[i]):
                population[i] = xp1
                fitness[i] = fp1

            # Step 12. Phase 2: Moving to the nearest member (Exploitation)
            # Step 13. Calculate new position of ith population member using Eq. (7)
            r2 = np.random.rand(dim)
            xp2 = population[i] + r2 * (population[nearest_idx] - I * population[i])
            xp2 = np.clip(xp2, lb, ub)
            fp2 = objective_func(xp2)

            # Step 14. Update ith population member using Eq. (8)
            if (minimize and fp2 <= fitness[i]) or (not minimize and fp2 >= fitness[i]):
                population[i] = xp2
                fitness[i] = fp2

        # Step 16. Save best candidate solution so far.
        best_idx = np.argmin(fitness) if minimize else np.argmax(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct
