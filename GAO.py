import time
import numpy as np


def GAO(X, fobj, lb, ub, Max_iterations):
    N, dimension = X.shape[0], X.shape[1]
    fit = fobj(X[:])
    best_so_far = np.zeros(Max_iterations)
    average = np.zeros(Max_iterations)

    ct = time.time()
    for t in range(Max_iterations):
        # Update best solution
        Fbest = np.min(fit)
        blocation = np.argmin(fit)

        if t == 0:
            xbest = X[blocation].copy()
            fbest = Fbest
        elif Fbest < fbest:
            fbest = Fbest
            xbest = X[blocation].copy()

        # Update each search agent
        for i in range(N):
            # Phase 1: Attack on termite mounds (exploration)
            TM_location = np.where(fit < fit[i])[0]

            if TM_location.size == 0:
                STM = xbest
            else:
                K = np.random.choice(TM_location)
                STM = X[K]

            I = round(1 + np.random.rand())
            X_new_P1 = X[i] + np.random.rand() * (STM - I * X[i])
            X_new_P1 = np.clip(X_new_P1, lb, ub)

            fit_new_P1 = fit(X_new_P1)
            if fit_new_P1 < fit[i]:
                X[i] = X_new_P1
                fit[i] = fit_new_P1

            # Phase 2: Digging in termite mounds (exploitation)
            X_new_P2 = X[i] + (1 - 2 * np.random.rand(dimension)) * (ub - lb) / (t + 1)
            X_new_P2 = np.clip(X_new_P2, lb / (t + 1), ub / (t + 1))

            f_new = fit(X_new_P2)
            if f_new <= fit[i]:
                X[i] = X_new_P2
                fit[i] = f_new
                if f_new < fbest:
                    xbest = X_new_P2
                    fbest = f_new

        best_so_far[t] = fbest
        average[t] = np.mean(fit)

    Best_score = fbest
    Best_pos = xbest
    GAO_curve = best_so_far
    ct = time.time() - ct

    return Best_score, GAO_curve, Best_pos, ct
