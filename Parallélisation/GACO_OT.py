import openturns as ot
import math
import time
import multiprocessing

### A noter
### GACO réalise une évaluation séquentielle de la première population, on n'observe donc pas la parallélisation au début.

HUGE = 10**8  # Réduire pour diminuer le temps d'un appel
ncpus = 4 # Nombre de coeurs à utiliser
N = 8 # Taille de la population initiale
Niter = 5 # Nombre d'itérations de l'algorithme

def minlp_obj(x):
    t0 = time.time()
    obj = 0
    for i in range(3):
        obj += (
            (x[2 * i - 2] - 3) ** 2 / 1000.0
            - (x[2 * i - 2] - x[2 * i - 1])
            + math.exp(20.0 * (x[2 * i - 2] - x[2 * i - 1]))
        )
    tmp = ot.Normal().getSample(HUGE).computeMean() # augmenter le temps d'une éxécution pour voir la différence
    print(time.time()-t0) # pour voir combien de temps prend un appel à la fonction

    print(f"minlp_obj executed on PID {multiprocessing.current_process().pid} with x = {x}") # pour voir si les calculs se ofnt bien sur des coeurs différents 
    return [obj]

def minlp_cstr(x):
    ce1 = 4 * (x[0] - x[1]) ** 2 + x[1] - x[2] ** 2 + x[2] - x[3] ** 2
    ce2 = (
        8 * x[1] * (x[1] ** 2 - x[0])
        - 2 * (1 - x[1])
        + 4 * (x[1] - x[2]) ** 2
        + x[0] ** 2
        + x[2]
        - x[3] ** 2
        + x[3]
        - x[4] ** 2
    )
    ce3 = (
        8 * x[2] * (x[2] ** 2 - x[1])
        - 2 * (1 - x[2])
        + 4 * (x[2] - x[3]) ** 2
        + x[1] ** 2
        - x[0]
        + x[3]
        - x[4] ** 2
        + x[0] ** 2
        + x[4]
        - x[5] ** 2
    )
    ce4 = (
        8 * x[3] * (x[3] ** 2 - x[2])
        - 2 * (1 - x[3])
        + 4 * (x[3] - x[4]) ** 2
        + x[2] ** 2
        - x[1]
        + x[4]
        - x[5] ** 2
        + x[1] ** 2
        + x[5]
        - x[0]
    )
    ci1 = (
        8 * x[4] * (x[4] ** 2 - x[3])
        - 2 * (1 - x[4])
        + 4 * (x[4] - x[5]) ** 2
        + x[3] ** 2
        - x[2]
        + x[5]
        + x[2] ** 2
        - x[1]
    )
    ci2 = -(
        8 * x[5] * (x[5] ** 2 - x[4])
        - 2 * (1 - x[5])
        + x[4] ** 2
        - x[3]
        + x[3] ** 2
        - x[4]
    )
    print(f"minlp_cstr executed on PID {multiprocessing.current_process().pid} with x = {x}")
    return [-ce1, -ce2, -ce3, -ce4, -ci1, -ci2]

if __name__ == "__main__": # protéger la parallélisation
    f = ot.PythonFunction(6, 1, minlp_obj, n_cpus=ncpus)
    bounds = ot.Interval([-5.0] * 6, [5.0] * 6)
    ineq = ot.PythonFunction(6, 6, minlp_cstr, n_cpus=ncpus)

    # création pop initiale

    pop0 = ot.ComposedDistribution(
        [ot.Uniform(-5.0, 5.0)] * 4 + [ot.UserDefined([[i - 5] for i in range(11)])] * 2
    ).getSample(N)

    # création problème

    problem = ot.OptimizationProblem(f)
    problem.setBounds(bounds)
    problem.setInequalityConstraint(ineq)
    problem.setVariablesType(
        [ot.OptimizationProblemImplementation.CONTINUOUS] * 4
        + [ot.OptimizationProblemImplementation.INTEGER] * 2
    )

    # choix de l'algo
    
    algo = ot.Pagmo(problem, 'gaco', pop0)
    algo.setMaximumIterationNumber(Niter)
    algo.setBlockSize(N)
    t0 = time.time()
    algo.run()
    t1 = time.time()
    result = algo.getResult()
    x = result.getOptimalPoint()
    y = result.getOptimalValue()

    print('gaco', x, y, "time_elapsed = " ,t1 - t0, "s,", "n_cpus = ", ncpus)
