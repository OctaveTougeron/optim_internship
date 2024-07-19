import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot

### A noter
### NSGAII n'est pas parallélisable comme on essaie de le faire ici. Un autre fichier décrit comment le faire. 
### Ici, seuls les calculs à la fin de l'optimisation sont parallélisés, ce qui fait gagner un temps plus faible.

HUGE = 10**2  # Réduire pour simplifier le test ou augmenter
ncpus = 1 # Nombre de coeurs
N = 1000 # Taille pop initiale
Niter = 100 # Nombre itérations de l'algo

dim = 2

def obj_batch(X):
    Xarray = np.array(X, copy=False)
    
    obj1 = Xarray[:, 0]**2 - Xarray[:, 1] + Xarray[:, 2] + 3*Xarray[:, 3] + 2*Xarray[:, 4] + Xarray[:, 5]
    obj2 = 2*Xarray[:, 0]**2 + Xarray[:, 2]**2 - 3*Xarray[:, 0] + Xarray[:, 1] - 2*Xarray[:, 3] + Xarray[:, 4] - 2*Xarray[:, 5]
    
    result = np.column_stack((obj1, obj2))
    tmp = ot.Normal().getSample(HUGE).computeMean()
    print(f"minlp_obj executed on PID {multiprocessing.current_process().pid} with X = {X}")
    return result

def ineq_batch(X):
    Xarray = np.array(X, copy=False)
    
    e1 = 3*Xarray[:, 0] - Xarray[:, 1] + Xarray[:, 2] + 2*Xarray[:, 3]
    e2 = 4*Xarray[:, 0]**2 + 2*Xarray[:, 0] + Xarray[:, 1] + Xarray[:, 2] + Xarray[:, 3] + 7*Xarray[:, 4] - 40
    e3 = -Xarray[:, 0] - 2*Xarray[:, 1] + 3*Xarray[:, 2] + 7*Xarray[:, 5]
    e4 = -Xarray[:, 0] + 12*Xarray[:, 3] - 10
    e5 = Xarray[:, 0] - 2*Xarray[:, 3] - 5
    e6 = -Xarray[:, 1] + Xarray[:, 4] - 20
    e7 = Xarray[:, 1] - Xarray[:, 4] - 40
    e8 = -Xarray[:, 2] + Xarray[:, 5] - 17
    e9 = Xarray[:, 2] - Xarray[:, 5] - 25
    tmp = ot.Normal().getSample(HUGE).computeMean()
    
    result = np.column_stack((-e1, -e2, -e3, -e4, -e5, -e6, -e7, -e8, -e9))
    
    print(f"minlp_cstr executed on PID {multiprocessing.current_process().pid} with X = {X}")
    return result

if __name__ == '__main__':
    obje = ot.PythonFunction(6, 2, func_sample = obj_batch, n_cpus=ncpus)
    ineqe = ot.PythonFunction(6, 9, func_sample = ineq_batch, n_cpus=ncpus)

    bounds = ot.Interval([-1e99, -1e99, -1e99, 0, 0, 0], [1e99, 1e99, 1e99, 1, 1, 1])
    integer_values = [[i] for i in range(0, 2)]
    discrete_values = ot.Sample(integer_values)
    factory = ot.UserDefinedFactory()
    distribution = factory.build(discrete_values)

    # Distribution composée pour les deux types de variables
    dis = ot.ComposedDistribution([ot.Uniform(-10, 10)] * 3 + [ot.UserDefined([[0], [1]])] * 3)
    pop0 = dis.getSample(N)

    problem = ot.OptimizationProblem(obje)
    problem.setBounds(bounds)
    problem.setInequalityConstraint(ineqe)
    problem.setVariablesType(
        [ot.OptimizationProblemImplementation.CONTINUOUS] * 3
        + [ot.OptimizationProblemImplementation.INTEGER] * 3
    )
    algo = ot.Pagmo(problem, 'nsga2', pop0)
    algo.setMaximumIterationNumber(Niter)
    algo.setBlockSize(N)

    t0 = time.time()
    algo.run()
    t1 = time.time()

    print('nsga2', "time_elapsed =", t1 - t0, "s,", "n_cpus =", ncpus)
    result = algo.getResult()

    final_pop_x = result.getFinalPoints()
    final_pop_y = result.getFinalValues()
    front0 = result.getParetoFrontsIndices()[0]

    front0_x = final_pop_x.select(front0)
    front0_y = final_pop_y.select(front0)
    tab1 = [front0_y[i][0] for i in range(len(front0_y))]
    tab2 = [front0_y[i][1] for i in range(len(front0_y))]
    plt.scatter(tab1, tab2)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Front de Pareto')
    plt.show()
    print(final_pop_x)
