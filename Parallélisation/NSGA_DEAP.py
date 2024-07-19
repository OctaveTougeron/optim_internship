import multiprocessing
import time
from deap import base
from deap import creator
import openturns as ot
import numpy as np
from deap import tools, algorithms
import matplotlib.pyplot as plt
import random 

HUGE = 10**2 # Réduire ou augmenter pour un temps d'exécution plus ou moins long
ncpus=1 # Nombre de coeurs
Niter = 100 # Nombre d'itérations de NSGAII
N=1000 # Taille pop initiale

# Fonction d'évaluation, prise en compte des contraintes par pénalisation car c'est plus simple avec DEAP.
def evaluate(x):
    t0 = time.time()
    #time.sleep(4)
    tmp = ot.Normal().getSample(HUGE).computeMean()
    print(time.time()-t0)
    print(f"minlp_obj executed on PID {multiprocessing.current_process().pid} with x = {x}")
    obj1 = (x[0]**2
            -x[1]
            +x[2]
            +3*x[3]
            +2*x[4]
            +x[5]
    )

    obj2 = (2*x[0]**2
            +x[2]**2
            -3*x[0]
            +x[1]
            -2*x[3]
            +x[4]
            -2*x[5]
    )
    e1 = (3*x[0]
          -x[1]
          +x[2]
          +2*x[3]
    )
    e2 = (4*x[0]**2
          +2*x[0]
          +x[1]
          +x[2]
          +x[3]
          +7*x[4]
          -40
    )
    e3 = (-x[0]
          -2*x[1]
          +3*x[2]
          +7*x[5]
    )
    e4 = (-x[0] 
          +12*x[3]
          -10
    )
    e5 = (x[0]
          -2*x[3]
          -5
    )
    e6 = (-x[1]
          +x[4]
          -20
    )
    e7 = (x[1]
          -x[4]
          -40
    )
    e8 = (-x[2]
          +x[5]
          -17
    )
    e9 = (x[2]
          -x[5]
          -25
    )
    penalty = 0
    r = 10e-1
    cons = [e1, e2, e3, e4, e5, e6, e7, e8, e9]
    for c in cons:
        penalty+=r*max(0,c)**2
    return(obj1+penalty,obj2+penalty)


if __name__=='__main__':
    time0 = time.time()
      
    # Définition du problème d'optimisation
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Définition des bornes des variables
    BOUND_LOW, BOUND_UP = -10.0, 10.0
    BOUND = [BOUND_LOW, BOUND_UP]

    # Création de la toolbox
    toolbox = base.Toolbox()

    # Création du type d'individu
    toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)
    toolbox.register("attr_int", random.randint,0,1)
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_int, toolbox.attr_int, toolbox.attr_int), n=1)
    
    # Définition de la mutation, car on a des entiers et des variables continues.
    def mutate(individual):
        # Mutation pour les variables continues
        for i in range(3):
            if random.random() < 0.2:
                individual[i] += random.gauss(0, 1)
                individual[i] = max(-10, min(10, individual[i]))  # Borne les variables continues
    
        # Mutation pour les variables discrètes
        for i in range(3, 6):
            if random.random() < 0.2:
                individual[i] = 1 - individual[i]  # Toggle 0/1
    
        return individual,
    
    # Population, mutation, sélection, évaluation
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)

    # Multiprocessing
    pool = multiprocessing.Pool(processes=ncpus)
    toolbox.register("map", pool.map)

    # Paramètres de l'algorithme génétique
    pop_size = N
    max_gen = Niter

    # Création de la population initiale
    pop = toolbox.population(n=pop_size)

    # Évolution de la population
    pop, _ = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.2, ngen=max_gen, verbose=False)

    # Extraction du front de Pareto
    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    # Affichage du front de Pareto
    front_f1_reel = [ind.fitness.values[0] for ind in front]
    front_f2_reel = [ind.fitness.values[1] for ind in front]

    plt.figure(figsize=(8, 6))
    plt.scatter(front_f1_reel, front_f2_reel, color='blue')
    plt.title('Front de Pareto avec contraintes pénalisées')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.grid(True)
    plt.show()

    # Individus finaux  
    print(time.time()-time0)
    param_values = [list(ind) for ind in front]
    for i, ind in enumerate(param_values):
        print(f"Individu {i}: {ind}")