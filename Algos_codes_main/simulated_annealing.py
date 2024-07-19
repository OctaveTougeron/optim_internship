import random
import math


# Algorithme de recuit simulé
def recuit_simule(fonction_objectif, xmin, xmax, ymin, ymax, T_init, T_min, alpha, max_iterations):
    '''

    Entrées : 
    fonction_objectif : objectif à minimiser, fonction python;
    xmin, xmax, ymin, ymax : bornes des variables x et y;
    T_init : température initiale;
    T_min : température finale minimale;
    alpha : facteur de réduction de la température;
    max_iterations : maximum d'itérations pour la boucle;
    
    Sorties :
    best_solution : meilleure solution pour les paramètres;
    best_value : valeur de l'objectif en ce meilleur point.
    
    '''
    # Initialisation de la solution initiale
    x_current = random.uniform(xmin, xmax)
    y_current = random.uniform(ymin, ymax)
    best_solution = (x_current, y_current)
    best_value = fonction_objectif(x_current, y_current)

    T = T_init

    for i in range(max_iterations):
        # Sélection d'une nouvelle solution voisine
        x_new = x_current + random.uniform(-0.1, 0.1)
        y_new = y_current + random.uniform(-0.1, 0.1)
        
        # Vérifier les limites
        x_new = min(max(xmin, x_new), xmax)
        y_new = min(max(ymin, y_new), ymax)

        # Calcul de la variation de la fonction objectif
        delta = fonction_objectif(x_new, y_new) - fonction_objectif(x_current, y_current)

        # Accepter la nouvelle solution avec une certaine probabilité
        if delta < 0 or random.random() < math.exp(-delta / T):
            x_current, y_current = x_new, y_new

            # Mettre à jour la meilleure solution si nécessaire
            if fonction_objectif(x_current, y_current) < best_value:
                best_solution = (x_current, y_current)
                best_value = fonction_objectif(x_current, y_current)

        # Refroidissement de la température
        T *= alpha
        if T < T_min:
            break

    return best_solution, best_value





# Algorithme de recuit simulé multi-objectif
def multi_objective_simulated_annealing(objective_functions, xmin, xmax, ymin, ymax, T_init, T_min, alpha, max_iterations):
    '''
    
    Entrées : 
    fonction_objectif : objectif à minimiser, fonction python;
    xmin, xmax, ymin, ymax : bornes des variables x et y;
    T_init : température initiale;
    T_min : température finale minimale;
    alpha : facteur de réduction de la température;
    max_iterations : maximum d'itérations pour la boucle;
    
    Sorties :
    best_solution : meilleure solution pour les paramètres;
    best_value : valeur de l'objectif en ce meilleur point.
    
    '''
    
    # Initialisation de la solution initiale
    x_current = random.uniform(xmin, xmax)
    y_current = random.uniform(ymin, ymax)
    best_solution = (x_current, y_current)
    best_value = [f(x_current, y_current) for f in objective_functions]

    T = T_init

    for i in range(max_iterations):
        # Sélection d'une nouvelle solution voisine
        x_new = x_current + random.uniform(-0.1, 0.1)
        y_new = y_current + random.uniform(-0.1, 0.1)
        
        # Vérifier les limites
        x_new = min(max(xmin, x_new), xmax)
        y_new = min(max(ymin, y_new), ymax)

        # Calcul de la variation des fonctions objectif
        delta = [f(x_new, y_new) - f(x_current, y_current) for f in objective_functions]

        # Accepter la nouvelle solution avec une certaine probabilité
        accept = all(d <= 0 or random.random() < math.exp(-d / T) for d in delta)
        if accept:
            x_current, y_current = x_new, y_new

            # Mettre à jour la meilleure solution si nécessaire
            current_values = [f(x_current, y_current) for f in objective_functions]
            if all(cv <= bv for cv, bv in zip(current_values, best_value)):
                best_solution = (x_current, y_current)
                best_value = current_values

        # Refroidissement de la température
        T *= alpha
        if T < T_min:
            break

    return best_solution, best_value
