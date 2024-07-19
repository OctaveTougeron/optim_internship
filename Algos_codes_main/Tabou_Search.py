import random
# Définition du voisinage avec contraintes
def get_neighbors(constraints,x, y, step_size=1):
    '''
    
    Entrées :
    Constraints : liste de fonctions pythons qui symbolisent les contraintes du problème;
    x,y : variables;
    step_size : variable de déplacements dans le voisinage du point x,y;
    
    Sorties : 
    neighbors : liste des voisins acceptables.
    
    '''
    neighbors = []
    for dx in [-step_size, 0, step_size]:
        for dy in [-step_size, 0, step_size]:
            new_x = x + dx
            new_y = y + dy
            if all(constraints[i](new_x, new_y) <= 0 for i in range(len(constraints))):
                neighbors.append((new_x, new_y))
    return neighbors

# Algorithme de recherche tabou avec contraintes
def tabu_search(objective_function,constraints, initial_solution, max_iterations, tabu_size):
    '''
    
    Entrées :
    objective_function : fonction python de l'objectif;
    constraints : liste des contraintes;
    initial_solution : point de départ de la recherche tabou;
    max_iterations : maximum d'itérations de la boucle;
    tabu_size : taille maximale de la liste tabou qui enregistre les déplacements déjà effectués;
    
    Sorties :
    best_solution : meilleur point
    
    '''
    current_solution = initial_solution
    best_solution = current_solution
    tabu_list = []

    for _ in range(max_iterations):
        neighbors = get_neighbors(constraints, *current_solution)
        best_neighbor = None
        best_neighbor_value = float('inf')

        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_value = objective_function(*neighbor)
                if neighbor_value < best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

        if best_neighbor:
            current_solution = best_neighbor
            if objective_function(*best_neighbor) < objective_function(*best_solution):
                best_solution = best_neighbor

        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_solution



# Algorithme de recherche tabou avec contraintes pour 2 objectifs
def tabu_search_2(objective_function1, objective_function2, constraints,initial_solution, max_iterations, tabu_size):
    '''
    
    Entrées :
    objective_function1 et 2 : fonctions python des objectifs;
    constraints : listes des contraintes;
    initial_solution : point de départ de la recherche tabou;
    max_iterations : maximum d'itérations de la boucle;
    tabu_size : taille maximale de la liste tabou qui enregistre les déplacements déjà effectués;
    
    Sorties :
    best_solution : meilleur point
    
    '''
    current_solution = initial_solution
    best_solution = current_solution
    tabu_list = []

    for _ in range(max_iterations):
        neighbors = get_neighbors(constraints, *current_solution)
        best_neighbor = None
        best_neighbor_value1 = float('inf')
        best_neighbor_value2 = float('inf')

        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_value1 = objective_function1(*neighbor)
                neighbor_value2 = objective_function2(*neighbor)
                if neighbor_value1 < best_neighbor_value1 or (neighbor_value1 == best_neighbor_value1 and neighbor_value2 < best_neighbor_value2):
                    best_neighbor = neighbor
                    best_neighbor_value1 = neighbor_value1
                    best_neighbor_value2 = neighbor_value2

        if best_neighbor:
            current_solution = best_neighbor
            if (objective_function1(*best_neighbor), objective_function2(*best_neighbor)) < (objective_function1(*best_solution), objective_function2(*best_solution)):
                best_solution = best_neighbor

        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_solution


