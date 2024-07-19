import numpy as np
import openturns as ot
from scipy.stats import norm

# Calcul de l'EI

def EI(Y, y_min):
    std = np.std(Y)
    mu = np.mean(Y)
    diff = y_min-mu
    if std == 0:
        return(0, 0)
    u = diff/std
    EI = []
    for i in range (len(Y)):

        ei = diff*norm.cdf(u)+std*norm.pdf(u)
        EI.append(ei)
    return(np.argmax(EI), np.max(EI)) # retourne l'individu de Y qui maximise l'EI

# Séparation de l'espace du front de Pareto, ici pour la dimension 2

def divivide_plan(y_sample1,y_sample2, theta, n):
    Y1 = list()
    Y2 = list()
    IND = list()
    for i in range(n-1):
        y1 = []
        y2 = []
        ind = []
        for j in range (len(y_sample1)): # on regarde individu par individu sa position dans le plan
            
            if np.tan((i+1)*theta) * y_sample1[j][0] >= y_sample2[j][0] and np.tan(i*theta) * y_sample1[j][0] <= y_sample2[j][0]:
                y1.append(y_sample1[j][0])
                y2.append(y_sample2[j][0])
                ind.append(j)
            if i+1 == n and np.tan(i*theta) * y_sample1[j][0] <= y_sample2[j][0]:
                y1.append(y_sample1[j][0])
                y2.append(y_sample2[j][0])
                ind.append(j)
        Y1.append(y1)
        Y2.append(y1)
        IND.append(ind)
    return(Y1, Y2, IND)  

def MOEGO_NSGAII(objectif1, objectif2,  x_train, N_appels_max,  bounds_low, bounds_up,n = 2, N_iter = 100, N_pop = 128,dimension = 2,var_types = None, constraints_ineq = None, constraints_eq = None):
    '''
    Entrées :
    objectif1 : Fonction OpenTurns (SymbolicFunction ou PythonFunction);
    objectif2 : Fonction OpenTurns (SymbolicFunction ou PythonFunction);
    x_train : échantillon des variables sous forme d'ot.Sample ou de tableau numpy;
    N_appels_max : entier du nombre max d'appels aux objectifs;
    bounds_low : liste des bornes inférieures des variables, si il n'y en a pas, mettre -1e99;
    bounds_up : liste des bornes supérieures des variables, si il n'y en a pas, mettre 1e99;
    n : entier, nombre de divisions du plan dans divide_plan, par défaut à 2;
    N_iter : entier, nombre d'itérations de NSGAII sur les méta-modèles, par défaut à 100;
    N_pop : entier, nombre d'individus dans la population pour NSGAII, par défaut à 128;
    dimension : entier, nombre d'objectifs, par défaut 2;
    var_types : liste des types des variables dans l'ordre, 0 pour CONTINUOUS, 1 pour INTEGER, 2 pour BINARY, si rien on les considère toutes comme continues;
    constraints_ineq : Fonction OpenTurns (SymbolicFunction ou PythonFunction) des contraintes d'inégalité g(x) >= 0;
    constraints_eq : Fonction OpenTurns (SymbolicFunction ou PythonFunction) des contraintes d'inégalité h(x) = 0;
    
    Sorties : 
    Tableaux FP1 et FP2, couplés ils forment le front de Pareto, ces tableaux permettent le plot matplotlib
    
    '''
    # Initalisation et construction des tableaux d'entraînement pour le krigeage
    start = objectif1.getCallsNumber() + objectif2.getCallsNumber()
    y_train1 = objectif1(x_train)
    y_train2 = objectif2(x_train)
    end = objectif1.getCallsNumber() + objectif2.getCallsNumber()
    N_ite = end-start
    FP1 = []
    FP2 = []
    dim_exp = objectif1.getInputDimension()
    # Boucle principale
    while N_ite <= N_appels_max:
        start = objectif1.getCallsNumber() + objectif2.getCallsNumber()
        
        # Krigeage pour objectif 1
        basis = ot.ConstantBasisFactory(dimension).build()
        basis = ot.QuadraticBasisFactory(dimension).build()
        covarianceModel = ot.MaternModel( [1.0] * dimension, 1.5)
        algo = ot.KrigingAlgorithm(x_train, y_train1, covarianceModel,  basis)
        algo.run()
        result1 = algo.getResult()
        krigeageMM1 = result1.getMetaModel()
        
        # Krigeage pour objectif2
        basis = ot.ConstantBasisFactory(dimension).build()
        covarianceModel = ot.MaternModel([1.0] * dimension, 1.5)
        algo = ot.KrigingAlgorithm(x_train, y_train2, covarianceModel,  basis)
        algo.run()
        result2 = algo.getResult()
        krigeageMM2 = result2.getMetaModel()
        
        # Réunion des modèles
        functions = []
        functions.append(krigeageMM1)
        functions.append(krigeageMM2)
        model = ot.AggregatedFunction(functions)

        # Créer le problème d'optimisation openturns
        problem = ot.OptimizationProblem(model)
        bounds = ot.Interval(bounds_low , bounds_up )
        problem.setBounds(bounds)

        if var_types == None:
            problem.setVariablesType([ot.OptimizationProblemImplementation.CONTINUOUS]*dim_exp)
        else :
            list = []
            for i in range (len(var_types)):
                if var_types[i]==0:
                    list.append(ot.OptimizationProblemImplementation.CONTINUOUS)
                if var_types[i]==1:
                    list.append(ot.OptimizationProblemImplementation.INTEGER)
                if var_types[i]==2:
                    list.append(ot.OptimizationProblemImplementation.BINARY)
            problem.setVariablesType(list)
        if constraints_ineq:   
            problem.setInequalityConstraint(constraints_ineq)
        if constraints_eq:   
            problem.setEqualityConstraint(constraints_eq)
        
        distr = []
        for i in range (len(bounds_low)):
            if var_types[i]==0:
                distr.append(ot.Uniform(bounds_low[i], bounds_up[i]))
            if var_types[i]==1:
                integer_values = [[j] for j in range (bounds_low[i], bounds_up[i])]
                discrete_values = ot.Sample(integer_values)
                factory = ot.UserDefinedFactory()
                distribution = factory.build(discrete_values)
                distr.append(distribution)
            if var_types[i]==2:
                integer_values = [[j] for j in range (bounds_low[i], bounds_up[i])]
                discrete_values = ot.Sample(integer_values)
                factory = ot.UserDefinedFactory()
                distribution = factory.build(discrete_values)
                distr.append(distribution)
        
        uniform_mixed = ot.ComposedDistribution(distr)

        ot.RandomGenerator.SetSeed(0)

        init_pop = uniform_mixed.getSample(N_pop)

        # Algo NSGAII
        algo = ot.Pagmo(problem, 'nsga2', init_pop) 

        algo.setMaximumIterationNumber(N_iter) 

        algo.run() 

        result = algo.getResult() 

        final_pop_x = result.getFinalPoints() 
        final_pop_y = result.getFinalValues()

        front0 = result.getParetoFrontsIndices()[0] 
        front0_x = final_pop_x.select(front0) 
        front0_y = final_pop_y.select(front0)

        # Sélection des points
        y_min1 = np.min(y_train1)
        y_min2 = np.min(y_train2)
        Y1, Y2, IND= divivide_plan(front0_y[:,0],front0_y[:,1], np.pi/n, n)

        for i in range (len(Y1)):
            if len(Y1[i])==0 or len(Y2[i])==0:
                None
            else :    
                arg1, ei1 = EI(Y1[i], y_min1)
                arg2, ei2 = EI(Y2[i], y_min2)
                V11 = objectif1(front0_x[IND[i]][arg2])
                V21 = objectif2(front0_x[IND[i]][arg2])
                V12 = objectif1(front0_x[IND[i]][arg1])
                V22 = objectif2(front0_x[IND[i]][arg1])
                if ei2>ei1:
                    y_train1 = np.concatenate ((y_train1, np.array([V11])))
                    y_train2 = np.concatenate((y_train2, np.array([V21])))
                    x_train = np.concatenate((x_train, np.array([front0_x[IND[i]][arg2]])))
                    FP1.append(V11)
                    FP2.append(V21)
                else : 
                    y_train1 = np.concatenate ((y_train1, np.array([V12])))
                    y_train2 = np.concatenate((y_train2, np.array([V22])))
                    x_train = np.concatenate((x_train, np.array([front0_x[IND[i]][arg1]])))
                    FP1.append(V12)
                    FP2.append(V22)
        end = objectif1.getCallsNumber() + objectif2.getCallsNumber()
        N_ite+=end-start

        # Montrer l'avancement de l'algorithme à chaque boucle
        print('Appels réalisés : ', N_ite, '/', N_appels_max)
    return(FP1,FP2)