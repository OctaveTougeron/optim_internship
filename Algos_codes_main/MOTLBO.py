import random
import math    
import copy    
import sys     
import numpy as np 

'''Cet algorithme est un algorithme d'optimisation qui sélectionne le meilleur 'élève' dans une population par rapport à sa fitness pour orienter les autres élèves vers lui.
Pour la documentation, voir : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8937805 

Une partie du code est issue de : https://www.geeksforgeeks.org/implementation-of-teaching-learning-based-optimization/?ref=ml_lbp

'''
###### Mono-objective TLBO

# Student class
class Student:
  def __init__(self, fitness, dim, minx, maxx, seed):
    self.rnd = random.Random(seed)
 
    # a list of size dim 
    # with 0.0 as value of all the elements
    self.position = [0.0 for i in range(dim)]
 
    # loop dim times and randomly select value of decision var
    # value should be in between minx and maxx
    for i in range(dim):
      self.position[i] = ((maxx - minx) *
        self.rnd.random() + minx)
 
    # compute the fitness of student
    self.fitness = fitness(self.position)
 
 
# Teaching learning based optimization
def tlbo(fitness, max_iter, n, dim, minx, maxx):
  '''
  Inputs : 
  fitness : python function defined as usual;
  max_iter : maximum iteration for the loop;
  n : number of students in the class;
  dim : dimension of the space;
  minx : lower bound;
  maxx : upper bound;
  
  Outputs : Best student of the classroom;
  
  '''
  
  rnd = random.Random(0)
 
  # create n random students
  classroom = [Student(fitness, dim, minx, maxx, i) for i in range(n)] 
 
  # compute the value of best_position and best_fitness in the classroom
  Xbest = [10.0 for i in range(dim)]

  ### Modify Fbest according to number of objective

  Fbest = sys.float_info.max  

  X = []  
  X.append([classroom[j].position for j in range (len(classroom))])
  
  for i in range(n): # check each Student
    if classroom[i].fitness < Fbest:
      Fbest = classroom[i].fitness
      Xbest = copy.copy(classroom[i].position) 
 
  # main loop of tlbo
  Iter = 0
  while Iter < max_iter:
     
    # after every 10 iterations 
    # print iteration number and best fitness value so far
    if Iter % 10 == 0 and Iter > 1:
      print("Iter = " + str(Iter) + " best fitness = " ,Fbest)
 
    # for each student of classroom
    for i in range(n): 
 
      ### Teaching phase of ith student
 
      # compute the mean of all the students in the class
      Xmean = [0.0 for i in range(dim)]
      for k in range(n):
          for j in range(dim):
              Xmean[j]+= classroom[k].position[j]
       
      for j in range(dim):
          Xmean[j]/= n;
       
      # initialize new solution
      Xnew = [0.0 for i in range(dim)]
 
      # teaching factor (TF)
      # either 1 or 2 ( randomly chosen)
      TF = random.randint(1, 3)
 
      # best student of the class is teacher
      Xteacher = Xbest
      F =  classroom[i].fitness 
      # compute new solution 
      for j in range(dim):
          Xnew[j] = classroom[i].position[j] + rnd.random()*(Xteacher[j] - TF*Xmean[j])
       
      # if Xnew < minx OR Xnew > maxx
      # then clip it 
      for j in range(dim):
          Xnew[j] = max(Xnew[j], minx)
          Xnew[j] = min(Xnew[j], maxx)
       
      # compute fitness of new solution
      fnew = fitness(Xnew)
 
      # if new solution is better than old 
      # replace old with new solution
      if(fnew < F):
          classroom[i].position = Xnew
          classroom[i].fitness = fnew
        
      # update best student
      if(fnew < Fbest):
          Fbest = fnew
          Xbest = Xnew
 
      ### learning phase of ith student
 
      # randomly choose a solution from classroom
      # chosen solution should not be ith student
      p = random.randint(0, n-1)
      while(p==i):
          p = random.randint(0, n-1)
       
      # partner solution
      Xpartner = classroom[p]
 
      Xnew = [0.0 for i in range(dim)]
      if(F < Xpartner.fitness):
          for j in range(dim):
              Xnew[j] = classroom[i].position[j] + rnd.random()*(classroom[i].position[j] - Xpartner.position[j])
      else:
          for j in range(dim):
              Xnew[j] = classroom[i].position[j] - rnd.random()*(classroom[i].position[j] - Xpartner.position[j])
 
      # if Xnew < minx OR Xnew > maxx
      # then clip it 
      for j in range(dim):
          Xnew[j] = max(Xnew[j], minx)
          Xnew[j] = min(Xnew[j], maxx)
       
      # compute fitness of new solution
      fnew = fitness(Xnew)
 
      # if new solution is better than old 
      # replace old with new solution
      if(fnew < F):
          classroom[i].position = Xnew
          classroom[i].fitness = fnew
        
      # update best student
      if(fnew < Fbest):
          Fbest = fnew
          Xbest = Xnew
    X.append([classroom[j].position for j in range (len(classroom))])
    Iter += 1
  # end-while
 
  # return best student from classroom
  return Xbest, X

######## Multi-objective algorithm TLBO

# Teaching learning based optimization for MO with 2 objectives
def motlbo(fitness, max_iter, n, dim, minx, maxx):
  '''
  Inputs : 
  fitness : python function defined as usual;
  max_iter : maximum iteration for the loop;
  n : number of students in the class;
  dim : dimension of the space;
  minx : lower bound;
  maxx : upper bound;
  
  Outputs : Best student of the classroom;
  
  '''
  
  rnd = random.Random(0)
 
  # create n random students
  classroom = [Student(fitness, dim, minx, maxx, i) for i in range(n)] 
 
  # compute the value of best_position and best_fitness in the classroom
  Xbest = [10.0 for i in range(dim)]

  ### Modify Fbest according to number of objective

  Fbest = [sys.float_info.max, sys.float_info.max]

  X = []  
  X.append([classroom[j].position for j in range (len(classroom))])
  
  for i in range(n): # check each Student
    if classroom[i].fitness < Fbest:
      Fbest = classroom[i].fitness
      Xbest = copy.copy(classroom[i].position) 
 
  # main loop of tlbo
  Iter = 0
  while Iter < max_iter:
     
    # after every 10 iterations 
    # print iteration number and best fitness value so far
    if Iter % 10 == 0 and Iter > 1:
      print("Iter = " + str(Iter) + " best fitness = " ,Fbest)
 
    # for each student of classroom
    for i in range(n): 
 
      ### Teaching phase of ith student
 
      # compute the mean of all the students in the class
      Xmean = [0.0 for i in range(dim)]
      for k in range(n):
          for j in range(dim):
              Xmean[j]+= classroom[k].position[j]
       
      for j in range(dim):
          Xmean[j]/= n;
       
      # initialize new solution
      Xnew = [0.0 for i in range(dim)]
 
      # teaching factor (TF)
      # either 1 or 2 ( randomly chosen)
      TF = random.randint(1, 3)
 
      # best student of the class is teacher
      Xteacher = Xbest
      F =  classroom[i].fitness 
      # compute new solution 
      for j in range(dim):
          Xnew[j] = classroom[i].position[j] + rnd.random()*(Xteacher[j] - TF*Xmean[j])
       
      # if Xnew < minx OR Xnew > maxx
      # then clip it 
      for j in range(dim):
          Xnew[j] = max(Xnew[j], minx)
          Xnew[j] = min(Xnew[j], maxx)
       
      # compute fitness of new solution
      fnew = fitness(Xnew)
 
      # if new solution is better than old 
      # replace old with new solution
      if(fnew < F):
          classroom[i].position = Xnew
          classroom[i].fitness = fnew
        
      # update best student
      if(fnew < Fbest):
          Fbest = fnew
          Xbest = Xnew
 
 
 
      ### learning phase of ith student
 
      # randomly choose a solution from classroom
      # chosen solution should not be ith student
      p = random.randint(0, n-1)
      while(p==i):
          p = random.randint(0, n-1)
       
      # partner solution
      Xpartner = classroom[p]
 
      Xnew = [0.0 for i in range(dim)]
      if(F < Xpartner.fitness):
          for j in range(dim):
              Xnew[j] = classroom[i].position[j] + rnd.random()*(classroom[i].position[j] - Xpartner.position[j])
      else:
          for j in range(dim):
              Xnew[j] = classroom[i].position[j] - rnd.random()*(classroom[i].position[j] - Xpartner.position[j])
 
      # if Xnew < minx OR Xnew > maxx
      # then clip it 
      for j in range(dim):
          Xnew[j] = max(Xnew[j], minx)
          Xnew[j] = min(Xnew[j], maxx)
       
      # compute fitness of new solution
      fnew = fitness(Xnew)
 
      # if new solution is better than old 
      # replace old with new solution
      if(fnew < F):
          classroom[i].position = Xnew
          classroom[i].fitness = fnew
        
      # update best student
      if(fnew < Fbest):
          Fbest = fnew
          Xbest = Xnew
    X.append([classroom[j].position for j in range (len(classroom))])
    Iter += 1
  # end-while
 
  # return best student from classroom
  return Xbest, X