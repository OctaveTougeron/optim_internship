Parallélisation
---------------

Ce dossier regroupe la mise en oeuvre de la parallélisation pour deux algorithmes:

- GACO (Global Ant Colony Optimization)
- NSGAII (Non-dominated Sorting Genetic Algorithm II)

Cette parallélisation est ici testée avec 2 modules Python : DEAP et Openturns (OT).

- GACO_OT détaille la parallélisation possible des algorithmes mono-objectifs sous openturns, en particulier de GACO;
- NSGA_OT montre que la parallélisation avec des fonctions classiques d'openturns n'est pas possible pour les algos multi-objectifs;
- NSGA_DEAP détaille la parallélisation de NSGAII sous le module DEAP de Python;
- NSGA_OT_2 montre qu'on peut observer la parallélisation sous openturns avec NSGAII en construisant des méta-modèles qui relève de classes C++.
