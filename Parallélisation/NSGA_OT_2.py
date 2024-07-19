#! /usr/bin/env python

import openturns as ot
import math
import sys

# check we don't expose penalized values
f = ot.SymbolicFunction(
    ["x1", "x2"], ["x1", "var g := 1.0 + 9.0 * (x1 + x2); g * (1.0 - sqrt(x1 / g))"]
)

dist = ot.JointDistribution([ot.Uniform(0.0, 5.0)] * 2)

# Easy way to get a function which can be called in parallel by pagmo
# As a rule:
# + SymbolicFunction is not thread-safe
# + PythonFunction is not thread-safe
# + Any other Function based on the C++ classes are thread-safe
inS = dist.getSample(10000)
outS = f(inS)
algo = ot.LeastSquaresExpansion(inS, outS, dist)
algo.run()
f_approx = algo.getResult().getMetaModel()

# You can replace  f_approx by f to see that there is no parallelism with PythonFunction
zdt1 = ot.OptimizationProblem(f_approx)
bounds = ot.Interval([0.0] * 2, [5.0] * 2)
zdt1.setBounds(bounds)
ineq = ot.ComposedFunction(
    ot.SymbolicFunction(["y1", "y2"], ["2-y1", "2-y2"]), f
)  # 0 <= y1,y2 <= 2
zdt1.setInequalityConstraint(ineq)
pop0 = dist.getSample(10000)
algo = ot.Pagmo(zdt1, "nsga2", pop0)
algo.setMaximumIterationNumber(50)
algo.run()
result = algo.getResult()
x = result.getFinalPoints()
y = result.getFinalValues()
front = result.getParetoFrontsIndices()[0]
print("penalized", front)
