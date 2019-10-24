
slides:
    Data analysis and Machine learning: Logistic regression
==

convex function

jupyter:
    LogReg
==
        ==
|| Gradient Methods ||
        ==
    Standard (simple) gradient descent (GD)
    Stochastic GD (SGD)
        - plain batch SGD
        - Momentum SGD
        - ADAM SGD
        :
        :
        :

    Commonality: parameter(s) 
        Learning rate \gamma

b[n+1] = b[n] - dCost(B)/ddCost(B) = B[n] - gamma*dCost(B[n])  = B[n] - gamma*(X.T@(P(B[n])-y))

[[
Steepest descent need second derivative (Hessian)
Conjugate gradient 
Broyden-Fletcher et al. Estimate Hessian
    
    --All require handling hessian matrix
]]


Convex function

///     BREAK       ////

optimization problem

jupyter: 
    Splines
==


