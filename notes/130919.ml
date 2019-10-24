////
main goal, note terms to expand later. 
    o   go through the Feynman method for the important themes
    o   keep focus on lecture and meat of talking points. 
////

o   o   Design matrix

o   hypeparameter/shrinking paramter lambda

resampling methods(?)
o   OLS
o   Ridge Regression
o   Lasso

====
forgot last lecture
o   cross validation:
        you can also keep constant k, randomize values and make groups. 
        each group can posibly become a test.
====

o   resampling
    in essence for use with small data sets. 

o   train-test split


a central issue -> lacking likelihood function


true bottleneck: 
    lasso(?)

linear regression

///Optimization and descent methods//// #Gradient methods

o   Cost function 
        minimization of-

o   Gradient descent

o   norm-1 mathematics

o   optimization problems with constraints


o   support-vector machine

general implementation of lasso:
    delta_j * MSE(beta) + lambda*sign(beta) =   0         if abs(beta_j)  >   0
    delta_j * MSE(beta)                    <=   lambda    if beta_j       =   0


/////Gradient methods//////

o   Newton-Raphson method
        x^(n+1) = x^(n) - ( f(x^(n))/f'(x^(n)) )

o   Jacobian matrix

o   Hessian matrix

o   Learning rate | and it's relation to the hessian and jacobian

///////// break //////////////7

o   benchmarking

o   What do you do when your matrix becomes way too large?


holy grail: minimisee MSE

o   learning parameter/rate

o   singular value decompositiono   
o   LU decomposition


////Logistic regression (Classification)//////// #Binary outcomes
o   Logistic equation
o   Logistic Regression # non-linear eq. in beta
        
        general case


o   intercept


