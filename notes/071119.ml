
*XGBoost*
    : preferred boosting method

bagging or boosting
    : main 2 types of decision trees

lot on slides. following on blackboard

##Boosting
    adaptive boost
    gradient boost
    XG boost
basic math
Additive modelling
e.g. binary classification
    
    G_m(x) \in {-1,1}

basic filosophy
set of basis funcs, f_m(x) = \sum_{m=1}{M} \beta_m b(x; \gamma)
    \beta = expansion coeff
    \b simple ficm of a multivariable argument 

###Algorithm 10.1 Hastings book, AdaBoost.M1.
    - minimize cost function C(y, f)
        i) initialize f_0(x)
        ii) for m=1:M
            a) compute 
                ( \beta_m, \gamma_m ) 
                = argmin_{\beta, \gamma} 
                    \sum_{i=0}^{n-1} L(y_i. f_{m-1}(x_i) + \beta_m b(x_i, \gamma) )
            b) set f_m(x) = f_{m-1}(x) + \beta_m b8x; \gamma_m)
        :
        :
        :
###!Algorithm

something something errors. 

I think the essence is that you start with the initial set, make a fit of some sort, then weight the 
set and draw a new one. Repeat until something, and the error is better.
or something


