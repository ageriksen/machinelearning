
Classification problem

linear regression -> trying to fit continuous function


data and machine learning basics: Logistic Regression -> finding likelihood important

--ucl machine learning repository--
--Kaggle.com-- ==> tons of resources for code and datasets

    ////    logistic regression ////

support vector machines

bayesian statistics
    - posterior probability

logistic regression -> gradient method

||gradient descent|| <- derivatives

(principal componen analysis)

i.i.d.
ansats
Maximum likelihood estimation principle

cost function <== how to find $\beta$

        //      break       //

negative log --- convex function

in general, need data, PDF and a cost function

Cross-entropy

regularization parameters (also used in Ridge & Lasso) -- regularization power

gradient descetn - find gradient of cost function - Newton's method-family lends itself well

algo{
beta_opt[n+1] = beta_opt[n] - [dcost]/[ddcost]

beta_opt[n+19 = beta_opt[n] - (X.T@W@X)^-1 (X.T(p-y))

terate, ||(beta_opt[n+1] - beta_opt[n])||_2 > epsilon ~ 10^-10
w/ initial guess beta_opt[0]
normally replace Hessian{ (X.T@W@X)^-1 } with constant gamma -> std. gradient descent - aka learning rate

|||| beta_opt[n+1] = beta_opt[n] - gamma*(X.T(P(beta_opt[n])-y)) ||||
}

