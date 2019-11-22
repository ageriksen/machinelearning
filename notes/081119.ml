# Return to Boosting

### XGBoost - eXtreme Gradient Boost

### Basic philosophy of Boosting - combine weak classifyers to create a stronger classifier
    : weak classifier - slightly better prediction than random guess.

def f_m = sum_m ( \beta b(x; \gamma_m))
def cost func C(y, f) = sum_i L(y_i, f(x_i))

###Algorithm
- establish a cost function, eg. C(y, f) = (1/n) \sum_i(y_i - f_M(x_i))^2, f_M(x) = \sum_i(\beta_m b(x; \gamma_m))
- initial val f_0(x) // eg. 0
- for m=1:M (regression case)
    a) C(y, f) = sum_i ( y_i - f_{m-1}(x_i) - \beta b(x_i; \gamma)^2 // \beta b is a small correction
    b) minimize C(y, f) wrt \gamma and \beta
        (\beta_m, \gamma_m) = argmin_{\beta. \gamma}(
            \sum_i ( y_i - f_{m-1}(x_i) - \beta b(x_i, \gamma) )^2 )
    c) update f_m(x) = f_{m-1} + \beta_m b(x; \gamma_m)
###!Algorithm

1) guess on a function. 
2) regression / classification
3) update function
4) repeat

_what the hell is a confusion matrix??_

XGBoost is a library you need to install, but it interfaces nicely with sklearn
    : seems to work really really well with classification. Not as impressive wrt regression

##Gradient boosting
###Regression

cost func C(y, f) = \sum_i L(y_i, f_i)
f = argmin( C(u, f) )

f_M = \sum_m (h_m) 


