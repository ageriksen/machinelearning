
# Support Vector Machines (SVM)
    :p dim space, find hyper p-1 hypersurface demarkation between the specifiers. 

misclassifier defined, so that a correct classification is above 0, while a misclassification 
means below 0 output. `y*(x^T*w + b) > 0` e.g. // y is the output, x is the prediction

SVM was developed mainly for classification. You can perform regression, but mainly for classification

SVM makes the line into a border, signed boundary around a positive M. 
I've fallen off at this point, but there is some minimization of this and that. And lagrange multipliers. 

The lagrange multiplier becomes a sum over the misclassifications. Then you differentiate this multiplier L
wrt. weights and biases. 
