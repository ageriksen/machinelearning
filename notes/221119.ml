
# Recap SVM

mean deadpoint between classes. using a vector from midpoint of dead land 
in between. 
define a 
    f(x) = x^Tw + b = 0

and include a slack variable - new lagrangian multiplier to the problem

cases where SVM would not work, e.g. cases where "Kernel Classification" would be better suited

The classes can often be remapped so that the SVM can be used. then you need to massage your data. 

At the end of the day, find a constraint, set up Lagrangian multipliers and find the lagrangian
optimise from there.


    ===========================
              break
    ===========================

# solving de's with ml

## Using DNNs to solve de's

intuition: find a function satisfying the characteristics 

    trial sol
    satisfying boundary conditions
    use output from nn

e.g.
    `g_trial = f(X)*nn.out`
slides likely to be released later
