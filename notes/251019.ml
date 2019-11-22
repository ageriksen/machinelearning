
# Decision trees

## regression
## classification

*Decision tree + Forests*
    : split your area into boxes

### regression example
                            |root node|
                            /           \
             |internal node|            |internal node|
            /               \           /              \
       |leaf|               |leaf|  |leaf|             |leaf|

    
    ------------
    |r2|    |r5|
    |--| r3 |--|
    |r1|    |r4|
    ------------

*mainly for regression. classification is difference.*
    : regression <=, classification different

### classification example

target/output: yes = ride   no = stay home

features/preditcors
    
        weather outlook (o)
       /        |           \
    sunny      overcast     rainy



    day | O         | T | H | W | RIDE
    -----------------------------------
    1   |sun        |   |   |   |
    2   |sun        |   |   |   |
    3   |overcast   |   |   |   |
    4   |rain       |   |   |   |
    5   |           |   |   |   |
    6   |           |   |   |   |
_fill in the rest of the table_
*our tree*

                                    |outlook| <- root node
                     sunny-> /          | <-overcast    \ <-rain
                          |H|           y               |W|
               high-> /         \<-normal         T->/      \ <-F
                    N           Y                   N       Y


deciding importance of features

    - methods to classify
        -   cross/information/-entropy   =>  -sum_{k=1}^M P_k \log{ P_k }
            
                14 events
                --------
                9 days w rides --> P=9/14
    
                5 days w/o rides --> P=5/14


### misclassification error
    
in a node -m-, representing region R\_m

- misclassification
- gini index    ]
- entropy       ]
    ] differentiable

## Entropy and ID3 algo
    :   most used algorithm for decision trees with regards to classification

*entropy without features*
    Entropy uses log_2

*entropy for features*

*entropy of features/categories*
    \# feature events/\#total events * entropy_of_feature
and sum over the proportioanl fearure entropies

### information gain for a specific feture
Total outcome entropy - entropy of specific feature
check for each feature and set the one with teh largest information gain. 
Use the feature with the largest information gain is the root node. 
then take away the chose tree and redo for the next nodes downward. 

    
                                    |outlook| <- root node
                     sunny-> /          | <-overcast    \ <-rain
                          |H|           Y              |W|
               high-> /         \<-normal         T->/      \ <-F
                    N           Y                   N       Y

###pseudocode Classification
'''
    - create a list of features and categories
    - find maximum information gain amongst features. This is then the root node. 
    - remove feature chosen as root and find new max feature(largest information gain)
      for each specific branch and repeat until you reach the final internal node
'''
this bit should be included in the lecture slides later, with example code


