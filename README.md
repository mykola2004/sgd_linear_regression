# Stochastic gradient descent from scratch for linear regression
The goal of the project is to program the gradient descent algorithm for linear regression model.

# Gradient descent for linear regression
Let's say we have a data set with independent features $x_1, x2, ... , x_n$ and dependent feature $y$. And we assume that the dependency between independent features and dependent is given by this model:
\[y = \bold_symbol{x} \cdot \bold_symbol{w} + b, \]
which is called linear regression model. 
Then our task is to find the best in some sense suitable for our dataset weights $w$. To measure how well the model suits the data let's introduce the loss function
\[ 
    L = 1/N(Xw-y)^\top(Xw-y), 
]\
which measures overall squared error that the model makes on each observation of our data set. Naturally, we would want to minimize $L$. We could do that analytically by writing out the derivative of $L$ with respect to $w$ and then finding roots of that derivative, and checking whether in the found root the Hesian of $L$ is positively defined. But, there is an iterative method to find the minimum of the $L$ - stochastic gradient descent. 

# Testing of the algorithm 
In the test.ipynb file I have applied the model to the synthetic dataset. Eventually, the algorithm found the correct approximates for coefficients of independent variables. 