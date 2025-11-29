# Stochastic gradient descent from scratch for linear regression
The goal of the project is to program the gradient descent algorithm for linear regression model ($L_2$ regularized).

# Theoretical description of gradient descent for $L_2$ regularized linear regression
Let's say we have a data set with independent features $X_1, X_2, ... , X_n$ and dependent feature $Y$. And we assume that the dependency between independent features and the dependent is given by this model: $$Y = (X_1, X_2, ..., X_n) \cdot \boldsymbol{w} + b$$ ,
which is called linear regression model; $w$ is the parameters vector and $b$ is the bias term. Then our task is to find the best in some sense suitable for our dataset weights $w$ and bias $b$. To measure how well the model suits the data let's introduce the loss function $$L = \frac{1}{m}(Xw+b-y)^\top(Xw+b-y) + \lambda w^\top w $$, where $m$ - is the number of observations in a dataset and $X$ - is $m$ by $n$ matrix, which $i$-th row corresponds to $i$-th observation. The second term in $L$ is responisble for regularization, which helps when multicolinearity exists between independent features; simply forces the weights vector to be small. $L$ measures overall squared error that the model makes on each observation of our data set. Naturally, we would want to minimize $L$, which we could do analytically by writing out the derivative of $L$ with respect to $w$ and then finding roots of that derivative, and checking whether in the found root the Hesian of $L$ is positively definite. But, there is also an iterative method to find the minimum of the $L$ - stochastic gradient descent. The idea of which is to consequently update all parameters of the model in the direction of antigradient (as it points in the direction of steepest descent of a function): $w^{(t+1)} = w^{(t)} - \alpha \frac{dL}{dw}$, $b^{(t+1)} = b^{(t)} - \alpha \frac{dL}{db}$, where $t$ - is the number of the iteration. The algorithm stops when the maximum number of iterations is reached or the update term became to small (in the project I will be checking the norm of update vector only for weights $w$).

# Derivatives of loss fucntion
$$
\frac{dL}{dw} = \frac{1}{m} ( (\frac{d}{dw} (Xw+b-y))^\top (Xw+b-y) + (Xw+b-y)^\top \frac{d}{dw}(Xw+b-y)) + \lambda ( (\frac{d}{dw} w)^\top w + w^\top \frac{d}{dw} w) = 
\frac{1}{m} (X^\top (Xw+b-y) + (Xw+b-y)^\top X) + \lambda (2w)) = \frac{1}{m} (X^\top (Xw+b-y) + X^\top (Xw+b-y)) + \lambda (2w)) = \frac{1}{m} (2X^\top (Xw+b-y) + 2 \lambda)
$$

$$
\frac{dL}{db} = \frac{1}{m} ( (\frac{d}{db} (Xw+b-y))^\top (Xw+b-y) + (Xw+b-y)^\top \frac{d}{db}(Xw+b-y)) = 
\frac{1}{m} (X^\top (Xw+b-y) + (Xw+b-y)^\top X
$$

# Testing of the algorithm 
In the test.ipynb file I have applied the model to the synthetic dataset. Eventually, the algorithm found the correct approximates for coefficients of independent variables. 
