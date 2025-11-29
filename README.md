# Stochastic gradient descent from scratch for linear regression
The goal of the project is to program the stochastic gradient descent algorithm for linear regression model ($L_2$ regularized). Fristly, small chunk of theory will be given with introduction to the linear regression model, describing the gradient descent method for optimization, derivation of gradients for parameters, description of mini-batch gradient descent. At the end, will mentioned few words about validation of the implemented algorithm. 

# Theoretical description of gradient descent for $L_2$ regularized linear regression
Let's say we have a data set with independent features $X_1, X_2, ... , X_n$ and the dependent feature $Y$. And we assume that the dependency between independent features and the dependent is given by this model: $$Y = (X_1, X_2, ..., X_n) \boldsymbol{\cdot} \boldsymbol{\omega} + b$$ ,
which is called linear regression model; where $\omega \in \mathbb{R^{n}}$ is the weights vector and $b \in \mathbb{R}$ is the bias term. 

Then our task is to find the best in some sense suitable for our dataset set of weights $w$ and bias $b$ term. To measure how well the model suits the data let's introduce a loss function $$L(\omega, b) = \frac{1}{m}(X\omega+b-y)^\top(X\omega+b-y) + \lambda \omega^\top \omega $$, where $m$ - is the number of observations in the dataset; $X$ - is $m$ by $n$ matrix, which $i$-th row corresponds to $i$-th observation. The second term in $L$ is responsible for regularization, which helps when multicolinearity exists between independent features; regularization forces the weights vector to become smaller (how small it would be depends on the value of $\lambda$ - regularization parameter). $L$ measures overall squared error that the model makes on each observation of our data set. 

Naturally, we want to minimize $L$, which we could do analytically by writing out the derivative of $L$ with respect to $w$ and then finding roots of that derivative, and then checking whether in the found root the Hesian of $L$ is positively definite. 

But, there is also an iterative method to find the minimum of the $L$ - stochastic gradient descent. The idea of which is to consequently update all parameters of the model in the direction of antigradient (as it points in the direction of steepest descent of a function): $\omega^{(t+1)} = \omega^{(t)} - \alpha \frac{dL}{d\omega}$, $b^{(t+1)} = b^{(t)} - \alpha \frac{dL}{db}$, where $t$ - is the number of the iteration, $\alpha$ - is learning rate, which determines the size of step in the direction of antigradient. The algorithm stops when the maximum number of iterations is reached or the update term becomes too small (in the project I will be checking the norm of update vector only for weights $\omega$).

In the next section, will be derived the gradients of the cost function with respect to parameters of linear regression model.

# Derivatives of loss function with respect to parameters $\omega$ and $b$
$$
\frac{dL}{d\omega} = \frac{1}{\omega} ( (\frac{d}{d\omega} (X\omega+b-y))^\top (X\omega+b-y) + (X\omega+b-y)^\top \frac{d}{d\omega}(X\omega+b-y)) + \lambda ( (\frac{d}{d\omega} \omega)^\top \omega + \omega^\top \frac{d}{d\omega} \omega) = 
\frac{1}{\omega} (X^\top (X\omega+b-y) + (X\omega+b-y)^\top X) + \lambda (2\omega)) = \frac{1}{\omega} (X^\top (X\omega+b-y) + X^\top (X\omega+b-y)) + \lambda (2\omega)) = \frac{1}{m} (2X^\top (X\omega+b-y) + 2 \lambda \omega)
$$

$$
\frac{dL}{db} = \frac{1}{m} ( (\frac{d}{db} (X\omega+b-y))^\top (X\omega+b-y) + (X\omega+b-y)^\top \frac{d}{db}(X\omega+b-y)) = 
\frac{2}{m} \sum_{1}^{m}(X\omega+b-y)
$$ 

# Batch gradient descent
Weights can be updated after processing smaller part of a dataset - called mini-batch. This increases the amount of updates applied to the parameters, speeding up learning process. In practice at the beginning of each epoch the dataset is shuffled (this part is skipped in my implementation), then it is divided into subsequent mini-batches of size $k$ ($$ 1 <= k = <m $$), the updates for the weights are performed consequnetly for each of the extracted mini-batches.

# Testing of the algorithm 
In the test.ipynb file I have applied the model to the synthetic dataset. Can be seen that the algorithm found optimal weights correctly by comparing found coeficients and the real coefficients in the functional dependency between indepedent features and the dependent one.
