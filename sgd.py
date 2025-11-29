import numpy as np
from sklearn.base import RegressorMixin
from numpy import linalg as LA

class SGDLinearRegressor(RegressorMixin):
    def __init__(self, lr=0.01, regularization=1, delta_converged=1e-3, max_steps=1000, batch_size=64):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None
        self.b = None

    def fit(self, X, Y):
        self.W = np.random.normal(loc=0, scale=0.01, size=X.shape[1]) #Initialization of weights as small random values
        self.b = 0 #Intialization of bias term as 0

        step = 0 #counter for completed epochs
        update_too_small = False #indicator for an update term in weights becoming too small 
        while step < self.max_steps and update_too_small == False: #continue the algorithm untill the maximum number of epochs is reached or the update term for weights becomes too small
            for i in range(0, len(X), self.batch_size): #loop through the training dataset with the step equal to size of one batch
                X_train, y_train = X[i:i+self.batch_size], Y[i:i+self.batch_size] #separate one batch from a dataset
                
                #calculate the update terms for weights and bias
                step_w = self.lr * (2/self.batch_size) * np.transpose(X_train)@(X_train@self.W + self.b - y_train) + 2*self.lr*self.regularization*self.W
                step_b = self.lr * (2/self.batch_size) * np.sum((X_train@self.W + self.b - y_train))

                if LA.norm(step_w) < self.delta_converged: #if the update term for weights became smaller than threshold, turn the indicator and escape the current loop
                    update_too_small = True
                    break
                
                self.W -= step_w #update the weights and bias in the direction of antigradient
                self.b -= step_b
                
            step += 1 #increase the number of epoch
            
        return self

    def predict(self, X):
        #calculate predictions using passed data X and weights learned previously
        y_pred = (X@self.W + self.b)
        return y_pred
    
    def get_weights_and_bias(self):
        #return fitted weights and bias term 
        return (self.W, self.b)