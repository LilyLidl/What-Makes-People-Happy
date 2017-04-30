import numpy as np
import sys
import sklearn

class LogitR:
    def __init__(self, learning_rate=0.003, max_iter=10000):
        # default assuption:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialization of the model parameters:
        self.coef = np.random.normal(loc=0.0, scale=0.1, size=(n_features,))
        self.intercept = np.random.normal(loc=0.0, scale=0.1)
        
        # Normalize the input data
        X = sklearn.preprocessing.normalize(X.astype(float), axis=1)
        
        for i in range(self.max_iter):
            predict_y = self.inner_predict(X)
            max_obj = np.sum(np.dot(y,predict_y)+np.dot(1-y,1-predict_y))
            if i % 100 == 1:
                sys.stdout.write("\rLogistic Regression: Iteration {0}, Likelihood {1} Error {2}".format(i,max_obj,1.0*np.linalg.norm(y-predict_y,ord=1)/n_samples))
                sys.stdout.flush()
        
            #print self.coef[:10]
            grad_coef = np.dot(np.transpose(X),(y-predict_y))
            self.coef = self.coef + self.learning_rate * grad_coef
            #print grad_coef[:15]
            #print self.coef[:10]
        
            grad_intercept = np.sum(y-predict_y)
            self.intercept = self.intercept + self.learning_rate * grad_intercept
        
        return self


    def inner_predict(self, X):
        n_samples, n_features = X.shape
        expon = np.exp(-(np.dot(X,self.coef)+self.intercept))
        y = 1 / (1+expon)
        return y
    


    def predict(self, X):
        X = sklearn.preprocessing.normalize(X.astype(float), axis=1)
        n_samples, n_features = X.shape
        
        expon = np.exp(-(np.dot(X,self.coef)+self.intercept))
        y = np.round(1 / (1+expon)).astype(int)
        #print y
        
        return y

