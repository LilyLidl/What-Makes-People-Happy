import numpy as np
import sys
import sklearn

class LogitR:
    
    def __init__(self, learning_rate=0.0003, max_iter=100000):
        # default assuption:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coef = None
        self.intercept = None
    
    def normalize(self, X):
        return sklearn.preprocessing.normalize(X.astype(float), axis=0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialization of the model parameters:
        np.random.seed(32)
        self.coef = np.random.normal(loc=0.0, scale=0.1, size=(n_features,))
        np.random.seed(32)
        self.intercept = np.random.normal(loc=0.0, scale=0.1)
        
        # Normalize the input data
        X = self.normalize(X)
        
        for i in range(self.max_iter):
            predict_y = self.inner_predict(X)
            max_obj = np.sum(np.dot(y,predict_y)+np.dot(1-y,1-predict_y))
            if i % 10000 == 0:
                sys.stdout.write("\rLogistic Regression: Iteration {0}, Likelihood {1} Error {2}\n".format(i,max_obj,1.0*np.linalg.norm(y-predict_y,ord=1)/n_samples))
                sys.stdout.flush()
        
            #print self.coef[:10]
            grad_coef = np.dot(np.transpose(X),(y-predict_y))
            self.coef = self.coef + self.learning_rate * grad_coef
            #print self.learning_rate * grad_coef[:10]
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
        X = self.normalize(X)
        n_samples, n_features = X.shape
        
        expon = np.exp(-(np.dot(X,self.coef)+self.intercept))
        y = np.round(1 / (1+expon)).astype(int)
        #print y
        
        return y

    def score(self, X, y):
        n_samples = X.shape[0]
        predict_y = self.predict(X)
        
        return 1.0*(n_samples-np.linalg.norm(y-predict_y,ord=1))/n_samples

