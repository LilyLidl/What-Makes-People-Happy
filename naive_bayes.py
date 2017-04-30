import numpy as np
import math

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        # p(C1): happy class prior
        self.pos_prior = 0
        # p(C2): unhappy class prior
        self.neg_prior = 0
        # p(x|C1): happy class likelihood
        self.pos_likelihood = None
        # p(x|C2): unhappy class likelihood
        self.neg_likelihood = None

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        
        self.pos_likelihood = [dict()] * self.n_features
        self.neg_likelihood = [dict()] * self.n_features
        
        # Calculate the priors:
        pos_idx = np.where(y == 1)
        neg_idx = np.where(y == 0)
        self.pos_prior = 1.0 * pos_idx[0].shape[0] / self.n_samples
        self.neg_prior = 1.0 * neg_idx[0].shape[0] / self.n_samples
        
        # Calculate the likelihood:
        for i in range(self.n_features):
            # Scanning the i-th feature
            col = X[:,i]
            pos_col = col[pos_idx]
            neg_col = col[neg_idx]
            value_set = set(col.flatten())
            
            for value in value_set:
                pos_match_idx = np.where(pos_col == value)
                neg_match_idx = np.where(neg_col == value)
                self.pos_likelihood[i][value] = 1.0 * pos_match_idx[0].shape[0] / pos_col.shape[0]
                self.neg_likelihood[i][value] = 1.0 * neg_match_idx[0].shape[0] / neg_col.shape[0]
    
        return self

    def predict(self, X):
        # Decision rule:
        # Choose C1 (classify as happy) if p(x|C1)p(C1) > p(x|C2)p(C2)
        # Choose C2 otherwise
        n_samples, n_features = X.shape

        y = []
        for i in range(n_samples):
            posterior_C1 = self.pos_prior
            posterior_C2 = self.neg_prior
            for j in range(n_features):
                factor = X[i,j]
                posterior_C1 *= self.pos_likelihood[j][factor]
                posterior_C2 *= self.neg_likelihood[j][factor]
            
            if(posterior_C1 > posterior_C2):
                y.append(1)
            else:
                y.append(0)

        return np.array(y)

    def score(self, X, y):
        n_samples = X.shape[0]
        predict_y = self.predict(X)
        
        return 1.0*(n_samples-np.linalg.norm(y-predict_y,ord=1))/n_samples

