import numpy as np
import math

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_prior_ratio = 0
        self.likelihood_list=[[],[]]

    def fit(self, X, y):
    	happy = [] #C1
    	unhappy=[] #C2
    	for i in range(num_of_samples):
    		if(y[i]==1):
    			happy.append(X[i,:])
    		else:
    			unhappy.append(X[i,:])
    	# Column sums of happy and unhappy
    	happy_sum = happy.sum(axis=0)
    	unhappy_sum = unhappy.sum(axis=0)

        # Calculate log(P(C1)/P(C2))
        # C1: Happy = 1
        # C2: Happy = 0
        num_of_samples, num_of_features = X.shape
        tmp1 = np.sum(y)
        tmp2 = num_of_samples - tmp1
        self.log_prior_ratio = math.log(tmp1 / tmp2)

        # Store 2*d likelihoods for later use, d = num_of_features
        # P(xi=1|C1), P(xi=1|C2) for all i = 1,...,d
        for i in range(num_of_features):
        	likelihood_1i = happy_sum[i] / tmp1 
        	likelihood_2i = unhappy_sum[i] / tmp2
        	likelihood_list[0].append(likelihood_1i)
        	likelihood_list[1].append(likelihood_2i)

        return self

    def predict(self, X):
        # Decision rule:
        # Choose C1 (classify as happy) if log posteriors > 0 
        # Choose C2 otherwise
        num_of_samples, num_of_features = X.shape

        y = []
        for n in range(num_of_samples):
        	likelihood_C1 = 1
        	likelihood_C2 = 1
        	for i in range(num_of_features):
        		# Bernoulli distribution
        		likelihood_C1 *= (self.likelihood_list[0][i] * X[n,i] + (1-self.likelihood_list[0][i])(1-X[n,i]))
        		likelihood_C2 *= (self.likelihood_list[1][i] * X[n,i] + (1-self.likelihood_list[1][i])(1-X[n,i]))
        	log_likelihood = math.log(likelihood_C1 / likelihood_C2)
        	discriminant_result = self.log_prior_ratio + log_likelihood
        	if(discriminant_result > 0):
        		y.append(1)
        	else:
        		y.append(0)

        return y

