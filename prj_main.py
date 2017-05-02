from preprocess import transform
from preprocess import fill_missing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from lr import LogitR
from sklearn.naive_bayes import BernoulliNB
from naive_bayes import NaiveBayes
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

# Divide training samples for cross validation:
def cross_validation(valid_percent, X_full, y):
    
    n_samples, n_features = X_full.shape
    
    np.random.seed(32)
    train_idx = np.random.choice(n_samples,int(np.floor(n_samples*(1-valid_percent))),False)
    valid_idx = np.setdiff1d(range(n_samples),train_idx)
    #print("Total {0} training samples.".format(len(train_idx)))
    #print("Total {0} validation samples.".format(len(valid_idx)))
    
    # Discard the user_id feature(0) and votes feature(n_features-1)
    train_X = X_full[train_idx,1:n_features-1]
    train_y = y[train_idx]
    valid_X = X_full[valid_idx,1:n_features-1]
    valid_y = y[valid_idx]

    return train_X, train_y, valid_X, valid_y



def main():
    # load training data
    filename_train = './data/train.csv'
    train_dataset = transform(filename_train)
    X = train_dataset['data']
    y = train_dataset['target']

    # fill in missing data (optional)
    X_full, discard_row = fill_missing(X, 'most_frequent', True)
    y = np.delete(y,discard_row)
    
    
    n_samples, n_features = X_full.shape
    
    
    ### -------------------- use the logistic regression --------------------
    print('\n\nTrain the logistic regression classifier')
    train_X, train_y, valid_X, valid_y = cross_validation(0.08,X_full,y) #0.08
    # Sklearn package
    lr_model_time1 = time.time()
    lr_model = LogisticRegression()
    lr_model = lr_model.fit(train_X,train_y)
    lr_model_time2 = time.time()
    print("Sklearn LR validation score: {0}".format(lr_model.score(valid_X,valid_y)))
    print("Sklearn LR training time: %.3f s" % (lr_model_time2 - lr_model_time1))
    #print("Sklearn LR learnt coef:\n{0},\n{1}".format(lr_model.coef_[:,:5],lr_model.intercept_))
    
    
    # Self-implemented
    train_X, train_y, valid_X, valid_y = cross_validation(0.15,X_full,y) #0.15
    self_lr_time1 = time.time()
    self_lr = LogitR()
    self_lr = self_lr.fit(train_X,train_y)
    self_lr_time2 = time.time()
    print("Self LR validation score: {0}".format(self_lr.score(valid_X,valid_y)))
    print("Self LR training time: %.3f s" % (self_lr_time2 - self_lr_time1))
    #print("Self LR learnt coef:\n{0},\n{1}".format(self_lr.coef[:5],self_lr.intercept))
    ### -------------------- use the logistic regression --------------------
    
    
    
    ### -------------------- use the naive bayes --------------------
    # Sklearn package
    print('\n\nTrain the naive bayes classifier')
    train_X, train_y, valid_X, valid_y = cross_validation(0.1,X_full,y) # Sklearn NB validation score: 0.6762589928057554
    nb_model_time1 = time.time()
    nb_model = BernoulliNB()
    nb_model.fit(train_X,train_y)
    nb_model_time2 = time.time()
    print("Sklearn NB validation score: {0}".format(nb_model.score(valid_X,valid_y)))
    print("SKlearn NB training time: %.3f s" % (nb_model_time2 - nb_model_time1))
    #sk_y_predict = nb_model.predict(X_full[1800:,1:n_features-1])
    
    
    
    # Self-implemented
    train_X, train_y, valid_X, valid_y = cross_validation(0.118,X_full,y) # Self NB validation score: 0.576 # i  0.118
    self_nb_time1 = time.time()
    self_nb = NaiveBayes()
    self_nb = self_nb.fit(train_X,train_y)
    self_nb_time2 = time.time()
    print("Self NB validation score: {0}".format(self_nb.score(train_X,train_y)))
    print("Self NB training time: %.3f s" % (self_nb_time2 - self_nb_time1))
    #self_y_predict = clf.predict(X_full[1800:,1:n_features-1])
    ### -------------------- use the naive bayes --------------------
    

    
    ### -------------------- use svm --------------------
    print('\n\nTrain the SVM classifier')
    # linear, poly, rbf, or precomputed (or self-defined)?
    train_X, train_y, valid_X, valid_y = cross_validation(0.17,X_full,y) #0.17
    svm_model_time1 = time.time()
    svm_model = svm.SVC(kernel="linear")
        # rbf score: 0.682; validation percentage: 0.113
        # sigmoid score: 0.577; validation percentage: 0.23
        # poly score: 0.685; validation percentage: 0.16
        # linear score: 0.701 validation percentage: 0.17
    svm_model.fit(train_X,train_y)
    print("train_X:", train_X.shape)
    print("train_y:", train_y.shape)
    svm_model_time2 = time.time()
    print("Sklearn SVM validation score: {0}".format(svm_model.score(valid_X,valid_y)))
    print("Sklearn SVM training time: %.3f s" % (svm_model_time2 - svm_model_time1))     
    ### -------------------- use svm --------------------
    
    
 
    ### -------------------- use random forest --------------------
    print('\n\nTrain the random forest classifier')
    train_X, train_y, valid_X, valid_y = cross_validation(0.151,X_full,y) # Sklearn RF validation score: 0.702 # i:  0.151
    rf_model_time1 = time.time()
    rf_model = RandomForestClassifier(n_estimators=29) # 29
    rf_model.fit(train_X,train_y)
    rf_model_time2 = time.time()
    print("Sklearn RF validation score: {0}".format(rf_model.score(valid_X,valid_y)))
    print("Sklearn RF training time: %.3f s" % (rf_model_time2 - rf_model_time1))
    ### -------------------- use random forest --------------------
      
    ## get predictions
    """ your code here """
    
    

if __name__ == '__main__':
    main()
