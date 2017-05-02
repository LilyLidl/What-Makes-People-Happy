from preprocess import transform
from preprocess import fill_missing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from lr import LogitR
from sklearn.naive_bayes import BernoulliNB
from naive_bayes import NaiveBayes
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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
    
    
    # Divide training samples for cross validation:
    valid_percent = 0.3
    
    np.random.seed(32)
    train_idx = np.random.choice(n_samples,int(np.floor(n_samples*(1-valid_percent))),False)
    valid_idx = np.setdiff1d(range(n_samples),train_idx)
    print("Total {0} training samples.".format(len(train_idx)))
    print("Total {0} validation samples.".format(len(valid_idx)))
    
    # Discard the user_id feature(0) and votes feature(n_features-1)
    train_X = X_full[train_idx,1:n_features-1]
    train_y = y[train_idx]
    valid_X = X_full[valid_idx,1:n_features-1]
    valid_y = y[valid_idx]
    #print train_X[0,:]
    #print valid_X[0,:]
    
    
    '''
    ### -------------------- use the logistic regression --------------------
    print('\n\nTrain the logistic regression classifier')
    # Sklearn package
    lr_model = LogisticRegression()
    lr_model = lr_model.fit(train_X,train_y)
    print("Sklearn LR validation score: {0}".format(lr_model.score(valid_X,valid_y)))
    #print("Sklearn LR learnt coef:\n{0},\n{1}".format(lr_model.coef_[:,:5],lr_model.intercept_))
    
    # Self-implemented
    self_lr = LogitR()
    self_lr = self_lr.fit(train_X,train_y)
    print("Self LR validation score: {0}".format(self_lr.score(valid_X,valid_y)))
    #print("Self LR learnt coef:\n{0},\n{1}".format(self_lr.coef[:5],self_lr.intercept))
    ### -------------------- use the logistic regression --------------------
    '''
    
    '''   
    ### -------------------- use the naive bayes --------------------
    # Sklearn package
    print('\n\nTrain the naive bayes classifier')
    nb_model = BernoulliNB()
    nb_model.fit(train_X,train_y)
    print("Sklearn NB validation score: {0}".format(nb_model.score(valid_X,valid_y)))
    #sk_y_predict = nb_model.predict(X_full[1800:,1:n_features-1])
    
    # Self-implemented
    self_nb = NaiveBayes()
    self_nb = self_nb.fit(train_X,train_y)
    print("Self NB validation score: {0}".format(self_nb.score(train_X,train_y)))
    #self_y_predict = clf.predict(X_full[1800:,1:n_features-1])
    ### -------------------- use the naive bayes --------------------
    '''

    '''
    ### -------------------- use svm --------------------
    print('\n\nTrain the SVM classifier')
    # linear, poly, rbf, sigmoid, or precomputed (or self-defined)?
    svm_model = svm.SVC(kernel="linear")
    svm_model.fit(train_X,train_y)
    print("Sklearn SVM validation score: {0}".format(svm_model.score(valid_X,valid_y)))
    ### -------------------- use svm --------------------
    '''

    ### -------------------- use random forest --------------------
    print('\n\nTrain the random forest classifier')
    rf_model = RandomForestClassifier(n_estimators=25)
    rf_model.fit(train_X,train_y)
    print("Sklearn RF validation score: {0}".format(rf_model.score(valid_X,valid_y)))
    ### -------------------- use random forest --------------------
    
    ## get predictions
    """ your code here """
    
    

if __name__ == '__main__':
    main()
