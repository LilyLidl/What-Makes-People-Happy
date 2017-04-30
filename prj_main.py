from preprocess import transform
from preprocess import fill_missing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from lr import LogitR
from sklearn.naive_bayes import BernoulliNB
from naive_bayes import NaiveBayes
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
    valid_percent = 0.05
    
    #np.random.seed(32)
    '''
    train_samples=
    valid_samples=
    '''
    
    ### -------------------- use the logistic regression --------------------
    print('\n\nTrain the logistic regression classifier')
    # Sklearn package
    lr_model = LogisticRegression()
    lr_model.fit(X_full[:,1:n_features-1],y)
    print("Sklearn LR learnt coef:\n{0},\n{1}".format(lr_model.coef_[:,:5],lr_model.intercept_))
    
    # Self-implemented
    self_lr = LogitR()
    self_lr.fit(X_full[:,1:n_features-1],y)
    print("Self LR learnt coef:\n{0},\n{1}".format(self_lr.coef[:5],self_lr.intercept))
    ### -------------------- use the logistic regression --------------------
    
    
    '''
    ### -------------------- use the naive bayes --------------------
    # Sklearn package
    print('\n\nTrain the naive bayes classifier')
    nb_model = BernoulliNB()
    nb_model.fit(X_full[:1800,1:n_features-1], y[:1800])
    sk_y_predict = nb_model.predict(X_full[1800:,1:n_features-1])
    
    # Self-implemented
    clf = NaiveBayes()
    clf = clf.fit(X_full[:1800,1:n_features-1], y[:1800])
    self_y_predict = clf.predict(X_full[1800:,1:n_features-1])
    ### -------------------- use the naive bayes --------------------

    ## use the svm
    print('\n\nTrain the SVM classifier')
    # linear, poly, rbf, sigmoid, or precomputed (or self-defined)?
    svm_model = svm.SVC(kernel="linear")
    svm_model.fit(X_full[:,1:], y)


    ## use the random forest
    print('\n\nTrain the random forest classifier')
    """ your code here """
    # rf_model = ...

    ## get predictions
    """ your code here """
    '''
    

if __name__ == '__main__':
    main()
