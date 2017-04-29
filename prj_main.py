from preprocess import transform
from preprocess import fill_missing
from sklearn import svm
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
    

    ### use the logistic regression
    print('Train the logistic regression classifier')
    """ your code here """
    # lr_model = ...

    ### use the naive bayes
    print('Train the naive bayes classifier')
    nb_model = BernoulliNB()
    nb_model.fit(X_full[:1800,1:], y)
    sk_y_predict = nb_model.predict()

    # Self-implemented
    clf = NaiveBayes()
    clf = clf.fit(X_full[:1800,:], y)
    self_y_predict = clf.predict()

    ## use the svm
    print('Train the SVM classifier')
    # linear, poly, rbf, sigmoid, or precomputed (or self-defined)?
    svm_model = svm.SVC(kernel="linear")
    svm_model.fit(X_full[:,1:], y)


    ## use the random forest
    print('Train the random forest classifier')
    """ your code here """
    # rf_model = ...

    ## get predictions
    """ your code here """

if __name__ == '__main__':
    main()
