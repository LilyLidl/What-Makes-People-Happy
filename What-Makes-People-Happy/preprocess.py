import numpy as np

def transform(filename):
    """ preprocess the training data"""
    """ your code here """
    data = np.zeros((100,100))
    target = np.zeros(100)
    return {'data':data,'target':target}


def getMean(col):
    col_sum = np.sum(np.nan_to_num(col))
    col_count = np.sum(np.isnan(col))
    if col_count != 0:
        return 1.0*col_sum/col_count
    else:
        return 0

def getMedian(col):
    if col.shape[0] <= 0:
        return 0
    else:
        return np.median(col)

def getMostFrequent(col):
    if col.shape[0] <= 0:
        return 0
    # Count and record occurance of all the values
    counter = dict()
    for entry in col:
        if not counter.has_key(entry):
            counter[entry] = 0
        else:
            counter[entry] += 1
    # Find the most frequent value
    vals = counter.keys()
    occur = 0
    fre_val = 0
    for val in vals
        if counter[val] > occur:
            fre_val = val
            occur = counter[val]
    return fre_val


def fill_missing(X, strategy, isClassified):
    """
     @X: input matrix with missing data filled by nan
     @strategy: string, 'median', 'mean', 'most_frequent'
     @isclassfied: boolean value, if isclassfied == true, then you need build a
     decision tree to classify users into different classes and use the
     median/mean/mode values of different classes to fill in the missing data;
     otherwise, just take the median/mean/most_frequent values of input data to
     fill in the missing data
    """
    
    
    """ your code here """
    (n, m) = X.shape
    col = X[:,i]
    
    if isClassified == false:
        for i in xrange(m):
            if strategy == 'median':
                sub_val = getMedian(col)
            if strategy == 'mean':
                sub_val = getMean(col)
            if strategy == 'most_frequent'
                sub_val = getMostFrequent(col)
            X[:,i] = np.nan_to_num(sub_val)

    else:
        gender_col_index = 2
        edu_col_index = 5
        edu_max = max(X[:,edu_col_index])
        for gen in xrange(2):
            for edu in xrange(edu_max+1):
                gen_indexes = np.where(X[:,gender_col_index] == gen)
                edu_indexes = np.where(X[:,edu_col_index] == edu)
                


    return X_full
