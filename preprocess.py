import pandas as pd
import numpy as np
import math

def map_income(income):
    if(income == "under $25,000"):
        return 0
    elif(income == "$25,001 - $50,000"):
        return 1
    elif(income == "$50,000 - $74,999"):
        return 2
    elif(income == "$75,000 - $100,000"):
        return 3
    elif(income == "$100,001 - $150,000"):
        return 4
    elif(income == "over $150,000"):
        return 5

def map_education(edu):
    if(edu == "Current K-12"):
        return 0
    elif(edu == "High School Diploma"):
        return 1
    elif(edu == "Associate's Degree"):
        return 2
    elif(edu == "Current Undergraduate"):
        return 3
    elif(edu == "Bachelor's Degree"):
        return 4 
    elif(edu == "Master's Degree"):
        return 5
    elif(edu == "Doctoral Degree"):
        return 6


def transform(filename):
    """ preprocess the training data"""
    """ your code here """
    df = pd.read_csv(filename)

    # Map Income
    df['Income'] = df['Income'].map(map_income, na_action='ignore')

    #print(df)
    # Map EducationLevel
    df['EducationLevel'] = df['EducationLevel'].map(map_education, na_action='ignore')
    
    # Map Other Attributes
    for col_name, values in df.iteritems():
        #print('{name}: {value}'.format(name=name, value=values[0]))
        if(col_name in ["UserID", "YOB", "Income", "EducationLevel", "Happy", "votes"]):
            continue
        else:
            unique_value_list = df[col_name].unique()
            unique_value_list = [value for value in unique_value_list if not pd.isnull(value)]
            unique_value_list = sorted(unique_value_list)
            num_unique_value = len(unique_value_list)
            tmp_dict={}
            i=0
            for value in unique_value_list:
                if(pd.isnull(value)):
                    continue
                else:
                    tmp_dict[value] = i
                    i += 1
            df[col_name] = df[col_name].map(tmp_dict,na_action='ignore')

  
    return {'data':df.as_matrix(),'target':df["Happy"].as_matrix()} 
    #return {'data':data,'target':target}

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
