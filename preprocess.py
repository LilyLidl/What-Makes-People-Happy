import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
        col=col[~np.isnan(col)]
        return np.median(col)

def getMostFrequent(col):
    col=col[~np.isnan(col)]
    if col.shape[0] <= 0:
        return 0
    # Count and record occurance of all the values
    counter = dict()
    for entry in col:
        if entry not in counter.keys():
            counter[entry] = 0
        else:
            counter[entry] += 1
    # Find the most frequent value
    vals = counter.keys()
    occur = 0
    fre_val = 0
    for val in vals:
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
    
    if isClassified == False:
        for i in range(1,m):
            col = X[:,i]
            # Self-defined strategy for filling missing data:
            # Mean: year of born
            if i == 1:
                sub_val = getMedian(col)
            # Median: income, education level
            if i == 3 or i == 5:
                sub_val = getMean(col)
            # Most Frequent Value: other uncompared labels
            else:
                sub_val = getMostFrequent(col)

            row_id = np.where(np.isnan(X[:,i]))
            X[row_id,i] = sub_val

    else:
        gender_col_index = 2
        edu_col_index = 5
        edu_max = max(X[:,edu_col_index])
        
        # delete data with no gender or  no education info:
        X = np.delete(X,np.where(np.isnan(X[:,gender_col_index])),0)
        X = np.delete(X,np.where(np.isnan(X[:,edu_col_index])),0)
        
        for i in range(1,m):
            
            if i == gender_col_index or i == edu_col_index:
                continue
            
            col = X[:,i]
            for gen in range(2):
                for edu in range(int(edu_max)+1):
                    # Get the indexes of instances belonging to a specific sub-classes:
                    gen_indexes = np.where(X[:,gender_col_index] == gen)
                    edu_indexes = np.where(X[:,edu_col_index] == edu)
                    indexes = np.intersect1d(gen_indexes,edu_indexes)
                    
                    sub_col = col[indexes]
                    # Self-defined strategy for filling missing data:
                    # Mean: year of born
                    if i == 1:
                        sub_val = getMedian(sub_col)
                    # Median: income, education level
                    if i == 3 or i == 5:
                        sub_val = getMean(sub_col)
                    # Most Frequent Value: other uncompared labels
                    else:
                        sub_val = getMostFrequent(sub_col)
                    

                    all_nan_indexes = np.where(np.isnan(X[:,i]))
                    sub_nan_indexes = np.intersect1d(all_nan_indexes,indexes)
                    #print X[sub_nan_indexes,i]
                    X[sub_nan_indexes,i] = sub_val
                    #print X[sub_nan_indexes,i]

    for i in range(m):
        col = X[:,i]
        if np.sum(np.isnan(col)) > 0:
            print "Column {0} has nan values!".format(i)

    X = X.astype(int)

    # Do one-hot encoding on uncomparable attributes:
    hold_col_index = 4
    party_col_index = 6

    append_col_set = np.concatenate(([party_col_index],[hold_col_index]))
    #print append_col_set

    for col_index in append_col_set:
        enc = OneHotEncoder()
        enc.fit(X[:,col_index:col_index+1])
        new_cols = enc.transform(X[:,col_index:col_index+1]).toarray()
        #print X[:10,:8]
        X = np.delete(X,col_index,axis=1)
        #print X.shape
        #print new_cols.shape
        X = np.insert(X,[col_index],new_cols,axis=1)
        #print X[:10,:14]



    return X
