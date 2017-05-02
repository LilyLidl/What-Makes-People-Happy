import numpy as np
from preprocess import transform
from preprocess import fill_missing
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

filename_train = '../data/train.csv'
train_dataset = transform(filename_train)
X = train_dataset['data']
y = train_dataset['target']
    
# fill in missing data (optional)
X_full, discard_row = fill_missing(X, 'most_frequent', False)
y = np.delete(y,discard_row)
print X_full

