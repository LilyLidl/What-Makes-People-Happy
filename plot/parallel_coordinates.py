import pandas as pd
import sys
sys.path.append('../')
from preprocess import transform
from pandas.tools.plotting import parallel_coordinates
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

filename_train = '../data/train.csv'
train_dataset = transform(filename_train)
X = train_dataset['data']
y = train_dataset['target']

row_idx = X[:,0]
X = X[:,2:7]
col_name = ['Gender', 'Income', 'HouseholdStatus', 'EducationLevel', 'Party']
X = pd.DataFrame(data=X, index=row_idx, columns=col_name)
y = pd.DataFrame(data=y, index=row_idx, columns=['Happy'])

data = ((pd.concat([X,y], axis=1)).dropna())[:100]
#print(data)

mapping = {1:'Happy', 0:'Unhappy'}
data = data.replace({'Happy': mapping})


fig = plt.figure()
fig.suptitle("Parallel coordinates plot")
parallel_coordinates(data,'Happy', color=['red','blue'])
plt.ylabel("The discrete numbers of the categories")
plt.xlabel("Dimensions")
fig.savefig("paral_coord.jpg")
