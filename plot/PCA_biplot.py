import numpy as np
import sys
sys.path.append('../')
from preprocess import transform
from preprocess import fill_missing
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn

filename_train = '../data/train.csv'
train_dataset = transform(filename_train)
X = train_dataset['data']
y = train_dataset['target']
    
# fill in missing data (optional)
X_full, discard_row = fill_missing(X, 'most_frequent', False)
X_full = X_full[:,1:X_full.shape[1]-1]
X_full = sklearn.preprocessing.normalize(X_full.astype(float), axis=0)
y = np.delete(y,discard_row)

pca = PCA(n_components=X_full.shape[1])
pca.fit(X_full)
components = pca.components_
# The first principle component:
component1 = components[0,:]
component2 = components[1,:]

n_features = X_full.shape[1]
year_unit = np.array([10]+[0]*(n_features-1))
income_unit = np.array([0,0]+[1]+[0]*(n_features-3))
edu_unit = np.array([0]*9+[1]+[0]*(n_features-10))

PC1_val = np.dot(X_full, component1)
PC2_val = np.dot(X_full, component2)



fig = plt.figure()
fig.suptitle("PCA biplot")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.scatter(PC1_val,PC2_val,norm=1.0,marker='.',alpha=0.4)

ax=plt.gca()
ax.quiver([0]*3,[0]*3,[np.dot(year_unit,component1),np.dot(income_unit,component1),np.dot(edu_unit,component1)],[np.dot(year_unit,component2),np.dot(income_unit,component2),np.dot(edu_unit,component2)],color=['r','g','b'])


fig.savefig("PCA_biplot.jpg")
