import numpy as np
import sys
sys.path.append('../')
from preprocess import transform
from preprocess import fill_missing
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

filename_train = '../data/train.csv'
train_dataset = transform(filename_train)
X = train_dataset['data']
y = train_dataset['target']
    
# fill in missing data (optional)
X_full, discard_row = fill_missing(X, 'most_frequent', False)
y = np.delete(y,discard_row)

pca = PCA(n_components=X_full.shape[1])
pca.fit(X_full)
components = pca.components_
# The first principle component:
component1 = components[0,:]
component2 = components[1,:]


PC1_val = np.dot(X_full, component1)
PC2_val = np.dot(X_full, component2)



fig = plt.figure()
fig.suptitle("PCA biplot")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.scatter(PC1_val,PC2_val,norm=1.0,marker='.')
fig.savefig("PCA_biplot.jpg")
