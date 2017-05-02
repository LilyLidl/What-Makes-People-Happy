import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import sys
sys.path.append('../')
from preprocess import transform
from preprocess import fill_missing
from sklearn import svm

income_id = 2
educate_id = 4
h = 0.2

filename_train = '../data/train.csv'
train_dataset = transform(filename_train)
X = train_dataset['data']
y = train_dataset['target']

'''
row_idx = X[:,0]
mat = X[:,income_id+1:educate_id+2:educate_id-income_id]
col_name = ['Income', 'EducationLevel']
X_show = pd.DataFrame(data=mat, index=row_idx, columns=col_name)
'''


X_full, discard_row = fill_missing(X, 'most_frequent', True)
y_full = np.delete(y,discard_row)


x_min, x_max = min(X[:,income_id])-1, max(X[:,income_id])+1
y_min, y_max = min(X[:,educate_id])-1, max(X[:,educate_id])+1
xx, yy =np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


svm_model = svm.SVC(kernel="linear",C=1)
svm_model.fit(X_full,y_full)

Z = svm_model.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(X_full[:,income_id], X_full[:,educate_id], c=y_full, cmap=plt.cm.coolwarm)
plt.xlabel('Income')
plt.ylabel('Education Level')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("Visualization of SVM result")
plt.savefig("SVM.jpg")
