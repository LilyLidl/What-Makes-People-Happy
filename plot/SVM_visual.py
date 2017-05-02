import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import sys
sys.path.append('../')
from preprocess import transform
from preprocess import fill_missing
from sklearn import svm

income_id = 3
educate_id = 10
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
#X_full = X_full[:,1:X_full.shape[1]-1]
X_full = X_full[:,income_id:educate_id+1:educate_id-income_id]
y_full = np.delete(y,discard_row)


#x_min, x_max = min(X_full[:,income_id])-1, max(X_full[:,income_id])+1
#y_min, y_max = min(X_full[:,educate_id])-1, max(X_full[:,educate_id])+1
x_min, x_max = min(X_full[:,0])-1, max(X_full[:,0])+1
y_min, y_max = min(X_full[:,1])-1, max(X_full[:,1])+1
xx, yy =np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


svm_model = svm.SVC(kernel="rbf")
#svm_model = svm.LinearSVC(C=1)
svm_model.fit(X_full,y_full)
print("SVM end")

print(np.c_[xx.ravel(),yy.ravel()])
Z = svm_model.predict(np.c_[xx.ravel(),yy.ravel()])
print Z
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

#plt.scatter(X_full[:,income_id], X_full[:,educate_id], c=y_full, cmap=plt.cm.coolwarm)
plt.scatter(X_full[:,0], X_full[:,1], c=y_full, cmap=plt.cm.coolwarm)

plt.xlabel('Income')
plt.ylabel('Education Level')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("Visualization of SVM result")
plt.savefig("SVM.jpg")
