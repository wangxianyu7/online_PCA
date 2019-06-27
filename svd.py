from sklearn import datasets
from sklearn import svm
import time
import matplotlib.pyplot as plt
import numpy as np


x_true = datasets.load_iris().data.astype("float64")
y_true = datasets.load_iris().target.reshape(-1, 1).astype("float64")
n_features = 2

def PCA_DATA(x_true):
    x_true -= np.mean(x_true, axis=0)
    cov = np.dot(x_true.T, x_true) / x_true.shape[0]
    U, S, V = np.linalg.svd(cov)
    SIndice=np.argsort(S)
    n_SIndice = SIndice[-1:-(n_features+1):-1]
    n_U = U[:,n_SIndice]
    x_true_rot = np.dot(x_true, n_U)
    return x_true_rot

start = time.clock()
x_true_rot = PCA_DATA(x_true)
elapsed = (time.clock() - start)
print("Time used:",elapsed)

# SVM PART AND SCORE IT
clf = svm.SVC()
clf.fit(x_true_rot[:, 0:n_features], y_true.reshape(y_true.shape[0], ))
y_predict = clf.predict(x_true_rot[:, 0:n_features])
score = clf.score(x_true_rot[:, 0:n_features], y_true)


plt.figure(figsize=(16, 12))
plt.subplot(131)
plt.xlabel("Before PCA data's first two features", fontsize=16)
ax = plt.gca()
scatter = ax.scatter(x_true[:, 0], x_true[:, 1], 
            c= y_true.reshape(y_true.shape[0], ), lw= 3)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.tick_params(labelsize=14)
plt.subplot(132)
plt.xlabel("After PCA data's first two features", fontsize=16)
plt.title("PCA WITH SVD", fontsize=16)
ax = plt.gca()
scatter = ax.scatter(x_true_rot[:, 0], x_true_rot[:, 1], 
            c= y_true.reshape(y_true.shape[0], ), lw= 3)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.tick_params(labelsize=14)
plt.subplot(133)
plt.xlabel("Use first two features after PCA to SVM", fontsize=16)
ax = plt.gca()
scatter = ax.scatter(x_true_rot[:, 0], x_true_rot[:, 1], 
            c= y_predict.reshape(y_true.shape[0], ), lw= 3)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.tick_params(labelsize=14)
plt.savefig("./1-PCA数据降维效果图for_svd.jpg")
print("利用PCA后的前"+str(n_features)+"维度特征对鸢尾花数据集分类精度为：")
print("score= ", score)

