from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import random
import time
def l2norm(B):
    B = B/np.linalg.norm(B, axis=0)
    B = B - np.mean(B, axis=0)
    
    return B

x_true = datasets.load_iris().data.astype("float64")
y_true = datasets.load_iris().target.reshape(-1, 1).astype("float64")

y = x_true 

b = 30  
n_features = 2
n_part = int(len(y)/b)
B = np.zeros(shape=(b, len(y[1])))

start = time.clock()
B = z = y[0: 29]
cov_mat = np.cov(B.transpose())
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

for t in range(n_part):
    z = y[b*t: b*(t+1)]
    for i in range(b):
        B = l2norm(z)
        delta_sum = 0
        for j in range(eigen_vecs[0, :].size):
            for l in range(len(B)):
                delta = np.dot(eigen_vecs[:, j].transpose(), B[l].transpose())**2
                delta_sum = delta_sum + delta
        if (random.random() <= delta_sum).any():
            # print(B[i])
            B[i] = B[i]/(b*np.sqrt(delta_sum))
            c = np.cov(B.transpose())
            eigen_vals, eigen_vecs = np.linalg.eig(c)
eigValIndice=np.argsort(eigen_vals)
n_eigValIndice = eigValIndice[-1:-(n_features+1):-1]
n_eigVect = eigen_vecs[:,n_eigValIndice]

x_true_rot = np.dot(x_true, eigen_vecs.transpose())
elapsed = (time.clock() - start)
print("Time used:",elapsed)





# SVM PART AND SCORE IT
clf = svm.SVC()
clf.fit(x_true_rot[:, 0:n_features], y_true.reshape(y_true.shape[0], ))
y_predict = clf.predict(x_true_rot[:, 0:n_features])
score = clf.score(x_true_rot[:, 0:n_features], y_true)


# PLOT PART
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
plt.title("OPCA", fontsize=16)
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
plt.savefig("./1-PCA数据降维效果图for_opca.jpg")
print("利用PCA后的前"+str(n_features)+"维度特征对鸢尾花数据集分类精度为：")
print("score= ", score)


