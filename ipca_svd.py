from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import random
import time 

def l2norm(B):
    B = B/np.linalg.norm(B, axis=0)
    return B


def mean_matrix(B):
    miu = np.mat(np.ones((1, B[0, :].size)))*0
    miu = np.mean(B, axis=0)
    B = B - np.mean(B, axis=0)
    return B, miu

#  load data
x_true = datasets.load_iris().data.astype("float64")
y_true = datasets.load_iris().target.reshape(-1, 1).astype("float64")

x_true = l2norm(x_true)  # todo?
y = z = x_true
alpha = 0.8
n = 50
B = y[0: n]

start = time.clock()
n_features = 2
miu = np.mat(np.ones((1, B[0, :].size)))*0
cov_mat = np.cov(B.transpose())
U, S, V = np.linalg.svd(cov_mat)




for k in range(len(B)+1, len(z)):
    yy = np.zeros(shape=(k, len(y[1])))
    yy, miu = mean_matrix(z[0:k])  # reload x_ture dont know why
    miu = miu + (1-alpha)*y[k]
    B = yy
    delta_sum = 0
    for j in range(S.size):
        for l in range(len(B)):
            delta = np.dot(S[j].transpose(), B[l].transpose())**2
            delta_sum = delta_sum + delta
    if (random.random() <= delta_sum).any():
        new_y = np.mat(np.ones((B[0, :].size, B[0, :].size)))*0  # !!!!!
        for m in range(S.size):
            new_y[m, :] = np.dot(np.sqrt(alpha*S[m].transpose()), 
                                 U[:, m])                                
        new_y = np.row_stack((new_y, np.sqrt(1-alpha)*B[-1]))
        cov_mat = np.cov(new_y.transpose())
        U, S, V  = np.linalg.svd(cov_mat)
SIndice=np.argsort(S)
n_SIndice = SIndice[-1:-(n_features+1):-1]
n_U = U[:,n_SIndice]


x_true_rot = np.dot(x_true, n_U)
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
plt.title("IPCA WITH SVD", fontsize=16)
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
plt.savefig("./1-PCA数据降维效果图for_ipca_svd.jpg")
print("利用PCA后的前"+str(n_features)+"维度特征对鸢尾花数据集分类精度为：")
print("score= ", score)
