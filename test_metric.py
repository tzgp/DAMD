from metric_learn import Covariance
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()['data']
print("iris.shape",iris.shape)
cov = Covariance().fit(iris)
x = cov.transform(iris)
print("x",x.shape)
#iris.shape (150, 4)
#x (150, 4)
# method.patch_xyz_lib.shape torch.Size([76518, 33])
# method.patch_rgb_lib.shape torch.Size([7651, 768])
#
# from metric_learn import RCA
# #chunks = [0, 0, 1, 1, 2, 2, 3, 3]
# rca = RCA()
# chunks = np.ones(150)
# rca.fit(iris,chunks)
# xr =rca.transform(iris)
# print("xr",xr.shape)# (150, 4)
#
#
#
# from sklearn.decomposition import PCA
# pca=PCA()
# pca.fit(iris)
# xp=pca.transform(iris)# (150, 4)
# print("xp",xp.shape)#(150, 4)
# pca=PCA(n_components=2)
# pca.fit(iris)
# xp=pca.transform(iris)# (150, 4)
# print("xp",xp.shape)#(150, 2)
#
#
# # >> > import numpy as np
# from sklearn.datasets import make_sparse_coded_signal
#
# from sklearn.decomposition import DictionaryLearning
# # X, dictionary, code = make_sparse_coded_signal(
# # n_samples = 30, n_components = 15, n_features = 20, n_nonzero_coefs = 10,random_state = 42,)
# dict_learner = DictionaryLearning(n_components = 15, transform_algorithm = 'lasso_lars', transform_alpha = 0.1,random_state = 42,)
# X_transformed = dict_learner.fit(iris).transform(iris)
# print("X_transformed",X_transformed.shape)#X_transformed (150, 15)


import sklearn.decomposition as sde
#
# "FastICA",
# "IncrementalPCA",
# "KernelPCA",
# "MiniBatchDictionaryLearning",
# "MiniBatchNMF",
# "MiniBatchSparsePCA",
# "NMF",
# "PCA",
# "SparseCoder",
# "SparsePCA",
# "dict_learning",
# "dict_learning_online",
# "fastica",
# "non_negative_factorization",
# "randomized_svd",
# "sparse_encode",
# "FactorAnalysis",
# "TruncatedSVD",
# "LatentDirichletAllocation",
dict_learner = sde.FastICA()
X_transformed = dict_learner.fit(iris).transform(iris)
print("FastICA X_transformed",X_transformed.shape)#(150, 4)
dict_learner = sde.IncrementalPCA()
X_transformed = dict_learner.fit(iris).transform(iris)
print("IncrementalPCA",X_transformed.shape)
dict_learner = sde.KernelPCA()
X_transformed = dict_learner.fit(iris).transform(iris)
print("KernelPCA",X_transformed.shape)
# FastICA X_transformed (150, 4)
# IncrementalPCA (150, 4)
# KernelPCA (150, 4)
#
