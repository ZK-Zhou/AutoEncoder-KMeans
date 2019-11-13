#!/usr/bin/env python
# coding: utf-8




import cv2
import os
import pickle
from numpy import *
from scipy import *
from pylab import *
from PIL import Image
from scipy.cluster.vq import *


# In[149]:


def pca(X):
    num_data,dim = X.shape
    print(dim)
    print(num_data)
    mean_X = X.mean(axis=0)
    X = X- mean_X
    
    if dim>num_data:
        M = dot(X.T,X)
        print('M shape is:',M.shape)
        print('M value is:',M)
        e,EV = linalg.eigh(M)
        print('e shape is:',e.shape)
        print('e value is:',e)
        print('EV shape is:',EV.shape)
        print('EV value is:',EV)
        print('X shape is:',X.shape)
#         tmp = dot(X.T,EV).T
#         print('tmp value is:',tmp)
#         print('tmp shape is:',tmp.shape)
#         V = tmp[::-1]
        index = argsort(-e)
        V = matrix(EV.T[index])
        print('V shape is:',V.shape)
        print('V value is:',V)
        S = sqrt(e)[::-1]
#         for i in range(V.shape[1]):
#             V[:,i] /= S
    else:
        U,S,V = linalg.svd(X)
        V = V[:num_data]
    return V,S,mean_X
	
# def pca(X):
#     num_data,dim = X.shape
#     # 数据中心化
#     mean_X = X.mean(axis=0)
#     X = X - mean_X
#     if dim>num_data:
#     # PCA- 使用紧致技巧
#         M = dot(X.T,X) # 协方差矩阵
#         e,EV = linalg.eigh(M) # 特征值和特征向量
# #         tmp = dot(X.T,EV).T # 这就是紧致技巧
# #         V = tmp[::-1] # 由于最后的特征向量是我们所需要的，所以需要将其逆转
#         index = argsort(-e)
#         V = matrix(EV.T[index])
#         S = sqrt(e)[::-1] # 由于特征值是按照递增顺序排列的，所以需要将其逆转
# #         for i in range(V.shape[1]):
# #             V[:,i] /= S
#     else:
#     # PCA- 使用 SVD 方法
#         U,S,V = linalg.svd(X)
#         V = V[:num_data] # 仅仅返回前 nun_data 维的数据才合理
#     # 返回投影矩阵、方差和均值
#     return V,S,mean_X



def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

imlist = get_imlist('C:\\Users\\zk_zhou\\Desktop\\a_selected_thumbs\\')
# imlist = get_imlist('C:\\Users\\zk_zhou\\Desktop\\celebA\\')

im = array(Image.open(imlist[0]))
m,n = im.shape[:2]
# d,m,n = im.shape[:3]
# print('d = ',d)
print('m = ',m)
print('n = ',n)

imnbr = len(imlist)
print("The number of images is %d" %imnbr)

immatrix = array([array(Image.open(imname)).flatten() for imname in imlist], 'f')

V, S, immean = pca(immatrix)

f = open('C:\\Users\\zk_zhou\\Desktop\\a_pca_modes.pkl', 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()

with open('C:\\Users\\zk_zhou\\Desktop\\a_pca_modes.pkl', 'rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)
    print(array(V).shape)

immean = immean.flatten()
print('V[:20]).shape is :',V[:20].shape)
print('immatrix[23].shape is',immatrix[23].shape)
projected = array([dot((V[:20]),immatrix[i]-immean) for i in range(imnbr)])
print('projected.shape is',projected.shape)
print('projected.value is',projected)

projected = squeeze(projected)
print('projected.shape is',projected.shape)
projected = whiten(projected)
centroids,distortion = kmeans(projected,2)
code,distance = vq(projected,centroids)
print(centroids)
print(distortion)
for k in range(2):
    ind = where(code==k)[0]
    print(len(ind))
#     print(ind[3])
    figure()
    gray()
    for i in range(len(ind)):
        imagename = str(i) + '.jpg'
        if i == 1:
            print(immatrix[ind[i]])
            print(immatrix[ind[i]].shape)
#         image = Image.fromarray(immatrix[ind[i]].astype(np.uint8).reshape((218,178,3)))
        image = Image.fromarray(immatrix[ind[i]].astype(np.uint8).reshape((25,25)))
        if image.mode == "F":
            image = image.convert('RGB')
        image.save('C:\\Users\\zk_zhou\\Desktop\\Oct\\' + str(k)+ '\\' + imagename)
    for i in range(minimum(len(ind),40)):
        subplot(4,10,i+1)
#         imshow((immatrix[ind[i]].astype(np.uint8)).reshape((218,178,3)))
        imshow((immatrix[ind[i]].astype(np.uint8)).reshape((25,25)))
        axis('off')
                  
show()                





