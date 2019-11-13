"""
File Function:
1.crop the celebA image from(178,218,3) to (160,160,3)
2.KMeans cluster: divide all image data to different file folders
"""



import os
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

####################################
################function: crop image

# dir = '/home/zhou/SMMD/convert/2k/n0'
# files = ['%s/%s' % (dir,x) for x in os.listdir(dir)]
#
# for imgfile in files:
#     n1 = imgfile.split('n0/')
#     n2 = n1[1].split('.jpg')
#     i = int(n2[0])
#     im = Image.open(imgfile)
#     imgsize = im.size
#     print('The size of image is:{}'.format(imgsize))
#     x = 9
#     y = 29
#     w = 160
#     h = 160
#     image_new = im.crop((9,29,169,189))
#     image_new.save('/home/zhou/SMMD/convert/2k1/' + str(i) + '.jpg')
#####################################


def KMeans_divide():
    embedding_all = np.loadtxt('figs/embedding_1984.txt', delimiter=',')
    print('embedding all shape is:', embedding_all.shape)
    print('embedding all type is:', type(embedding_all))
    print('embedding 1 shape is',embedding_all[0].shape)
    print('embedding 1 value is',embedding_all[0])
    n_clusters = 2
    km = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10)
    km.fit(embedding_all)

    for i in range(n_clusters):
        print('label %d num is:' % i ,(km.labels_==i).sum())
        # print(embedding_all[km.labels_==i])
        idxs = np.where(km.labels_==i)[0]
        # idx = np.where(km.labels_==i)

        print(idxs)
        imgs_dir = '/home/zhou/SMMD/convert/2k_original/n'+str(i+1)+'/'
        for idx in idxs:
            shutil.copy('/home/zhou/SMMD/convert/2k_original/n0/'+str(idx)+'.jpg',imgs_dir+str(idx)+'.jpg')
        print('The number {} cluster has been copied successfully!'.format(i))
        # print(idx)
        # print(idx.shape)
        # for idx in range((km.labels_==i).sum()):
        #     print(idx)
        # res = dataframe[(km.labels_==1)]
        # for idx in embedding_all:
        #     if embedding_all[idx].labels_ == i:
        #         print(idx)
        # # print(res[0])