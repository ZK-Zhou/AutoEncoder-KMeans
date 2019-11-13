# coding:utf-8

import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from sklearn.cluster import KMeans
import os
import numpy as np
from PIL import Image
import matplotlib.image as mpimg

# def load_data():
img_height, img_width, channels = 160, 160, 3
dir = '/home/zhou/SMMD/convert/2k/n0'
files = ['%s/%s' % (dir,x) for x in os.listdir(dir)]
print(len(files))
arr = np.empty((len(files), img_height, img_width, channels), dtype=np.float32)
print(type(arr))
print(arr.shape)
# s = 0

# for imgfile in (files):
#     image = mpimg.imread(imgfile)
#     print(type(image))
#     num1 = imgfile.split('n0/')
#     num2 = num1[1].split('.jpg')
#     i = int(num2[0])
#     i = i - 2
#     mpimg.imsave('/home/zhou/SMMD/convert/2k1/' + str(i) + '.jpg', image)
# # arr = []
# arr = np.empty(( img_height, img_width, channels), dtype=np.float32)
# print(arr.shape)
# print(files)
# # print(enumerate(files))
# with tf.Session() as sess:
def load_image():
    s = 0
    for  imgfile in (files):
        # print(imgfile)
        num1 = imgfile.split('n0/')
        num2 = num1[1].split('.jpg')
        i = int(num2[0])
        image = tf.gfile.FastGFile(imgfile, 'rb').read()
        image = load_img(imgfile)
        # image = tf.image.decode_jpeg(image, channels=3)
        # image = tf.cast(image, tf.float32)
        image = img_to_array(image).reshape(160,160,3)
        arr[i] = image
        # base_size = 160
        # random_crop = 9
        # bs = base_size + 2 * random_crop
        # cropped = tf.image.resize_image_with_crop_or_pad(image, bs, bs)
        # if random_crop >0:
        #     cropped = tf.image.random_flip_left_right(cropped)
        #     cropped = tf.random_crop(cropped,[base_size, base_size, 3])
        # image = cropped
        # image = tf.cast(image, tf.float32)/255
        # image = tf.expand_dims(image, 0)
        # image = tf.image.resize_bilinear(image, (160,160))
        # # print(image.shape)
        # image = tf.squeeze(image, axis=0)
        #
        # image = sess.run(image)
        # # print(image.shape)
        # # print('image type is', type(image))
        # # print('image shape is', image.shape)
        # # print('arr[%d] shape is: ' % (i) ,arr[i].shape)
        # arr[i] = image
        s += 1
        print(s)
        # if i == 453:
        #     # print(arr[i,1,1])
        #     print(arr[i].shape)
        # arr.append(image)
        # print(np.array(arr).shape)
        # arr(i,160,160,3) = image
    print(arr[0].shape)
    print(arr[0])
    plt.imshow(arr[0]/255)
    plt.savefig('figs/restore.jpg')
    return arr
# arr[i] = image
    # return arr
# with tf.Session() as sess:
#     image_batch = []
#     image_batch = load_data()
#
#     print(sess.run(image_batch))
def get_image_batch(idx,arr):
    image_raw_batch = arr[64*idx:64*idx+64]
    image_processed_batch = np.empty((64,160,160,3), dtype=np.float32)
    print('image_raw_batch shape is:', image_raw_batch.shape)
    for i in range(64):
        print('second i is:', i)
        image_raw = image_raw_batch[i]
        print('image_raw shape is:', image_raw.shape)
        # base_size = 160
        # random_crop = 9
        # bs = base_size + 2 * random_crop
        # cropped = tf.image.resize_image_with_crop_or_pad(image_raw, bs, bs)
        # if random_crop > 0:
        #     cropped = tf.image.random_flip_left_right(cropped)
        #     cropped = tf.random_crop(cropped, [base_size, base_size, 3])
        # image = cropped
        # image = tf.cast(image, tf.float32)/255.
        # image = tf.expand_dims(image, 0)
        # image = tf.image.resize_bilinear(image, (160,160))
        # image = tf.squeeze(image, axis=0)
        # print('image processed shape is:',image_processed_batch[i].shape)
        # print('image shape is:',image.shape)
        # print('image value is:\n',image)
        # image = image_raw.crop(9,29,169,189)
    #     image_processed_batch[i] = image
    #     min_after_dequeue = 64# tf.tarin.batch no this para
    # image_queue = tf.train.slice_input_producer(image_processed_batch, shuffle=False)
    # batch_size = 64
    # capacity = batch_size
    # image_batch = tf.train.batch(image_queue,
    #                             batch_size=batch_size,
    #                             capacity=capacity,
    #                             )
    # return image_batch
def main():
    arr = load_image()

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(30):
            print(i)
            # image_real_batch = sess.run(get_image_batch(i,arr))
            image_raw_batch = arr[64*i:64*i + 64]
            image_processed_batch = np.empty((64, 160, 160, 3), dtype=np.float32)
            # print('image_raw_batch shape is:', image_raw_batch.shape)
            for j in range(64):
            #     print('second i is:', j)
                image_raw = image_raw_batch[j]
            #     print('image_raw shape is:', image_raw.shape)
            #     # base_size = 160
                # random_crop = 9
                # bs = base_size + 2 * random_crop
                # cropped = tf.image.resize_image_with_crop_or_pad(image_raw, bs, bs)
                # if random_crop > 0:
                #     cropped = tf.image.random_flip_left_right(cropped)
                #     cropped = tf.random_crop(cropped, [base_size, base_size, 3])
                # image = cropped
                # image = tf.cast(image, tf.float32) / 255.
                # image = tf.expand_dims(image, 0)
                # image = tf.image.resize_bilinear(image, (160, 160))
                # image = tf.squeeze(image, axis=0)
                # # print('image processed shape is:', image_processed_batch[i].shape)
                # # print('image shape is:', image.shape)
                # # print('image value is:\n', image)
                # image = sess.run(image)
                # print('image shape is:', image.shape)
                # print('image value is:\n', image)
                image = image_raw
                # image = image_raw.crop(9, 29, 169, 189)
                image_processed_batch[j] = image
                min_after_dequeue = 64  # tf.tarin.batch no this para
            image_queue = tf.train.slice_input_producer([image_processed_batch], shuffle=False)
            batch_size = 64
            capacity = batch_size
            image_batch = tf.train.batch([image_queue],
                                         batch_size=batch_size,
                                         capacity=capacity,
                                         )
            image_batch = sess.run(image_batch)
            print('image batch shape is:', image_batch.shape)
            print('image batch type is:', type(image_batch))
            # print('image processed batch shape is:', image_processed_batch.shape)
            # print('image processed batch type is:', type(image_processed_batch))
            # print(image_processed_batch[0])
            # plt.imshow(image_processed_batch[0]/255)
            # plt.savefig('figs/test_' + str(i) + '.jpg')
            plt.imshow(image_batch[0]/255)
            plt.savefig('figs/test_' + str(i) + '.jpg')
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
