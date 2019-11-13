import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

files = tf.train.match_filenames_once(
                '/home/zhou/cnn_autoencoder_mnist-master/celebA_TFRecords_2k/celebA/tf_records_train/train*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/colorspace': tf.FixedLenFeature([], tf.string),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string)}
    #     features={
    #         'image_raw':tf.FixedLenFeature([],tf.string),
    #         'pixels':tf.FixedLenFeature([],tf.int64),
    #         'label':tf.FixedLenFeature([],tf.int64)}
)
#
# decoded_images = tf.decode_raw(features['image/encoded'], tf.uint8)
height = features['image/height']
width = features['image/width']
print('height is',height,'\n','width is',width)
decoded_images = tf.image.decode_jpeg(features['image/encoded'], channels=3)
print('decoded_image shape is:',decoded_images.shape)
retyped_images = tf.cast(decoded_images, tf.float32)
print('retyped_image shape is:',retyped_images.shape)
base_size = 160
random_crop = 9
bs = base_size + 2 * random_crop
cropped = tf.image.resize_image_with_crop_or_pad(retyped_images, bs, bs)
if random_crop > 0:
    cropped = tf.image.random_flip_left_right(cropped)
    cropped = tf.random_crop(cropped, [base_size, base_size, 3])
image = cropped
image = tf.cast(image, tf.float32)/255.
image = tf.expand_dims(image, 0)
image = tf.image.resize_bilinear(image, (160,160))
image = tf.squeeze(image, axis=0)

# labels = tf.cast(features['label'],tf.int32)
# pixels = tf.cast(features['pixels'],tf.int32)
# images = tf.reshape(retyped_images, [218, 178, 3])

min_after_dequeue = 1
batch_size = 64
capacity = min_after_dequeue + 3 * batch_size
# image_batch, label_batch = tf.train.shuffle_batch([images, labels],
#                                                     batch_size=batch_size,
#                                                     capacity=capacity,
# #                                                     min_after_dequeue=min_after_dequeue)
# image_batch = tf.train.shuffle_batch([image],
#                                      batch_size=batch_size,
#                                      capacity=capacity,
#                                      min_after_dequeue=2)

image_batch = tf.train.batch([image],
                                     batch_size=batch_size,
                                     capacity=capacity
                                     )
with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    print(sess.run(files))
    print('Begin to queue!')
    # print(sess.run(height))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(2):
        print('Begin to %d queue!' % i)

        cur_batch = sess.run(image_batch)
        print(cur_batch.shape)
        print(cur_batch[0])
        plt.imshow(cur_batch[0])
        plt.savefig('batch_test_' + str(i) + '.jpg')
        # cur_batch = cur_batch.reshape(2,160,160,3)
        # plt.savefig(cur_batch[0])
    coord.request_stop()
    coord.join(threads)