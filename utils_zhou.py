"""
This file is to store some function
Author: Zhaokun Zhou
Data: 2019.11.13
"""
##################################################################################
"""
Function: process the celebA tfrecords
"""
# def image_batch():
    #read the tf records
#     files = tf.train.match_filenames_once('/home/zhou/cnn_autoencoder_mnist-master/celebA_records_2k/celebA/tf_records_train/train*')
    #initiate the files thread
#     filename_queue = tf.train.string_input_producer(files, shuffle=False)
    #creat the reader
#     reader = tf.TFRecordReader()
#     _,serialized_example = reader.read(filename_queue)
#
#    # 解析读取的样例。
#     features = tf.parse_single_example(
#         serialized_example,
#         feature = {
#             'image/height': tf.FixedLenFeature([], tf.int64),
#             'image/width': tf.FixedLenFeature([], tf.int64),
#             'image/colorspace': tf.FixedLenFeature([], tf.string),
#             'image/channels': tf.FixedLenFeature([], tf.int64),
#             'image/format': tf.FixedLenFeature([], tf.string),
#             'image/encoded': tf.FixedLenFeature([], tf.string)}
#     #     features={
#     #         'image_raw':tf.FixedLenFeature([],tf.string),
#     #         'pixels':tf.FixedLenFeature([],tf.int64),
#     #         'label':tf.FixedLenFeature([],tf.int64)}
#         )

###     ####preprocess the single image, output dimension is [160,160,3]

#     decoded_images = tf.decode_raw(features['image/encoded'],tf.uint8)
#     retyped_images = tf.cast(decoded_images, tf.float32)
#     # labels = tf.cast(features['label'],tf.int32)
#     #pixels = tf.cast(features['pixels'],tf.int32)
#     images = tf.reshape(retyped_images, [218,178,3])

#       #creat the data batch queue
#     min_after_dequeue = 10000
#     batch_size = 64
#     capacity = min_after_dequeue + 3 * batch_size
#     # image_batch, label_batch = tf.train.shuffle_batch([images, labels],
#     #                                                     batch_size=batch_size,
#     #                                                     capacity=capacity,
#     #                                                     min_after_dequeue=min_after_dequeue)

#     return image_batch
##################################################################################

"""
Function : build the network model and get every layers' output dimensions 
"""
def build_network(self):
    self.embedding = self.encoder(self.image_batch, is_training=True, reuse=False)
    self.embedding_test = self.encoder(self.source, is_training=False, reuse=True)
    # self.embedding, net1, net2, net3, net4, net5 = self.encoder(self.image_batch, is_training=True, reuse=False)
    self.pred = self.decoder(self.embedding, is_training=True, reuse=False)
    # self.pred, net1, net2, net3, net4, net5 = self.decoder(self.embedding, is_training=True, reuse=False)
    # print('net1 shape is', net1.shape)
    # print('net2 shape is', net2.shape)
    # print('net3 shape is', net3.shape)
    # print('net4 shape is', net4.shape)
    # print('net5 shape is', net5.shape)
    # self.pred_test, net1, net2, net3, net4, net5 = self.decoder(self.embedding, is_training=False, reuse=True)
    self.pred_test = self.decoder(self.embedding, is_training=False, reuse=True)
    self.pred_use = self.decoder(self.embedding_test, is_training=False, reuse=True)

    self.mean_pred, self.std_pred = tf.nn.moments(self.pred, axes=[0, 1, 2, 3])
    # self.mean_pred, self.std_pred = tf.nn.moments(self.pred, axes=0)
    self.std_pred = tf.sqrt(self.std_pred)

##################################################################################
