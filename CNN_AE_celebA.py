# coding:utf-8
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from utils import *
from networks import *
import numpy as np
from keras.preprocessing.image import load_img, img_to_array



class AutoEncoder(BasicTrainFramework):
    def __init__(self, batch_size, version="AE"):
        super(AutoEncoder, self).__init__(batch_size, version)

        # self.data = datamanager_mnist(train_ratio=0.8, fold_k=None, expand_dim=True, norm=True)
        # self.sample_data = self.data(self.batch_size, phase='test', var_list=["data", "labels"])

        self.emb_dim = 256 #Encoder output dimensions [1,256]
        self.batch_size = 64
        self.encoder = CNN_Encoder(output_dim=self.emb_dim, sn=False) #initiate CNN Encoder
        self.decoder = CNN_Decoder(sn=False) # initiate CNN Decoder
        self.image_batch = self.image_batch() # Get the batch queue
        self.build_placeholder() # initiate the placeholder to input data to a pretrained model sequentially
        self.build_network() # build the network layer
        self.build_optimizer()

        self.build_sess()
        self.build_dirs()

    def image_batch(self):

        # read the tf records

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
            )

        #preprocess the single image, output dimension is [160,160,3]

        # decoded_images = tf.decode_raw(features['image/encoded'], tf.uint8)
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

        min_after_dequeue = 64
        batch_size = 64
        capacity = min_after_dequeue + 3 * batch_size

        image_batch = tf.train.shuffle_batch([image],
                                             batch_size=batch_size,
                                             capacity=capacity,
                                             min_after_dequeue=64)
        return image_batch

    def build_placeholder(self):

        self.source = tf.placeholder(shape=(self.batch_size, 160, 160, 3), dtype=tf.float32)
        self.embedding_use = tf.placeholder(shape=(self.batch_size,256),dtype=tf.float32)

    def build_network(self):
        self.embedding = self.encoder(self.image_batch, is_training=True, reuse=False)
        self.embedding_test = self.encoder(self.source, is_training=False, reuse=True)
        self.pred = self.decoder(self.embedding, is_training=True, reuse=False)
        self.pred_test = self.decoder(self.embedding, is_training=False, reuse=True)
        self.pred_use = self.decoder(self.embedding_test, is_training=False, reuse=True)
        self.mean_pred, self.std_pred = tf.nn.moments(self.pred, axes=[0, 1, 2, 3])
        self.std_pred = tf.sqrt(self.std_pred)

    def build_optimizer(self):
        print('image_batch shape is', self.image_batch.shape)
        print('embedding shape is', self.embedding.shape)
        # print('pred shape is', self.pred.shape)
        print('pred shape is', self.pred.shape)
        # print('target shape is', self.target.shape)
        self.loss = mse(self.pred, self.image_batch, self.batch_size)
        self.solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.loss,
                                                                                     var_list=self.encoder.vars + self.decoder.vars)
    """
    function: plot the histogram
    """
    def hist(self, epoch):
        real = self.sess.run(self.image_batch)

        pr, _ = np.histogram(real, bins=np.linspace(0., 1, 100), density=True)
        plt.plot(np.linspace(0., 1, len(pr)), pr, label='real', color='g', linewidth=2)
        emb, fake = [], []
        for _ in range(1):
            e, f = self.sess.run([self.embedding, self.pred_test])
            emb.append(e)
            fake.append(f)
        pe, _ = np.histogram(np.concatenate(emb), bins=np.linspace(0., 1, 100), density=True)
        plt.plot(np.linspace(0., 1, len(pe)), pe, label='embedding', color='b', linewidth=2)
        pf, _ = np.histogram(np.concatenate(fake), bins=np.linspace(0., 1, 100), density=True)
        plt.plot(np.linspace(0., 1, len(pf)), pf, label='fake', color='r', linewidth=2)

        plt.legend()
        plt.title("epoch_{}".format(epoch))
        plt.savefig(os.path.join(self.fig_dir, "hist_epoch_{}.png".format(epoch)))
        plt.clf()

    """
    Function: sample form the model and compare to the original images
    """

    def sample(self, epoch):
        real = self.sess.run(self.image_batch)
        print('real shape is',real.shape)
        print('real 2 is :\n', real[2])
        print('real 2 shape is:',real[2].shape)
        fake = self.sess.run(self.pred_test)
        print('fake shape is',fake.shape)
        print('fake 2 is :\n', fake[2])
        print('fake 2 shape is:',fake[2].shape)
        for i in range(4):
            for j in range(2):
                idx = i * 2 + j
                plt.subplot(4, 4, idx * 2 + 1)
                plt.imshow((real[idx]*255).astype(np.uint8))
                plt.subplot(4, 4, idx * 2 + 2)
                plt.imshow((fake[idx]*255).astype(np.uint8))
        plt.savefig(os.path.join(self.fig_dir, "sample_epoch_{}.png".format(epoch)))
        plt.clf()

    """
    Function: Train the model
    """
    def train(self, epoches=1):
        batches_per_epoch = int(2000 / 64)
        print('begin to train 1')
        for epoch in range(epoches):
            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx

                data = self.image_batch
                self.sess.run(self.image_batch)
                self.sess.run(self.solver)
                # print('iteration is begin!')
                # print('iteration:',cnt)
                if cnt % 10 == 0:
                    loss = self.sess.run(self.loss)
                    print("Epoch [%3d/%3d] iter [%3d/%3d] loss=%.3f" % (epoch, epoches, idx, batches_per_epoch, loss))
            mean, std = self.sess.run([self.mean_pred, self.std_pred])
            print("mean=%.3f std=%.3f" % (mean, std))
            if epoch % 5 == 0:
                self.hist(epoch)
                self.sample(epoch)
        self.hist(epoch)
        self.sample(epoch)
        #save the checkpoint
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)



"""
cache the pre-cropped images and prepare to pass the model
"""
img_height, img_width, channels = 160, 160, 3
dir = '/home/zhou/SMMD/convert/2k/n0'
files = ['%s/%s' % (dir, x) for x in os.listdir(dir)]
print(len(files))
arr = np.empty((len(files), img_height, img_width, channels), dtype=np.float32)
print(type(arr))
print(arr.shape)

"""
Load the arr(array of all images in dimension [(len(number of images),width,height,channel)])
notice that the framework is Keras
"""
def load_image():
    s = 0
    for  imgfile in (files):
        num1 = imgfile.split('n0/')
        num2 = num1[1].split('.jpg')
        i = int(num2[0])
        image = tf.gfile.FastGFile(imgfile, 'rb').read()
        image = load_img(imgfile)

        image = img_to_array(image).reshape(160,160,3)
        image = image.astype('float32') / 255.
        arr[i] = image
        s += 1
        print(s)
    print(arr[0].shape)
    print(arr[0])
    plt.imshow(arr[0])
    plt.savefig('figs/restore.jpg')
    return arr

"""
The encapsulation of AutoEncoder"""
def AE():
    # train AE
    ae = AutoEncoder(64)

    #train the model
    ae.train(epoches=200) #train model

    ae.load_model()

##############################################################################
#####function: Test after load the model
    # ae.sess.run(ae.encoder(ae.image_batch()))
    # fake, real = ae.sess.run([ae.pred_test,ae.image_batch])
    # print('real type is:',type(real))
    # print('real value is:', real[2])
    # print('fake shape is', fake.shape)
    # print('fake 2 is :\n', fake[2])
    # print('fake 2 shape is:', fake[2].shape)
    # for i in range(4):
    #     for j in range(2):
    #         idx = i * 2 + j
    #         plt.subplot(4, 4, idx * 2 + 1)
    #         plt.imshow((real[idx] * 255).astype(np.uint8))
    #         plt.subplot(4, 4, idx * 2 + 2)
    #         plt.imshow((fake[idx] * 255).astype(np.uint8))
    # plt.savefig(os.path.join(ae.fig_dir, "test_{}.png".format(1)))
    # plt.clf()
##############################################################################
    """
    Function: all images pass the Encoder and save the embedding data to txt
    """
    arr = load_image()
    embedding_all = np.empty((1984,256), dtype=np.float32)
    for i in range(31):
        print(i)
        image_raw_batch = arr[64*i:64*i+64]
        image_processed_batch = np.empty((64, 160, 160, 3), dtype=np.float32)
        for j in range(64):
            image_raw = image_raw_batch[j]
            image = image_raw
            image_processed_batch[j] = image

        print('image processed batch shape is:', image_processed_batch.shape)
        print('image processed batch type is:', type(image_processed_batch))
        # source = tf.placeholder(shape=(len(image_processed_batch), 160, 160, 3), dtype=tf.float32)
        # ae.image_batch = {source: image_processed_batch}
        # out_embed = ae.encoder(feed, is_training=False, reuse=True)
        # embedding = ae.sess.run(ae.embedding)
        embedding = ae.sess.run(ae.embedding_test, feed_dict={ae.source:image_processed_batch})
        print('ae embedding shape is:', embedding.shape)
        print('ae embedding type is:', type(embedding))
        # pred = ae.sess.run(ae.pred_use, feed_dict={ae.source:image_processed_batch})
        embedding_all[64*i:64*i+64] = embedding
    print('embedding all shape is:',embedding_all.shape)
    print('embedding all type is:', type(embedding_all))
    np.savetxt('figs/embedding_1984.txt', embedding_all, fmt='%f', delimiter=',')
        # pred = ae.sess.run(ae.pred_use, feed_dict={ae.embedding_use:embedding})
        # pred, embedding = ae.sess.run([ae.pred_use, ae.embedding_test], feed_dict={ae.source:image_processed_batch})
        # if i == 0 or i == 2 or i == 3:
            # print('ae embedding shape is:', embedding.shape)
            # print('ae embedding type is:', type(embedding))
            # print('ae pred test shape is:', pred.shape)
            # print('ae pred type is:', type(pred))# check success to reconstruct the picture
            # # print(image_processed_batch[0])
            # plt.imshow(ae.image_batch[0]/255)
            # print('The embedding value is" \n',embedding[0])
            # print('pred 0 value is \n', pred[0])
            # plt.imshow((pred[0]*255).astype(np.uint8))
            #
            # plt.savefig('figs/test_' + str(i) + '.jpg')

##############################################################################
    # coords and threads request to stop
    ae.coord.request_stop()
    ae.coord.join(ae.threads)

if __name__ == "__main__":
    AE()
    # AE_SUP()