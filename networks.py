# coding:utf-8
"""
Base class of CNN
"""
import tensorflow as tf 
from utils import *

class MLP(BasicBlock):
    def __init__(self, hidden_units, output_dim, sn=False, name=None):
        super(MLP, self).__init__(None, name or "MLP")
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.sn = sn
    
    def __call__(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            x = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            net = tf.nn.softplus(dense(x, self.hidden_units[0], sn=self.sn, name="fc0"))
            for i in range(1, len(self.hidden_units)):
                net = dense(net, self.hidden_units[i], sn=self.sn, name="fc{}".format(i))
                if not self.sn:
                    net = bn(net, is_training, name="bn{}".format(i))
                net = tf.nn.softplus(net) 
            net = dense(net, self.output_dim, sn=self.sn, name="fc{}".format(len(self.hidden_units)))
        return net 

class CNN_Encoder(BasicBlock):
    def __init__(self, output_dim, sn=False, name=None):
        super(CNN_Encoder, self).__init__(None, name or "CNN_Encoder")
        self.output_dim = output_dim
        self.sn = sn
    
    def __call__(self, x, sn=False, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            net1 = lrelu(conv2d(x, 3, 128, 4, 4, 2, 2, sn=self.sn, padding="SAME", name="conv1"), name="l1")
            net2 = lrelu(bn(conv2d(net1, 0, 256, 4, 4, 2, 2, sn=self.sn, padding="SAME", name="conv2"), is_training, name="bn2"), name="l2")
            net3 = lrelu(bn(conv2d(net2, 0, 512, 4, 4, 2, 2, sn=self.sn, padding="SAME", name="conv3"), is_training, name="bn3"), name="l3")
            net4 = tf.reshape(net3, [-1, 20*20*512])
            net5 = lrelu(bn(dense(net4, 1024, sn=self.sn, name="fc4"), is_training, name="bn4"), name="l4")
            out = dense(net5, self.output_dim, sn=self.sn, name="fc5")
        return out

class CNN_Decoder(BasicBlock):
    def __init__(self, sn=False, name=None):
        super(CNN_Decoder, self).__init__(None, name or "CNN_Decoder")
        self.sn = sn
    
    def __call__(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            net1 = tf.nn.relu(dense(x, 1024, name='fc1'))
            net2 = tf.nn.relu(bn(dense(net1, 512*20*20, name='fc2'), is_training, name='bn2'))
            net3 = tf.reshape(net2, [-1, 20, 20, 512])
            net4 = tf.nn.relu(bn(deconv2d(net3, 256, 4, 4, 2, 2, padding="SAME", name='dc3'), is_training, name='bn3'))
            net5 = tf.nn.relu(bn(deconv2d(net4, 128, 4, 4, 2, 2, padding="SAME", name='dc4'), is_training, name='bn4'))
            out = tf.nn.sigmoid(deconv2d(net5, 3, 4, 4, 2, 2, padding="SAME", name="dc5"))
        return out



# x = tf.random_normal(shape=(64,28,28,1), dtype=tf.float32)
# E = CNN_Encoder(128, sn=False)
# D = CNN_Decoder(sn=False)
# z = E(x)
# y = D(z)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(z).shape 
#     print sess.run(y).shape