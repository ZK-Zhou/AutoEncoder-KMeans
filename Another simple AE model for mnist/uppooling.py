import tensorflow as tf
def max_pool_with_argmax(net,stride):
# '''
# 重定义一个最大池化函数，返回最大池化结果以及每个最大值的位置(是个索引，形状和池化结果一致)
# args:
# net:输入数据 形状为[batch,in_height,in_width,in_channels]
# stride：步长，是一个int32类型，注意在最大池化操作中我们设置窗口大小和步长大小是一样的
# '''
# #使用mask保存每个最大值的位置 这个函数只支持GPU操作
    _, mask = tf.nn.max_pool_with_argmax( net,ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],padding='SAME')
    #将反向传播的mask梯度计算停止
    mask = tf.stop_gradient(mask)
    #计算最大池化操作
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME')
    #将池化结果和mask返回
    return net,mask
 
 
 
def un_max_pool(net,mask,stride):
# '''
# 定义一个反最大池化的函数，找到mask最大的索引，将max的值填到指定位置
# args:
# net:最大池化后的输出，形状为[batch, height, width, in_channels]
# mask：位置索引组数组，形状和net一样
# stride:步长，是一个int32类型，这里就是max_pool_with_argmax传入的stride参数
# '''
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    # calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

#定义一个形状为4x4x2的张量
img = tf.constant([
[[0.0,4.0],[0.0,4.0],[0.0,4.0],[0.0,4.0]],
[[1.0,5.0],[1.0,5.0],[1.0,5.0],[1.0,5.0]],
[[2.0,6.0],[2.0,6.0],[2.0,6.0],[2.0,6.0]],
[[3.0,7.0],[3.0,7.0],[3.0,7.0],[3.0,7.0]],
])
img = tf.reshape(img,[1,4,4,2])
#最大池化操作
pooling1 = tf.nn.max_pool(img,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#带有最大值位置的最大池化操作
pooling2,mask = max_pool_with_argmax(img,2)
#反最大池化
img2 = un_max_pool(pooling2,mask,2)
with tf.Session() as sess:
    print('image:')
    image = sess.run(img)
    print(image)
    print('image shape is', image.shape)
    #默认的最大池化输出
    result = sess.run(pooling1)
    print('max_pool:\n',result)
    print('max_pool shape is', result.shape)
    #带有最大值位置的最大池化输出
    result,mask2 = sess.run([pooling2,mask])
    print('max_pool_with_argmax:\n',result,'\n mask2 value: \n',mask2)
    print('result shape is',result.shape)
    print('mask2 shape is',mask2.shape)
    #反最大池化输出
    result = sess.run(img2)
    print('un_max_pool\n',result)
    print('un_max_pool shape is',result.shape)
