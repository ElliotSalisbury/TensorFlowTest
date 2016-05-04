################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import pickle
import random as pyrand


import tensorflow as tf

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read Image
def prepareImg(img):
    x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
    i = x_dummy.copy()
    i[0,:,:,:] = np.array(img).astype(float32)
    i = i-mean(i)
    return i
i = prepareImg(imread("poodle.png")[:,:,:3])

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))


net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())



# x_ = tf.Variable(i)

NUM_IMAGES = 1
x = tf.placeholder("float", shape=[NUM_IMAGES, 227,227,3])
# y_ = tf.placeholder("float", shape=[None, 2])

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [NUM_IMAGES, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
# fc8W = tf.Variable(net_data["fc8"][0])
# fc8b = tf.Variable(net_data["fc8"][1])
# fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
#
#
# #prob
# #softmax(name='prob'))
# prob = tf.nn.softmax(fc8)

# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
#
# output = sess.run(prob, feed_dict={x: i})
# # outfc8 = sess.run(fc8)
################################################################################




# def getTrainingBatch(size):
#     xs, ys = getTrainingData()
#     indexs = range(0,len(xs))
#     batch = pyrand.sample(indexs,size)
#
#     batch_xs = [xs[i] for i in batch]
#     batch_ys = [ys[i] for i in batch]
#     return batch_xs, batch_ys
#
# def getTestData():
#     Xs = importedData["testX"]
#     Ys = importedData["testY"]
#     return Xs, Ys
#
# #train new network on the n-1 layer (fc8)
# y_ = tf.placeholder("float", shape=[1, 2])
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
# my_fc8W = weight_variable([4096, 2])
# my_fc8b = bias_variable([2])
# my_fc8 = tf.nn.xw_plus_b(fc7, my_fc8W, my_fc8b)
# my_prob = tf.nn.softmax(my_fc8)

# #train model
# cross_entropy = -tf.reduce_sum(y_*tf.log(my_prob))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(my_prob,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# saver = tf.train.Saver()

# saver.restore(sess, "saves/model_013100.ckpt")

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# def getTrainingData(importedData):
#     Xs = np.array(importedData.tolist())[0][np.newaxis, ...]
#     Ys = np.array(np.array(importedData.tolist())[1])[np.newaxis, ...]
#     return Xs, Ys
#
# for samplePickle in os.listdir("./training/negative"):
#     with open("training/negative/%s"%samplePickle, "rb") as f:
#         importedData = np.array(pickle.load(f))
#
#
#     Xs,Ys = getTrainingData(importedData)
#     numBatches = int(math.ceil(Xs.shape[0] / float(NUM_IMAGES)))
#     bottlenecks = []
#     for b in range(numBatches):
#         xs = Xs[b * NUM_IMAGES:(b + 1) * NUM_IMAGES]
#         newxs = np.zeros([NUM_IMAGES, xs.shape[1], xs.shape[2], xs.shape[3]])
#         newxs[0:xs.shape[0]] = xs
#         bottleneck = sess.run(fc7, feed_dict={x:newxs})
#
#         for i in range(xs.shape[0]):
#             bottlenecks.append([bottleneck[i],Ys[i]])
#
#     with open('bottlenecks/negative/%s'%samplePickle, 'wb') as f:
#         pickle.dump(bottlenecks, f)

def getBottlenecks(imgs):
    bottlenecks = []
    for img in imgs:
        i = np.zeros((1,) + xdim).astype(float32)
        i[0, :, :, :] = np.array(img).astype(float32)
        i = i - mean(i)

        bottlenecks.append(sess.run(fc7, feed_dict={x: i}))
    return bottlenecks
