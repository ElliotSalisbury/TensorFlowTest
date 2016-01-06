import tensorflow as tf
import pickle
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#download dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with open("objs.pickle", "rb") as f:
  importedData = pickle.load(f)
def getTrainingData():
  Xs = importedData["trainingX"]
  Ys = importedData["trainingY"]
  return Xs, Ys

def getTrainingBatch(size):
  xs, ys = getTrainingData()
  indexs = range(0,len(xs))
  batch = random.sample(indexs,size)

  batch_xs = [xs[i] for i in batch]
  batch_ys = [ys[i] for i in batch]
  return batch_xs, batch_ys

def getTestData():
  Xs = importedData["testX"]
  Ys = importedData["testY"]
  return Xs, Ys


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([784, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = getTrainingBatch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
test_xs, test_ys = getTestData()
print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))