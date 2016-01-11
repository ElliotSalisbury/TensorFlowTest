import tensorflow as tf
import pickle
import random
import numpy as np
import cv2

#download dataset
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with open("objs.pickle", "rb") as f:
  importedData = pickle.load(f)

def getTrainingData():
  Xs = importedData["trainingX"]
  Ys = importedData["trainingY"]
  return Xs, Ys

xs, ys = getTrainingData()
indexs = range(0,len(xs))
def getTrainingBatch(size):
  batch = random.sample(indexs,size)

  batch_xs = [xs[i] for i in batch]
  batch_ys = [ys[i] for i in batch]
  return batch_xs, batch_ys

def getTestData():
  Xs = importedData["testX"]
  Ys = importedData["testY"]
  return Xs, Ys

#define helper functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

### Multilayer Convolutional Network

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 2])

# reshape x back into a 2d image
x_image = tf.reshape(x, [-1,28,28,1])

#first conv layer is a 5x5 kernel, with an input of 1, and output of 32 features.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#apply the convolution to the image, and then apply ReLU and max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#second conv layer is 5x5 kernel, wth an input of 32 features, output of 64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#third layer takes fully connected image (7x7 now) and outputs 1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#avoid overfitting
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#add softmax layer to convert the 1024 into just 10 outputs
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#train model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

def trainSession():
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  for i in range(10000):
    batch = getTrainingBatch(50)
    if i%100 == 0:
      print("step %d, training accuracy %g"%(i,sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  test_xs, test_ys = getTestData()
  print("test accuracy %g"%sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0}))

  return sess

def img2data(img):
  data = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  data = data.astype(np.float32) / 255.0
  data = data.flatten()

  return data

sess = trainSession()
windowSize = 28
videoPath = "vanuatu35.mp4"
capture = cv2.VideoCapture(videoPath)

ret, frame = capture.read()
frameId = 0
while ret:
  frame = cv2.resize(frame, (640,360))
  height,width,depth = frame.shape

  downscale = 10
  maskHeight = int((height-windowSize)/downscale)
  maskWidth = int((width-windowSize)/downscale)

  data = []
  for i in range(0,maskWidth):
      for j in range(0,maskHeight):
        xPos = i*downscale
        yPos = j*downscale

        window = frame[yPos:yPos+windowSize,xPos:xPos+windowSize]
        data.append(img2data(window))

  print("About to run classifier...")
  results = sess.run(y_conv, feed_dict={x: data, keep_prob: 1.0})

  mask = np.zeros((maskHeight,maskWidth),dtype=np.float64)
  for i, result in enumerate(results):
      a = i % maskHeight
      b = i / maskHeight

      mask[a,b] = result[1]

  print("Mask created, saving...")
  with open('maskFrames/%06d.pickle'%frameId, 'wb') as f:
    pickle.dump(mask, f)

  outMask = cv2.cvtColor((mask * 255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
  cv2.imwrite("maskFrames/frame_%06d.jpg"%frameId,outMask)

  print("Completed frame_%06d.jpg"%frameId)

  ret, frame = capture.read()
  frameId += 1