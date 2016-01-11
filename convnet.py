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


sess = tf.InteractiveSession()


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 2])

W = tf.Variable(tf.zeros([784,2]))
b = tf.Variable(tf.zeros([2]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
  batch_xs, batch_ys = getTrainingBatch(50)
  train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

test_xs, test_ys = getTestData()
print(accuracy.eval(feed_dict={x: test_xs, y_: test_ys}))

### Multilayer Convolutional Network

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

saver = tf.train.Saver()

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(20000):
  batch = getTrainingBatch(50)
  if i%100 == 0:
    print("step %d, training accuracy %g"%(i,sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})))
    save_path = saver.save(sess, "/saves/model_%06d.ckpt"%i)
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0}))

def img2data(img):
  data = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  data = data.astype(np.float32) / 255.0
  data = data.flatten()

  return data

windowSize = 28
videoPath = "vanuatu35.mp4"
capture = cv2.VideoCapture(videoPath)

ret, frame = capture.read()
frameId = 0
while ret:
  frame = cv2.resize(frame, (640,360))
  height,width,depth = frame.shape

  maskHeight = height-windowSize
  maskWidth = width-windowSize

  data = []
  for x in range(0,maskWidth):
      for y in range(0,maskHeight):
        window = frame[y:y+windowSize,x:x+windowSize]
        data.append(img2data(window))

  results = []

  mask = np.zeros((maskHeight,maskWidth),dtype=np.float64)
  for i, result in enumerate(results):
      y = i % maskHeight
      x = i / maskHeight

      mask[y,x] = result[1]

  with open('maskFrames/%06d.pickle'%frameId, 'wb') as f:
    pickle.dump(mask, f)

  outMask = cv2.cvtColor((mask * 255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
  cv2.imwrite("maskFrames/frame_%06d.jpg",outMask)

  ret, frame = capture.read()
  frameId += 1