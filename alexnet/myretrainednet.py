import tensorflow as tf
import pickle
import random
import numpy as np
import cv2
from alexnet.myalexnet import getBottlenecks

with open("./bottlenecks/bottlenecks", "rb") as f:
	importedData = pickle.load(f)
Xs = np.array(importedData[0])
Ys = np.array(importedData[1])
TRAINING_SIZE = 10000

def getTrainingData():
	return Xs[:TRAINING_SIZE], Ys[:TRAINING_SIZE]
def getTestData():
	return Xs[TRAINING_SIZE:], Ys[TRAINING_SIZE:]

def getTrainingBatch(size):
	xs, ys = getTrainingData()
	indexs = range(0,len(xs))
	batch = random.sample(indexs,size)

	batch_xs = [xs[i] for i in batch]
	batch_ys = [ys[i] for i in batch]
	return batch_xs, batch_ys

#############################################################################################

x_ = tf.placeholder("float", shape=[None, 4096])
y_ = tf.placeholder("float", shape=[None, 2])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

my_fc8W = weight_variable([4096, 2])
my_fc8b = bias_variable([2])
my_fc8 = tf.nn.xw_plus_b(x_, my_fc8W, my_fc8b)
my_prob = tf.nn.softmax(my_fc8)

#train model
cross_entropy = -tf.reduce_sum(y_*tf.log(my_prob))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(my_prob,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#############################################################################################

saver = tf.train.Saver()

saver.restore(sess, "./saves/model_009900.ckpt")

# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
#
# for i in range(10001):
# 	batch = getTrainingBatch(50)
# 	if i%100 == 0:
# 		print("step %d, training accuracy %g"%(i,sess.run(accuracy, feed_dict={x_:batch[0], y_: batch[1]})))
# 		save_path = saver.save(sess, "./saves/model_%06d.ckpt"%i)
# 	sess.run(train_step, feed_dict={x_: batch[0], y_: batch[1]})
#
# test_xs, test_ys = getTestData()
# print("test accuracy %g"%sess.run(accuracy, feed_dict={x_: test_xs, y_: test_ys}))

windowSize = 227
videoPath = "vanuatu35.mp4"
capture = cv2.VideoCapture(videoPath)

frameSize = (640,360)
minFrameSize = (64,36)
numOfSizes = 10
frameWidths = list(reversed(range(minFrameSize[0],frameSize[0]+1,frameSize[0]/numOfSizes)))
frameHeights = list(reversed(range(minFrameSize[1],frameSize[1]+1,frameSize[1]/numOfSizes)))

IndexsPerSize = []
for i in range(numOfSizes):
	fWidth = frameWidths[i]
	fHeight = frameHeights[i]

	fxvs, fyvs = [],[]
	for i in range(0,fWidth-windowSize,windowSize/2):
		for j in range(0,fHeight-windowSize,windowSize/2):
			xv,yv = np.meshgrid(range(i,i+windowSize),range(j,j+windowSize))
			fxvs.append(xv)
			fyvs.append(yv)

	IndexsPerSize.append((fxvs,fyvs))

ret, frame = capture.read()
frameId = 0
while ret:
	frame = cv2.resize(frame, (frameSize[0],frameSize[1]))
	height,width,depth = frame.shape
	frame = frame.astype(np.float32) / 255.0

	resultsPerSize = []
	for i in range(numOfSizes):
		# print("%s,%s"%((frameWidths[i],frameHeights[i])))
		smallFrame = cv2.resize(frame, (frameWidths[i],frameHeights[i]))
		indexs = IndexsPerSize[i]
		samples = smallFrame[indexs[1],indexs[0]]
		# print(np.array(samples).shape)

		bottlenecks = getBottlenecks(samples)
		results = sess.run(my_prob, feed_dict={x: bottlenecks})
		resultsPerSize.append(results)

	with open('resultsPerSize/%06d.pickle'%frameId, 'wb') as f:
		pickle.dump(resultsPerSize, f)

	ret, frame = capture.read()
	frameId += 1