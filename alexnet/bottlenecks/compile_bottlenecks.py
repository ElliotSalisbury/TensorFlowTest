import os
import pickle

full_Xs = []
full_Ys = []
for samplePickle in os.listdir("./positive"):
	with open("./positive/%s" % samplePickle, "rb") as f:
		positive = pickle.load(f)[0]
	with open("./negative/%s" % samplePickle, "rb") as f:
		negative = pickle.load(f)[0]

	full_Xs.append(positive[0])
	full_Xs.append(negative[0])

	full_Ys.append(positive[1])
	full_Ys.append(negative[1])

with open("./bottlenecks", "wb") as f:
	pickle.dump([full_Xs, full_Ys], f)