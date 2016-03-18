import numpy as np 
import pandas as pd 
import sys
import math
import psotest as pso 
import matplotlib.pyplot as plt 
import ast
import random

#qds is the quantified dataset (orthogonal representation/dummy encoded)
#selection is the features you would like to use in learning (numbers like described on kaggle description)
#warning: no support for features that have unique instances(levels) of more than 100 (shouldnt use those anyway)
global qds, selection
selection = [6,10,15,17,18,20,21]

#loads data from file 
#@param filename: path to file
#@param sample: boolean as to whether or not to extract just a sample
#@param samplesize: size of sample to extract
def loadData(filename, sample, samplesize):
	print "loading data.."
	dataset = []

	with open(filename, 'r') as f:
		progress = 0
		for line in f:
			progress += 1
			if sample and progress >= samplesize:
				break
			#sys.stdout.write("Loading Data.. %d	\r" % (progress))
			#sys.stdout.flush()
			dataset.append(line.split('\t'))
	return dataset

#calculates and prints statistical information of the dataset to: 'stats.txt' in your working directory
def getStats(dataset):
	clicks = 0
	uniqueCount = []
	keylist = ['weekday', 'hour', 'timestamp', 'logtype', 'user', 'agent', 'ip', 'region', 'city', 'adex',
	'domain', 'url', 'anon', 'slotid', 'width', 'height', 'vis', 'format', 'price', 'creative', 'keypage', 'advertId', 'usertags']
	for key in keylist:
		uniqueCount.append({})
	for d in dataset:
		if d[0] == '1':
			clicks += 1
		for i in range(1,len(d)):
			if d[i] not in uniqueCount[i-1]:
				if d[0] == '1': 
					uniqueCount[i-1][d[i]] = 1
				else:
					uniqueCount[i-1][d[i]] = 0
			else:
				if d[0] == '1':
					uniqueCount[i-1][d[i]] += 1
				else:
					pass
	f = open('stats.txt', 'w')
	for i in range(len(keylist)):
		if len(uniqueCount[i]) > 500:
			continue
		un = "Number of Unique Instances of " + keylist[i] + ": " + str(len(uniqueCount[i])) + ", "
		f.write(un)
		clicklist = []
		for key in uniqueCount[i]:
			clicklist.append(float(uniqueCount[i][key]))
		avg = np.mean(clicklist)
		std = np.std(clicklist)
		if std == 0:
			continue
		st = "Average: " + str(avg) + ", STDEV: " + str(std) + "\n"
		f.write(st)
		for key in uniqueCount[i]:
			clickRate = float(uniqueCount[i][key])/float(clicks)
			zscore = (float(uniqueCount[i][key])-avg)/std
			cr = "Click Rate for Level " + key + ": " + str(clickRate) + ", Zscore = " + str(zscore) + "\n"
			f.write(cr)
	f.close()

#creates template for orthogonal representation of categorical variables from all the levels
#currently existing in each feature of @dataset
#returns a dict of {string:pd.Dataframe} where string is the key from keylist and the dataframe contains the dummies
#also prints the text representation to 'dummy.txt'. from which u can generate the dummylist
def dummify(dataset):
	dummylist = []
	keylist = ['weekday', 'hour', 'timestamp', 'logtype', 'user', 'agent', 'ip', 'region', 'city', 'adex',
	'domain', 'url', 'anon', 'slotid', 'width', 'height', 'vis', 'format', 'price', 'creative', 'keypage', 'advertId', 'usertags']
	for key in keylist:
		dummylist.append({})
	print "counting levels"
	for d in dataset:
		for i in range(len(keylist)):
			level = d[i+1]
			if keylist[i] == 'agent':
				level = d[i+1].split('_')[0]
			if level not in dummylist[i]:
				dummylist[i][level] = 1
			else:
				pass
	print "collecting dummies"
	f = open('dummy.txt', 'w')
	f.write(str(dummylist))
	f.close()
	realdummylist = {}
	for i in range(len(dummylist)):
		if len(dummylist[i]) > 100:
			continue
		realdummylist[keylist[i]] = pd.get_dummies(dummylist[i].keys())
	print realdummylist['agent']	
	return realdummylist

#doesnt work well for large datasets, just dont use this or you will crash
def dummyFromFile():
	keylist = ['weekday', 'hour', 'timestamp', 'logtype', 'user', 'agent', 'ip', 'region', 'city', 'adex',
	'domain', 'url', 'anon', 'slotid', 'width', 'height', 'vis', 'format', 'price', 'creative', 'keypage', 'advertId', 'usertags']
	f = open('dummy.txt', 'r')
	line = f.readline()
	dummylist = ast.literal_eval(line)
	realdummylist = {}
	for i in range(len(dummylist)):
		if len(dummylist[i]) > 100:
			continue
		realdummylist[keylist[i]] = pd.get_dummies(dummylist[i].keys())
	print realdummylist['agent']
	return realdummylist	

#converts dataset to orthogonal representation
#@dataset: the dataset
#@dummylist: the dict returned from function dummify(dataset)
#@param testformat: boolean stating if we are quantifying a test dataset
#returns a list of quantified datapoints
#the first element of returned list is still the target variable (the click or no click)
def quantifyData(dataset, dummylist, testformat):
	print "Quantifying Categorical Data.. This may take a while"
	keydict = {1:'weekday', 2:'hour', 3:'timestamp', 4:'logtype', 5:'user', 6:'agent', 7:'ip', 8:'region', 9:'city', 10:'adex',
	11:'domain', 12:'url', 13:'anon', 14:'slotid', 15:'width', 16:'height', 17:'vis', 
	18:'format', 19:'price', 20:'creative', 21:'keypage', 22:'advertId', 23:'usertags'}
	quantified = []
	for d in dataset:
		newd = []
		if not testformat:
			newd.append(d[0])
			for i in selection:
				dummyset = dummylist[keydict[i]]
				level = d[i]
				if keydict[i] == 'agent':
					level = d[i].split('_')[0]
				dummy = list(dummyset[level])
				newd = newd + dummy
		elif testformat:
			for i in selection:
				dummyset = dummylist[keydict[i]]
				level = d[i-1]
				if keydict[i] == 'agent':
					level = d[i-1].split('_')[0]
				dummy = list(dummyset[level])
				newd = newd + dummy
		quantified.append(newd)
	print quantified[5]
	return quantified

#remember this from logistic regression?
def hypo(features, weights):
	#z = sum([x*w for x,w in zip(features, weights)])
	z = 0
	for i in range(len(features)):
		z += features[i] * weights[i]
	z += weights[-1]
	try:
		gz = 1/(1+math.exp(-z))
	except:
		if z > 0:
			gz = 0.99999999
		else:
			gz = 0.00000001
	return gz

#logistic regression cost function
#uses global qds, so make sure that is properly initialised first
def cost(weights):
	sumerror = 0	
	for d in qds:
		z = 0
		for i in range(1,len(d)):
			z += d[i] * weights[i-1]
		try:
			gz = 1/(1+math.exp(-z))
		except:
			if z > 0:
				gz = 1
			else:
				gz = 0
		y = int(d[0])
		if gz == 1:
			gz = 0.9999999999
		if gz == 0:
			gz = 0.0000000001
		sumerror = sumerror - y*(math.log(gz))*500 - (1-y)*(math.log(1-gz))*0.5
	return float(sumerror)/float(len(qds))

#function to learn using PSO
#writes resultant weight vector to file, from which u can load again if its good
def learn(search):
	pso.dmn = len(qds[0])
	pso.searchRange = float(search)
	s,i,g = pso.particleSwarmOptimize(MCC, True, True)
	f = open('weight.txt', 'w')
	f.write(str(s))
	f.close()
	return s, i, g

#gradient descent learning
#@param lrate: learning rate, iters:max iteration, r: range of initial weights
#also writes weights to file
def gdescent(lrate, iters, r):
	w = [random.uniform(-r,r)] * (len(qds[0]))
	iteration = 0
	while iteration < iters:
		c = cost(w)
		for d in qds:
			tempweights = []
			term1 = lrate * (int(d[0]) - hypo(d[1:], w))
			for i in range(1,len(d)):
				new = w[i-1] + (term1*d[i])
				tempweights.append(new)
			intercept = w[-1] + term1
			tempweights.append(intercept)
			w = list(tempweights)
		
		print "Iteration: " + str(iteration) + ", cost: " + str(c)
		print "Optimum: " + str(w)
		iteration += 1
	f = open('weight.txt', 'w')
	f.write(str(w))
	f.close()
	return w

#generate list of prediction on @param dataset (test data)
#@param testformat: boolean if test format or not
#@param weights duh
def predict(weights, dataset, testformat):
	prediction = []
	filename = "prediction.txt"
	if testformat:
		filename = "test_pred.csv"
	f = open(filename, 'w')
	f.write("Id,Prediction\n")
	iden = 1
	for d in dataset:
		z = 0
		if not testformat:
			for i in range(1,len(d)):
				z += d[i] * weights[i-1]
			z += weights[-1]
		elif testformat:
			for i in range(len(d)):
				z += d[i] * weights[i]
			z += weights[-1]
		prob = 1/(1+math.exp(-z))
		#prob = prob * 10
		prob = round(prob, 6)
		line = str(iden) + "," + str(prob) + "\n"
		f.write(line)
		
		iden += 1
		prediction.append(prob)

	f.close()

	return prediction

#matthew's correlation coefficient, for PSO
def MCC(weights):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for d in qds:
		z = 0
		for i in range(1,len(d)):
			z += d[i] * weights[i-1]
		z += weights[-1]
		try:
			prob = 1/(1+math.exp(-z))
		except:
			if z > 0:
				prob = 0.9999999
			else:
				prob = 0.0000001
		if d[0] == '1':
			if prob >= 0.5:
				tp += 1
			else:
				fn += 1
		else:
			if prob >= 0.5:
				fp += 1
			else:
				tn += 1
	stpfp = math.sqrt(tp+fp)
	stpfn = math.sqrt(tp+fn)
	stnfp = math.sqrt(tn+fp)
	stnfn = math.sqrt(tn+fn)
	denom = stpfp * stpfn * stnfp * stnfn
	num = tp*tn - fp*fn	
	mcc = 0
	if denom != 0:
		mcc = float(num)/float(denom)
	return -mcc

#helper function to calculate true positive and falso positive rates	
def accuracy(prediction, actual, tilt):
	positives = 0
	negatives = 0
	pred_right = 0
	pred_wrong = 0

	for i in range(len(actual)):
		if int(actual[i][0]) == 1:
			positives += 1
			if prediction[i] >= tilt:
				pred_right += 1
		if int(actual[i][0]) == 0:
			negatives += 1
			if prediction[i] >= tilt:
				pred_wrong += 1
	tprate = float(pred_right)/float(positives)
	fprate = float(pred_wrong)/float(negatives)

	return tprate, fprate

#chunk the data
def chunkit(ds, lsize, tsize):
	forlearning = ds[:lsize]
	fortesting = ds[lsize:lsize+tsize]
	return forlearning, fortesting

#extracts all the clicks=1 datapoints from the data,
#mixes it with random select of clicks=0 datapoints in @param proportion 1x or 2x or 3x
#this is needed for logistic regression because there is 2mil datapoints but only 2k clicks=1
#rare event
def balance(ds, proportion):
	clicks = []
	noclicks = []
	for d in ds:
		if d[0] == '1':
			clicks.append(d)
		else:
			noclicks.append(d)
	#random.shuffle(clicks)
	random.shuffle(noclicks)
	print len(clicks)
	nclen = len(clicks) * proportion
	final = clicks + noclicks[:nclen]
	random.shuffle(final)
	print len(final)
	return final

def loadPrediction():
	f = open('prediction.txt', 'r')
	prediction = []
	for line in f:
		l = line.split(',')
		prediction.append(float(l[1]))
	return prediction

def loadWeights():
	f = open('weight.txt', 'r')
	line = f.readline()
	w = ast.literal_eval(line)
	return w

#plot da ROC
def plotROC(prediction, learning):
	tilts = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
	ys = []
	xs = []
	for tilt in tilts:
		y, x = accuracy(prediction, learning, tilt)
		print y, x
		ys.append(y)
		xs.append(x)
	plt.clf()
	plt.plot(xs,ys)
	plt.savefig('ROC.png')
	return xs, ys

#get da AUC
def AUC(x, y):
	summ = 0
	for k in range(1, len(x)):
		summ = summ + ((x[k] - x[k-1])*(y[k]+y[k-1]))

	return 0.5*summ


ds = loadData('data_train.txt', False, 10000)
#ts = loadData('shuffle_data_test.txt', False, 100)
print ds[0]
#getStats(ds)
#bds = balance(ds, 1)


dum = dummify(ds)
random.shuffle(ds)
#dum = dummyFromFile()
#qds = quantifyData(ds, dum)

learning, testing = chunkit(ds, 10000, 1000)
#print len(learning), len(testing)

#meta = computeMeta(ds)
qds = quantifyData(learning, dum, False)
#testing = ds[:200000]
tds = quantifyData(testing, dum, False)
#print len(qds[0])


#tds = quantifyData(testing, dum)
s, i, g = learn(10)
#s = gdescent(0.1,40,10)
#s = loadWeights()
prediction = predict(s, tds, False)
#prediction = loadPrediction()
#print accuracy(prediction, testing, 0.5)
x,y = plotROC(prediction, testing)
print AUC(x, y)
