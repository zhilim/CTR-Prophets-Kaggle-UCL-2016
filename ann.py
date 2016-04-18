import numpy as np
import math

hiddenSize = 32

synapse_0 = 0
synapse_1 = 0
layer_0 = 0
layer_1 = 0
layer_2 = 0

def sigmoid(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Convenient function to separate the array of results of the array of dataset
def separate_result(dataset):
    training = []
    results = []
    for i in dataset:
        results.append(int(i[0]))
        training.append(i[1:])
    return training, results

def ann(dataset, learning_rate, iters):

    training, results = separate_result(dataset)

    x = np.array(training) # Transform to np array
    y = np.array([results]).T

    np.random.seed(1)

    global synapse_0
    global synapse_1
    synapse_0 = 2*np.random.random((len(x[0]),hiddenSize)) - 1
    synapse_1 = 2*np.random.random((hiddenSize,1)) - 1

    for j in xrange(iters):

        # Initialise weights
        layer_0 = x
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        layer_2_error = layer_2 - y

        if (j % 1000) == 0:
            print "Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error)))
            tp = 0
            fp = 0
            for i in range(len(y)):
                if(y[i] == 1):
                    if (layer_2[i][0] > 0.5):
                        tp = tp + 1
                        # print ("Correct positive!" + str(layer_2[i]))
                if(y[i] == 0):
                    if (layer_2[i][0] > 0.5):
                        fp = fp + 1
                        # print ("False positive!" + str(layer_2[i]))
            if ((fp + tp) != 0):
                print str(tp) + "," + str(fp)
                print (float(tp) / float(fp + tp))
            else:
                print "0 positives!"

        layer_2_delta = layer_2_error * sigmoid(layer_2, True)
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        layer_1_delta = layer_1_error * sigmoid(layer_1,True)

        synapse_0 -= learning_rate * (layer_0.T.dot(layer_1_delta))
        synapse_1 -= learning_rate * (layer_1.T.dot(layer_2_delta))

    return synapse_1

def predict(dataset, testformat):
    prediction = []
    filename = "prediction.txt"
    if testformat:
        filename = "test_pred.csv"
    f = open(filename, 'w')
    f.write("Id,Prediction\n")
    iden = 1
    global synapse_0
    global synapse_1
    if (testformat == False):
        # Remove actual prediction from dataset
        aux = []
        for i in dataset:
            aux.append(i[1:])
        dataset = aux

    x = np.array(dataset)
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0,synapse_0))
    layer_2 = sigmoid(np.dot(layer_1,synapse_1))
    
    for d in layer_2:
        prob = round(d[0], 6)
        line = str(iden) + "," + str(prob) + "\n"

        f.write(line)

        iden += 1
        prediction.append(prob)

    f.close()

    return prediction
