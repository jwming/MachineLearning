import random
import numpy as np

import kNNClassifier

def loadDataSet(file):
    samples, labels = [], []
    with open(file) as f:
        for r in f.readlines():
            s = r.strip().split('\t')
            samples.append([float(s[0]), float(s[1]), float(s[2])])
            labels.append(int(s[3]))
    return samples, labels

def normalizeDataSet(samples):
    min = np.min(samples, 0) 
    max = np.max(samples, 0)
    dim = (len(samples), 1)
    nsamples = (samples - np.tile(min, dim))/np.tile(max-min, dim)
    return nsamples

def splitDataSet(num, testRatio = 0.15):
    trainingSet, testSet = list(range(num)), []
    for i in range(int(num*testRatio)):
        ri = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[ri])
        del(trainingSet[ri])
    return trainingSet, testSet

def trainClassifier(samples, labels, trainingSet):
    trainingSamples = []
    trainingLabels = []
    for i in trainingSet:
        trainingSamples.append(samples[i])
        trainingLabels.append(labels[i])
    classifier = kNNClassifier.kNNClassifier()
    classifier.fit(trainingSamples, trainingLabels, 20)
    return classifier

def testClassifier(classifier, samples, labels, testSet):
    errorCount = 0
    for i in testSet:
        label = classifier.predict(samples[i])
        if __debug__ or label != labels[i]:
            print("sample index %d, real %s, predict %s" % (i, labels[i], label))
        if label != labels[i]:
            errorCount += 1            
    print("total error rate: %f" % (float(errorCount) / len(testSet)))

def classifyDating():
    samples, labels = loadDataSet("./data/datingTestSet2.txt")
    nsamples = normalizeDataSet(samples)
    trainingSet, testSet = splitDataSet(len(nsamples), 0.1)   
    classifier = trainClassifier(nsamples, labels, trainingSet)
    testClassifier(classifier, nsamples, labels, testSet)

if __name__ == "__main__":
    classifyDating()