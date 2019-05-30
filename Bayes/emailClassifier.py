import random

import textVectorizer
import bayesClassifier


def loadDataSet():
    samples, labels = [], []
    for i in range(1, 26):
        s = open('./data/email/spam/%d.txt' % (i), encoding='utf-8').read()
        samples.append(s)
        labels.append(1)        
        s = open('./data/email/ham/%d.txt' % (i), encoding='utf-8').read()
        samples.append(s)
        labels.append(0)
    return samples, labels

def normalizeDataSet(samples):
    vectorizer = textVectorizer.textVectorizer()
    for s in samples:
        vectorizer.fit(s)
    nsamples = []
    for s in samples:
        nsamples.append(vectorizer.transform(s))
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

    classifier = bayesClassifier.bayesClassifier()
    classifier.fit(trainingSamples, trainingLabels)
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


def classifyEmail():
    samples, labels = loadDataSet()
    nsamples = normalizeDataSet(samples)
    trainingSet, testSet = splitDataSet(len(samples), 0.15)
    classifier = trainClassifier(nsamples, labels, trainingSet)
    testClassifier(classifier, nsamples, labels, testSet)


if __name__ == "__main__":
    classifyEmail()
