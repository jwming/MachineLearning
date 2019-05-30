import random

import treeClassifier

def loadDataSet(file):
    samples, labels = [], []
    with open(file) as f:
        for r in f.readlines():
            s = r.strip().split('\t')
            samples.append(s[:4])
            labels.append(s[4])
    return samples, labels

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

    classifier = treeClassifier.treeClassifier()
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


def classifyLenses():
    samples, labels = loadDataSet("./data/lenses.txt")
    trainingSet, testSet = splitDataSet(len(samples), 0.15)
    classifier = trainClassifier(samples, labels, trainingSet)
    testClassifier(classifier, samples, labels, testSet)


if __name__ == "__main__":
    classifyLenses()
