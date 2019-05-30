import math
import pickle

class treeClassifier(object):
    def load(self, filename):
        with open(filename, mode="rb") as f:
            self._tree = pickle.load(f)
    
    def save(self, filename):
        with open(filename, mode="wb") as f:
            pickle.dump(self._tree, f)

    def fit(self, samples, labels):
        featureIndexes = list(range(len(samples[0])))
        self._tree = self._createTree(samples, labels, featureIndexes)

    def predict(self, sample):
        return self._searchFeatureLabel(self._tree, sample)   

    def _calcEntropy(self, samples, labels):
        labelCounts = {}
        for label in labels:
            labelCounts[label] = labelCounts.get(label, 0) + 1
        entropy = 0.0
        sampleCount = len(samples)
        for label in labelCounts:
            prob = float(labelCounts[label])/sampleCount
            entropy -= prob * math.log2(prob)
        return entropy
        
    def _reduceSamples(self, samples, labels, feature, value):
        subSamples, subLabels = [], []
        for i in range(len(samples)):
            if samples[i][feature] == value:
                v = samples[i][:feature]
                v.extend(samples[i][feature+1:])
                subSamples.append(v)
                subLabels.append(labels[i])
        return subSamples, subLabels
    
    def _selectPrimaryFeature(self, samples, labels):
        primaryFeature = -1
        minEntropy = float('inf')
        for feature in range(len(samples[0])):
            featureEntropy = 0.0
            featureValues = set([sample[feature] for sample in samples])         
            for value in featureValues:
                subSamples, subLabels = self._reduceSamples(samples, labels, feature, value)
                prob = float(len(subSamples))/len(samples)
                featureEntropy += prob * self._calcEntropy(subSamples, subLabels)
            if (featureEntropy < minEntropy):
                minEntropy = featureEntropy
                primaryFeature = feature
        return primaryFeature

    def _selectPrimaryLabel(self, labels):
        labelCounts = {}
        for label in labels:
            labelCounts[label] = labelCounts.get(label, 0) + 1
        return max(labelCounts, key = labelCounts.get)

    def _createTree(self, samples, labels, featureIndexes):
        if len(labels) == labels.count(labels[0]): # only one unique label
            return labels[0]
        if samples[0] == None or len(samples[0]) == 0: # no feature value
            return self._selectPrimaryLabel(labels)

        feature = self._selectPrimaryFeature(samples, labels)        
        featureIndex = featureIndexes[feature]
        del(featureIndexes[feature])
        featureTree = {featureIndex : {}}
        featureValues = set([sample[feature] for sample in samples])
        for value in featureValues:
            subSamples, subLabels = self._reduceSamples(samples, labels, feature, value)
            subFeatureIndexes = featureIndexes[:]
            featureTree[featureIndex][value] = self._createTree(subSamples, subLabels, subFeatureIndexes)
            if __debug__:
                print("featureTree[%s][%s]: %s" % (featureIndex, value, featureTree[featureIndex][value]))
        if __debug__:
            print("featureTree[%s]: %s" % (featureIndex, featureTree[featureIndex]))
        return featureTree

    def _searchFeatureLabel(self, tree, sample):
        feature = next(iter(tree))
        value = sample[feature]
        label = tree[feature][value]
        if isinstance(label, dict):
            label = self._searchFeatureLabel(label, sample)
        return label
