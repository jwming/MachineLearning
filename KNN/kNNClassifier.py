import numpy as np

class kNNClassifier(object):   
    def fit(self, samples, labels, k):
        self._samples = np.array(samples)
        self._labels = labels
        self._k = k
    
    def predict(self, sample):
        dist = ((np.tile(sample, (self._samples.shape[0], 1)) - self._samples)**2).sum(1)**0.5    
        distIndex = dist.argsort()
        labelCount = {}
        for i in range(self._k):
            label = self._labels[distIndex[i]]
            labelCount[label] = labelCount.get(label, 0) + 1
        return max(labelCount, key = labelCount.get)
