import numpy as np
import pickle
import textVectorizer

class bayesClassifier(object):
    def load(self, filename):
        with open(filename, mode="rb") as f:
            self._samplesP0 = pickle.load(f)
            self._samplesP1 = 1.0 - self._samplesP0
            self._termsP0 = pickle.load(f)
            self._termsP1 = pickle.load(f)
    
    def save(self, filename):
        with open(filename, mode="wb") as f:
            pickle.dump(self._samplesP0, f)            
            pickle.dump(self._termsP0, f)
            pickle.dump(self._termsP1, f)

    def fit(self, samples, labels):
        samplesCount, termsCount = len(samples), len(samples[0])
        self._samplesP1 = np.sum(labels) / float(samplesCount)
        self._samplesP0 = 1.0 - self._samplesP1
        termsP0Count, termsP1Count = np.ones(termsCount), np.ones(termsCount)
        for i in range(samplesCount):
            if labels[i] == 0:
                termsP0Count += samples[i]
            else:
                termsP1Count += samples[i]
        self._termsP0 = np.log(termsP0Count / np.sum(termsP0Count))
        self._termsP1 = np.log(termsP1Count / np.sum(termsP1Count))

    def predict(self, sample):
        p0 = np.sum(sample * self._termsP0) + np.log(self._samplesP0)
        p1 = np.sum(sample * self._termsP1) + np.log(self._samplesP1)
        if p0 > p1:
            return 0
        else:
            return 1
