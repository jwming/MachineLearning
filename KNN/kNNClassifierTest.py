import kNNClassifier

def testKNNClassifier():
    samples = [[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]
    labels = ['A', 'A', 'B', 'B']
    test = [[0.5, 0.4], [0.8, 0.9]]

    classifier = kNNClassifier.kNNClassifier()
    classifier.fit(samples, labels, 2)
    
    for t in test:
        print("%s -- %s" % (t, classifier.predict(t)))

if __name__ == "__main__":
    testKNNClassifier()
