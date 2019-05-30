import treeClassifier

def testTreeClassifier():
    samples = [[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]]
    labels = [1, 1, 0, 0, 0]
    test = [[1, 0], [1, 1]]

    classifier = treeClassifier.treeClassifier()
    classifier.fit(samples, labels)
    # classifier.save("./model/tree.bin")
    # classifier.load("./model/tree.bin")

    for t in test:
        print("%s -- %s" % (t, classifier.predict(t)))
    
if __name__ == "__main__":
    testTreeClassifier()