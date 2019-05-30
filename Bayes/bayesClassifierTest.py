import textVectorizer
import bayesClassifier


def testBayesClassifier():
    samples = ['my dog has flea problems, help please',
               'maybe not, take him to dog park, stupid',
               'my dalmation is so cute, I love him',
               'stop posting stupid worthless garbage',
               'mr licks ate my steak, how to stop him',
               'quit buying worthless dog food, stupid']
    labels = [0, 1, 0, 1, 0, 1]
    test = ['i, love my dalmation',
            'you, a stupid garbage']
    
    transformer = textVectorizer.textVectorizer()
    for s in samples:
        transformer.fit(s)
    nsamples = []
    for s in samples:
        nsamples.append(transformer.transform(s))
    
    classifier = bayesClassifier.bayesClassifier()
    #classifier.fit(nsamples, labels)
    #classifier.save('./model/bayes.bin')
    classifier.load('./model/bayes.bin')

    for t in test:
        print("%s -- %s" % (t, classifier.predict(transformer.transform(t))))

if __name__ == '__main__':
    testBayesClassifier()
