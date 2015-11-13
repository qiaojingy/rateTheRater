import json
import collections
import os
import softmax
from util import *

#list of stop words
from nltk.corpus import stopwords
from nltk import PorterStemmer
cachedStopWords = stopwords.words("english")

trainingFile = "train.json"
testFile = "test.json"
processedTrainingFile = "train_processed.json"
processedTestFile = "test_processed.json"

def textPrepare(data):
  x = data['text']
  import string
  tokens = [x.strip(':;,.?"()') for x in string.split(x)]
  tokens = [x for x in tokens if x != '']
  tokens = [PorterStemmer().stem_word(x) for x in tokens if x not in cachedStopWords]
  data['text'] = ' '.join(tokens)
  return data

def loadTrainingData():
  # Load training data
  print "=" * 30
  print "Loading Training Data"
  if not os.path.isfile(processedTrainingFile):
    print "=" * 30
    print "Preprocessing Reviews"
    f = open(trainingFile, 'r')
    trainingData = list()
    while (1):
      line = f.readline()
      if not line:
        break
      trainingData.append(json.loads(line))
    processedTrainingData = list()
    for data in trainingData:
      processedTrainingData.append(textPrepare(data))
    f = open(processedTrainingFile, 'w')
    j_dumped = []
    for item in processedTrainingData:
      j_dumped.append(json.dumps(item)+'\n')
    f.writelines(j_dumped)
    f.close()
    
  f = open(processedTrainingFile, 'r')
  trainingData = list()
  while (1):
    line = f.readline()
    if not line:
      break
    trainingData.append(json.loads(line))
  return trainingData

def loadTestData():
  # Load test data
  print "=" * 30
  print "Loading Test Data"
  if not os.path.isfile(processedTestFile):
    print "=" * 30
    print "Preprocessing Reviews"
    f = open(testFile, 'r')
    testData = list()
    while (1):
      line = f.readline()
      if not line:
        break
      testData.append(json.loads(line))
    processedTestData = list()
    for data in testData:
      processedTestData.append(textPrepare(data))
    f = open(processedTestFile, 'w')
    j_dumped = []
    for item in processedTestData:
      j_dumped.append(json.dumps(item)+'\n')
    f.writelines(j_dumped)
    f.close()
    
  f = open(processedTestFile, 'r')
  testData = list()
  while (1):
    line = f.readline()
    if not line:
      break
    testData.append(json.loads(line))
  return testData

def extractWordFeatures(data):
  x = data['text']
  import string
  tokens = string.split(x)
  featureDict = dict()
  for token in tokens:
      featureDict[token] = featureDict.get(token, 0) + 1
  return featureDict

def extractIdFeatures(data):
  featureDict = dict()
  id = data['user_id']
  featureDict[id] = 1
  return featureDict

def extractWordAndIdFeatures(data):
  featureDict = extractWordFeatures(data)
  featureDict.update(extractIdFeatures(data))
  return featureDict

def perceptron(trainExamples, testExamples, featureExtractor):
  print "=" * 30
  print "Doing Perceptron"
  weights = {}  # feature => weight
  numIters = 20
  for iter in range(numIters):
    print "Iteration   ", iter+1
    for (x, y) in trainExamples:
      features = featureExtractor(x)
      margin = dotProduct(weights, features) * y
      if margin >= 1:
        continue
      else:
	for f, v in features.items():
          weights[f] = weights.get(f, 0) + v * y   
  trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
  print "=" * 30
  print "Result"
  print 'Iter : %s   Training error = %s' % (iter+1, trainError)
  
  testError = evaluatePredictor(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
  print "=" * 30
  print "Result"
  print 'Iter : %s   Test error = %s' % (iter+1, testError)
  return weights

def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)

if __name__ == '__main__':
  # Load training data
  trainingData = loadTrainingData()
  testData = loadTestData()
  trainExamples = [(data, 1 if data['stars'] > 3 else -1) for data in trainingData]
  testExamples = [(data, 1 if data['stars'] > 3 else -1) for data in testData]
  perceptron(trainExamples, testExamples, extractWordAndIdFeatures)
  trainExamples = [(data, data['stars']) for data in trainingData]
  testExamples = [(data, data['stars']) for data in testData]
  softmax.train(trainExamples, testExamples, extractWordFeatures)
