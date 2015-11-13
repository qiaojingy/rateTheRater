from baseline import *
from util import *
import math

def classProbs(weights, features):
  probs = []
  k = len(weights)
  for i in range(k):
    probs.append(math.exp(dotProduct(weights[i], features)))
  s = sum(probs)
  for i in range(k):
    probs[i] = probs[i] * 1.0/s
  return probs

def train(trainExamples, testExamples, featureExtractor):
  print "=" * 30
  print "Doing Softmax"
  alpha = 0.1
  weights = []
  for i in range(5):
    weights.append({})
  numIters = 5
  for iter in range(numIters):
    print "iteration   ", iter+1
    for (x, y) in trainExamples:
      features = featureExtractor(x)
      N = []
      for i in range(5):
        if i+1 == y:
          N.append(features)
        else:
          N.append({})
      probs = classProbs(weights, features)
      for i in range(5):
        increment(N[i], -probs[i], features)
      for i in range(5):
        increment(weights[i], 0.01, N[i])
  trainError = evaluatePredictor(trainExamples, lambda(x) : max((prob, k) for (k, prob) in enumerate(classProbs(weights, featureExtractor(x))))[1] + 1)
  print "=" * 30
  print "Result"
  print 'Iter : %s   Training error = %s' % (iter+1, trainError)

  """
  testError = evaluatePredictor(testExamples, lambda(x) : max((prob, k) for (k, prob) in enumerate(classProbs(weights, featureExtractor(x))))[1] + 1)
  print "=" * 30
  print "Result"
  print 'Iter : %s   Test error = %s' % (iter+1, testError)
  """
  return weights

      


  

