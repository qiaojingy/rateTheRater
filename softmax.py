from baseline import *
from util import *
import math

# Calculating the probability of belonging to each class, given weights and features
"""
def classProbs(weights, features):
  probs = []
  k = len(weights)
  for i in range(k):
    probs.append(math.exp(dotProduct(weights[i], features)))
  s = sum(probs)
  for i in range(k):
    probs[i] = probs[i] * 1.0/s
  return probs
"""

def classProbs(weights, features):
  probs = []
  k = len(weights)
  for i in range(k):
    w1 = dotProduct(weights[i], features)
    temp = []
    for j in range(k):
      if j == i:
        temp.append(1)
      else:
        temp.append(math.exp(dotProduct(weights[j], features) - w1))
    s = sum(temp)
    probs.append(1.0/s)
  return probs

# Using softmax to training on data, print training error and test error, return weights
def train(trainExamples, testExamples, featureExtractor):
  print "=" * 30
  print "Doing Softmax"
  alpha = 0.01      # learning rate
  weights = []
  # There are a total of five classes: 1 star, 2 star, ... 5 star. Each has its only weights. 
  # These weights are stored in a list of length 5. 
  for i in range(5):
    weights.append({})
  numIters = 20     # number of iterations through test examples
  # Doing stachastic gradient descent
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
        increment(weights[i], alpha, N[i])
  
  # calculate training error
  trainError = evaluatePredictor(trainExamples, lambda(x) : max((prob, k) for (k, prob) in enumerate(classProbs(weights, featureExtractor(x))))[1] + 1)
  print "=" * 30
  print "Result"
  print 'Iter : %s   Training error = %s' % (iter+1, trainError)
  
  # calculate test error
  testError = evaluatePredictor(testExamples, lambda(x) : max((prob, k) for (k, prob) in enumerate(classProbs(weights, featureExtractor(x))))[1] + 1)
  print "=" * 30
  print "Result"
  print 'Iter : %s   Test error = %s' % (iter+1, testError)
  return weights

      


  

