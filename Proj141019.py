import csv
import random
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

LR = 0.001
EPOCHS = 50
BIAS = 0.05
NUM_PERCEPTRONS = 10

class Classifier(object):
    def start(self, fileName):
        self.inputFile = open(fileName, encoding='utf-8-sig')
        with self.inputFile as file:
            self.array2d = [[float(digit) for digit in line.split(',')] for line in file]
        self.data = np.asarray(self.array2d)
        self.data[:,1:] = self.data[:,1:]/255
        self.thresholds = np.full((len(self.data), 1), BIAS)
        self.weights = np.random.uniform(-0.05, 0.05, (10, 785))
        self.weights = np.c_[np.ones(10), self.weights]
        self.data = np.c_[self.thresholds, self.data]

    def updateInput(self, fileName):
        self.inputFile = open(fileName, encoding='utf-8-sig')
        with self.inputFile as file:
            self.array2d = [[float(digit) for digit in line.split(',')] for line in file]
        self.data = np.asarray(self.array2d)
        self.data[:,1:] = self.data[:,1:]/255
        self.thresholds = np.full((len(self.data), 1), BIAS)
        self.data = np.c_[self.thresholds, self.data]

    def startTest(self, fileName):
        self.inputFile = open(fileName, encoding='utf-8-sig')
        with self.inputFile as file:
            self.array2d = [[float(digit) for digit in line.split(',')] for line in file]
        self.data = np.asarray(self.array2d)
        self.data[:,1:] = self.data[:,1:]/255
        self.thresholds = np.full((len(self.data), 1), BIAS)
        self.weights = np.random.uniform(-0.05, 0.05, (10, 785))
        self.weights = np.c_[np.ones(10), self.weights]
        self.data = np.c_[self.thresholds, self.data]

    def analyze(self, testPhase):
        for e in range(EPOCHS):
            print("Epoch #: ", e)
            correct_response = 0
            for n in range(len(self.data)):
                yk = np.zeros(NUM_PERCEPTRONS)
                output = np.dot(self.data[n,], self.weights.transpose()) + self.weights.transpose()[0]
                yk = np.greater(output, 0).astype(int)
                predictions = np.argmax(output)
                isequal = np.equal(predictions, self.data[n,1])
                tk = np.zeros(NUM_PERCEPTRONS)
                cval = self.data[n,1].astype(int)
                tk[cval] = 1
                diff = tk - yk
                if(testPhase == True):
                    self.weights += LR * np.outer(diff, self.data[n,])
                    self.weights.transpose()[0] += LR * diff
                correct_response += len(isequal[isequal > 0])
            print("Correct: ", correct_response)

c = Classifier()
c.start("proj1/mnist_train_part1.csv")
c.analyze(True)
c.updateInput("proj1/mnist_train_part2.csv")
c.analyze(True)
c.updateInput("proj1/mnist_test.csv")
c.analyze(False)

#inputFile = open("proj1/mnist_train_part1.csv", encoding='utf-8-sig')

#with inputFile as file:
    #array2d = [[float(digit) for digit in line.split(',')] for line in file]

#data = np.asarray(array2d)

#data[:,1:] = data[:,1:]/255

#thresholds = np.full((len(data), 1), BIAS)

#weights = np.random.uniform(-0.05, 0.05, (10, 785))
#weights = np.c_[np.ones(10), weights]

#data = np.c_[thresholds, data]


#for e in range(EPOCHS):
#    print("Epoch #: ", e)
#    correct_response = 0
#    for n in range(len(data)):
#        yk = np.zeros(NUM_PERCEPTRONS)
#        output = np.dot(data[n,], weights.transpose()) + weights.transpose()[0]
#        yk = np.greater(output, 0).astype(int)
#        predictions = np.argmax(output)
#        isequal = np.equal(predictions, data[n,1])
#        tk = np.zeros(NUM_PERCEPTRONS)
#        cval = data[n,1].astype(int)
##        tk[cval] = 1
 #       diff = tk - yk
 #       weights += LR * np.outer(diff, data[n,])
 #       weights.transpose()[0] += LR * diff
 #       correct_response += len(isequal[isequal > 0])
 #   print("Correct: ", correct_response)
