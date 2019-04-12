import csv
import random
import sys
import numpy as np

#For debugging purposes, allows unabridged printing of arrays.
np.set_printoptions(threshold=sys.maxsize)

#Constants and hyperparameters:
LR = 0.001
EPOCHS = 50
BIAS = 0.05
NUM_PERCEPTRONS = 10

class Classifier(object):
    #Starts up data reader and creates weights array.
    #Adds bias (constant "BIAS") and bias weight (always 1) to arrays.
    def start(self, fileName):
        #utf-8-sig avoids garbage "start of file" symbols being read:
        self.inputFile = open(fileName, encoding='utf-8-sig')

        #read in data:
        with self.inputFile as file:
            self.array2d = [[float(digit) for digit in line.split(',')] for line in file]

        #convert to NumPy array and normalize (divide by 255):
        self.data = np.asarray(self.array2d)
        self.data[:,1:] = self.data[:,1:]/255

        #create thresholds array and weights, being sure to add bias.
        self.thresholds = np.full((len(self.data), 1), BIAS)
        self.weights = np.random.uniform(-0.05, 0.05, (10, 785))
        self.weights = np.c_[np.ones(10), self.weights]
        self.data = np.c_[self.thresholds, self.data]
        self.tracker = np.zeros((10,2))  #keeps track of each digit for confusion matrix

    #Reads in a new data set. Training data sets were split in two to avoid stack overflow
    #errors. Alternative solution could have been to use 16-bit floating point numbers
    #instead of 32-bit, but this works anyway.
    #Note: this function does NOT update weights. Can also be used for testing data.
    def updateInput(self, fileName):
        self.inputFile = open(fileName, encoding='utf-8-sig')
        with self.inputFile as file:
            self.array2d = [[float(digit) for digit in line.split(',')] for line in file]
        self.data = np.asarray(self.array2d)
        self.data[:,1:] = self.data[:,1:]/255
        self.thresholds = np.full((len(self.data), 1), BIAS)
        self.data = np.c_[self.thresholds, self.data]

    #Analyze data set. Loops through all data entries/epochs and does the following:
        #1) computes dot product
        #2) determines whether dot product plus bias value is above 0 (yk)
        #3) makes prediction and compares with correct value
        #4) if "testPhase" argument (a boolean) is TRUE then update weights
            #if not (testing phase) then do NOT update weights
        #5) compute accuracy for this epoch
    def analyze(self, testPhase):
        num_entries = len(self.data)
        for e in range(EPOCHS):
            print("Epoch #: ", e)
            correct_response = 0 #number of correct responses
            for n in range(len(self.data)):
                yk = np.zeros(NUM_PERCEPTRONS)

                #compute dot product
                output = np.dot(self.data[n,], self.weights.transpose()) + self.weights.transpose()[0]

                #check if dot product is above zero
                yk = np.greater(output, 0).astype(int)

                #make prediction
                predictions = np.argmax(output)

                #compare prediction with actual
                isequal = np.equal(predictions, self.data[n,1])

                #create "correctness" array represented by tk
                tk = np.zeros(NUM_PERCEPTRONS)
                cval = self.data[n,1].astype(int)
                self.tracker[cval, 1] += 1
                tk[cval] = 1

                #calculate difference (tk - yk)
                diff = tk - yk

                #conditionally, update weights
                if(testPhase == True):
                    self.weights += LR * np.outer(diff, self.data[n,])
                    self.weights.transpose()[0] += LR * diff

                #total up number of correct responses
                correct_response += len(isequal[isequal > 0])
                self.tracker[isequal > 0] += 1
            #print("Correct: ", correct_response)
            if(testPhase == False):
                print("Done with test dataset. Final accuracy: {0:.2f}".format(100*correct_response/num_entries))
                break
            print("%: ", float("{0:.2f}".format(100*correct_response/num_entries)))

    def printTracker(self):
        print(self.tracker)

#Execution begins here.
c = Classifier()
c.start("proj1/mnist_train_part1.csv")
c.analyze(True)
c.updateInput("proj1/mnist_train_part2.csv")
c.analyze(True)
c.updateInput("proj1/mnist_test.csv")
c.analyze(False)
c.printTracker()
