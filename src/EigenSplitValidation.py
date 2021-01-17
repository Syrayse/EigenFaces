from scipy.spatial import distance
import random

from EigenSpace import *

class EigenSplitValidation:
    def __init__(self, imgsUrls, percentage, eigenFun = eigenWithSVD, distanceFun = distance.euclidean, kFinder = 'MIN-THRESHOLD',  useOptimalK = True):
        self._percentage = percentage
        self._urls = imgsUrls.copy()
        random.shuffle(self._urls)
        testData, trainData = self._partition(percentage)
        eSpace = EigenSpace(trainData, eigenFun, distanceFun, kFinder, useOptimalK)
        self._accuracies = []
        self._errorCases = []
        matchingElems = 0;

        for j in range(0, len(testData)):
            predTag, _, _ = eSpace.predictFace(testData[j]['img'])

            if testData[j]['tag'] == predTag:
                matchingElems += 1
            else:
                self._errorCases.append(testData[j])

        self._accuracy = matchingElems / len(testData)

        
    def _partition(self, percentage):
        numElems = int(round(percentage * len(self._urls)))
        return self._urls[numElems:], self._urls[:numElems]

    def accuracy(self):
        return self._accuracy

    def errorCases(self):
        return self._errorCases.copy()