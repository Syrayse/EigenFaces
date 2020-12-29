from scipy.spatial import distance
from PIL import Image
import random

from EigenSpace import *

class EigenSplitValidation:
    def __init__(self, imgsUrls, percentage, eigenFun = eigenWithSVD, distanceFun = distance.euclidean, kFinder = 'MIN-THRESHOLD'):
        self._percentage = percentage
        self._urls = imgsUrls.copy()
        random.shuffle(self._urls)
        testData, trainData = self._partition(percentage)
        eSpace = EigenSpace(trainData, eigenFun, distanceFun, kFinder)
        self._accuracies = []
        self._errorCases = []
        matchingElems = 0;

        for j in range(0, len(testData)):
            testImage = Image.open(f'{testData[j]["url"]}').convert('L')

            predTag, _ = eSpace.predictFace(testImage.getdata())

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