import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from PIL import Image
import random

from EigenSpace import *

class EigenCrossValidation:
    def __init__(self, imgsUrls, K, eigenFun = eigenWithSVD, distanceFun = distance.euclidean, kFinder = 'MIN-THRESHOLD'):
        """
        should create create K folds and create K EigenSpaces,
        one for each of the K splits.
        """
        self._K = K
        self._urls = imgsUrls.copy()
        random.shuffle(self._urls)
        self._splitSize = N = int(round(len(self._urls)/K))
        self._splits = [ self._urls[i:i+N] for i in range(0, len(self._urls), N) ]
        self._accuracies = []
        self._errorCases = []

        # for each split, create an eigen space using train data
        for i in range(0, K):
            testData, trainData = self._trainTestData(i)
            eSpace = EigenSpace(trainData, eigenFun, distanceFun, kFinder)
            matchingElems = 0;

            for j in range(0, len(testData)):
                testImage = Image.open(f'{testData[j]["url"]}').convert('L')
                predTag, _ = eSpace.predictFace(testImage.getdata())

                if testData[j]['tag'] == predTag:
                    matchingElems += 1
                else:
                    self._errorCases.append(testData[j])

            self._accuracies.append(matchingElems / len(testData))

        
    def _trainTestData(self, idx):
        testData = self._splits[idx]
        trainData = []

        for i in range(0, self._K):
            if i != idx:
                trainData.extend(self._splits[i])

        return testData, trainData

    def accuracy(self):
        return sum(self._accuracies) / self._K

    def errorCases(self):
        return self._errorCases.copy()