import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict
from EigenCrossValidation import EigenCrossValidation
from pprint import pprint
from EigenSpace import *
from PIL import Image

class ModelerView:
    def __init__(self, imgsUrls, crossK = 5, eigenFun = eigenWithSVD, distanceFun = distance.euclidean, kFinder = 'MIN-THRESHOLD',  useOptimalK = True):
        self.viewEigenSpace = {}
        self.viewDictionary = defaultdict(list)
        self.crossValidation = {}

        for img in imgsUrls:
            currView = img['view']
            self.viewDictionary[currView].append(img)

        for itView, itList in self.viewDictionary.items():
            self.viewEigenSpace[itView] = EigenSpace(itList, eigenFun, distanceFun, kFinder, useOptimalK)
            self.crossValidation[itView] = EigenCrossValidation(itList, crossK, eigenFun, distanceFun, kFinder)

    def projectFace(self, viewName, imgData):
        if not viewName in self.viewEigenSpace:
            print('invalid view mode')
        else:
            self.viewEigenSpace[viewName].projectFace(imgData)

    def predictFace(self, viewName, imgData, threshold = 7000):
        if not viewName in self.viewEigenSpace:
            print('invalid view mode')
        else:
            self.viewEigenSpace[viewName].predictFace(imgData, threshold)
            
    def plotAverageFace(self, viewName):
        if not viewName in self.viewEigenSpace:
            print('invalid view mode')
        else:
            self.viewEigenSpace[viewName].plotAverageFace()
            
    def plotEigenValues(self, viewName):
        if not viewName in self.viewEigenSpace:
            print('invalid view mode')
        else:
            self.viewEigenSpace[viewName].plotEigenValues()
           
            
    def plotEigenFace(self, viewName, idxFace):   
        if not viewName in self.viewEigenSpace:
            print('invalid view mode')
        else:
            self.viewEigenSpace[viewName].plotEigenFace(idxFace)

    def accuracyP(self, viewName):     
        if not viewName in self.viewEigenSpace:
            print('invalid view mode')
        else:
            print(self.crossValidation[viewName].accuracy())

    '''def f(self, img):   
        newPhi = img
        width, height = img.size
        plt.matshow(np.reshape(newPhi, (height, width)))'''
       

    def accuracy(self):
        return np.mean([cval.accuracy() for _, cval in self.crossValidation.items()])

    def plotAccuracyByGroup(self):
        names = []
        accrs = []

        for key, val in self.crossValidation.items():
            names.append(key)
            accrs.append(val.accuracy())

        plt.plot(names, accrs, 'go-', label='line 1', linewidth=2)
