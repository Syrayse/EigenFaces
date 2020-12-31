import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from PIL import Image

def eigenWithSVD(phi):
    eFaces, sigma, _ = np.linalg.svd(phi.transpose(), full_matrices=False)
    prop = sigma * sigma
    return eFaces, prop

def eigenWithMatrixFact(phi):
    """
    TODO: find eigen values using matrix factorization.
    @delagatedTo: Tiago.
    """
    return None,None

class EigenSpace:
    def __init__(self, imgUrls, eigenFun = eigenWithSVD, distanceFun = distance.euclidean, kFinder = 'MIN-THRESHOLD'):
        self.images = [Image.open(f'{im["url"]}').convert('L')  for im in imgUrls]
        self.size = len(self.images)

        if self.size > 0:
            self.imgData = [self.images[i].getdata() for i in range(self.size)]
            self.tags = [im["tag"] for im in imgUrls]
            self.avgFace = np.mean(np.array(self.imgData),0)
            self._eigenFun = eigenFun
            self.centerFace = self._centerFaces()
            self.eigenFaces, self.eigenValues = self._eigenFun(self.centerFace)
            self.optimalK, self.optimalConfidence = self._applyKFinder(kFinder)
            self._distanceFun = distanceFun
            self.baseProjections = [np.dot(self.centerFace[i], self.eigenFaces) for i in range(self.size)]

    def _applyKFinder(self, kFinder = 'MIN-THRESHOLD'):
        mapKFinderAlgorithms = {
            "MIN-THRESHOLD": self._findOptimalK_BASIC,
            "ELBOW": self._findOptimalK_ELBOW
        }

        if kFinder in mapKFinderAlgorithms:
            return mapKFinderAlgorithms[kFinder]()
        else:
            return self._findOptimalK_BASIC() # defaults to basic finder.

    def _findOptimalK_BASIC(self):
        trace = sum(self.eigenValues);
        targetConfidence, baseK = 0.8, 0

        confidence = 0
        while confidence < targetConfidence:
            confidence = confidence + self.eigenValues[baseK]/trace
            baseK += 1

        return baseK, confidence

    def _findOptimalK_ELBOW(self):
        trace = sum(self.eigenValues)

        confidence = 0
        pointDistance = []
        confidencePerPoint = []
        n = self.eigenValues.shape
        n = n[0]-1
        
        aux = 0
        p1 = np.array([0, self.eigenValues[0]])
        p2 = np.array([n, self.eigenValues[n]])
        for x in self.eigenValues:   
            p3 = np.array([aux, x])               
            pointDistance.append(abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)))
            confidence = confidence + self.eigenValues[aux]/trace
            confidencePerPoint.append(confidence)
            aux += 1

        baseK = pointDistance.index(max(pointDistance))
        confidence = confidencePerPoint[baseK]
        
        return baseK, confidence

    def _centerFaces(self):
        return self.imgData - self.avgFace


    def projectFace(self, imgData):
        newPhi = imgData - self.avgFace
        width, height = self.images[0].size
        newProj = np.dot(newPhi, self.eigenFaces)
        return np.reshape(np.dot(newProj, np.transpose(self.eigenFaces)) + self.avgFace, (height, width))

    def predictFace(self, imgData, threshold = 7000):
        # center the new face
        newPhi = imgData - self.avgFace

        # project new phi onto eigen space
        newProj = np.dot(newPhi, self.eigenFaces)

        # calculate distance between each initial entry and new projection 
        dist = [self._distanceFun(self.baseProjections[i], newProj) for i in range(self.size)]
        
        # Select closest eigen face
        d_min = np.min(dist)

        if d_min > threshold:
            tagPred = None
        else:
            tagPred = self.tags[np.argmin(dist)]

        return tagPred, d_min

    def plotEigenValues(self, pltWidth = 10, pltHeight = 10):
        plt.figure(figsize=(pltWidth,pltHeight))
        t = np.arange(0, self.size, 1)
        plt.plot(t, self.eigenValues, 'x')
        plt.plot(self.optimalK, self.eigenValues[self.optimalK], 'o')
        plt.show()

    def plotAverageFace(self):
        width, height = self.images[0].size
        plt.matshow(np.reshape(self.avgFace, (height, width)), cmap='gray')

    def plotEigenFace(self, idxFace):
        if idxFace < self.size:
            eigenData = self.eigenFaces.T[idxFace]
            width, height = self.images[idxFace].size
            plt.matshow(np.reshape(eigenData, (height, width)), cmap='gray')
