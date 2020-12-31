import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from PIL import Image

def getImageByUrl(urlName):
    return Image.open(f'{urlName}').convert('L').getdata()

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
    def __init__(self, imgUrls, eigenFun = eigenWithSVD, distanceFun = distance.euclidean, kFinder = 'MIN-THRESHOLD', useOptimalK = True):
        self.images = [im['img']  for im in imgUrls]
        self.size = len(self.images)

        if self.size > 0:
            self.imgData = [self.images[i] for i in range(self.size)]
            self.tags = [im["tag"] for im in imgUrls]
            self.avgFace = np.mean(np.array(self.imgData),0)
            self._eigenFun = eigenFun
            self.centerFace = self._centerFaces()
            self.eigenFaces, self.eigenValues = self._eigenFun(self.centerFace)
            self._useOptimalK = useOptimalK

            if useOptimalK:
                self.optimalK, self.optimalConfidence = self._applyKFinder(kFinder)
                self.optimalEigenFaces = self.eigenFaces[:,0:self.optimalK]
            else:                
                self.optimalEigenFaces = self.eigenFaces

            self._distanceFun = distanceFun
            self.baseProjections = [np.dot(self.centerFace[i], self.optimalEigenFaces) for i in range(self.size)]

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
        """
        TODO: find optimal K using optimized elbow-finding algorithm      
        @delagatedTo: Benjamim.  
        """
        return None,None

    def _centerFaces(self):
        return self.imgData - self.avgFace


    def projectFace(self, imgData):
        newPhi = imgData - self.avgFace
        width, height = self.images[0].size
        newProj = np.dot(newPhi, self.optimalEigenFaces)
        return np.reshape(np.dot(newProj, np.transpose(self.optimalEigenFaces)) + self.avgFace, (height, width))

    def predictFace(self, imgData, threshold = 7000):
        # center the new face
        newPhi = imgData - self.avgFace

        # project new phi onto eigen space
        newProj = np.dot(newPhi, self.optimalEigenFaces)

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

        if self._useOptimalK:
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
