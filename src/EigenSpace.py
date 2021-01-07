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

        # get image id
        index = dist.index(d_min)

        if d_min > threshold:
            tagPred = None
        else:
            tagPred = self.tags[np.argmin(dist)]

        return tagPred, d_min, index

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

    # Predict occluded part of a face 
    def predictOcclusion(self, imgFile, left, top, right, bottom, factor):
        # predict person
        occludedImage = Image.open(f'{imgFile}').convert('L')
        occludedImageData = occludedImage.getdata()
        personId, distance, index = self.predictFace(occludedImageData)
        # get indexes of the person's data
        indexes = [i for i, x in enumerate(self.tags) if x == personId]
        # get the mean eigenface of the person
        data = np.take(self.imgData, indexes, 0)
        avgFace = np.mean(np.array(data),0)
        # get image dimension 
        width, height = self.images[0].size
        # show person's mean eigenface
        #plt.matshow(np.reshape(avgFace, (height, width)), cmap='gray')
        # reconstruct face
        reconstructedFace = self.reconstructFace(index)
        # turn reconstructed face into an image
        data = np.reshape(reconstructedFace, (height, width))
        reconstructedImage = Image.fromarray(data)
        # adjust reconstructed image brightness
        #enhancer = ImageEnhance.Brightness(reconstructedImage)
        #newFace = self.replacePixels(occludedImage, reconstructedImageA, left, top, right, bottom)
        #reconstructedImageA = enhancer.enhance(factor)
        im2 = reconstructedImage.point(lambda p: p * factor)
        newFace = self.replacePixels(occludedImage, im2, left, top, right, bottom)
        # replace occlusion
        #newFace = self.replacePixels(occludedImage, reconstructedImage, left, top, right, bottom)
        # show images
        plt.matshow(np.reshape(occludedImage, (height, width)), cmap='gray')
        plt.title('Occluded Image')
        self.plotEigenFace(index)
        plt.title('Closest Eigenface')
        plt.matshow(np.reshape(reconstructedImage, (height, width)), cmap='gray')
        plt.title('Closest Eigenface Reconstructed')
        plt.matshow(np.reshape(newFace, (height, width)), cmap='gray')
        plt.title('Oclusion Prediction')

    # Reconstruct image by dotting the face’s weights with each eigenface
    def reconstructFace(self, imageId):
        # obtain the weights by dotting the mean centered data with the eigenfaces
        weights = np.dot(self.imgData, self.eigenFaces)
        # dot the face’s weights with the transpose of the eigenfaces and add the mean face back in
        reconstructed = self.avgFace + np.dot(weights[imageId, :], self.eigenFaces.T)
        return reconstructed

    # Reconstruct image using only k eigenfaces
    def reconstructFaceK(self, imageId, k):
        # obtain the weights by dotting the mean centered data with the eigenfaces
        weights = np.dot(self.imgData, self.eigenFaces)
        # dot the face’s weights with the transpose of the eigenfaces and add the mean face back in
        reconstructed  = self.avgFace + np.dot(weights[imageId, 0:k], self.eigenFaces[:, 0:k].T)
        return reconstructed

    # Replace part of an image with part of another image
    def replacePixels(self, occludedImage, reconstructedImage, left, top, right, bottom):
        replaceImg = reconstructedImage.crop((left, top, right, bottom))
        newFace = occludedImage.copy()
        newFace.paste(replaceImg, (left, top))
        return newFace

    # Create occluded image
    def createOcclusion(self, sourceImagePath, destination, left, top, right, bottom):
        face = Image.open(f'{sourceImagePath}').convert('L')
        black = Image.open(f'../images/black.png').convert('L')
        newImage = self.replacePixels(face, black, left, top, right, bottom)
        newImage.save(destination)
