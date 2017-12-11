# Process raw data and save them into pickle file.
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
from scipy import misc
import scipy.io
from skimage import io
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation
from constants import *



img_size = INPUT_SIZE
salmap_size = INPUT_SIZE


listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToMaps, '*'))[0:1000]]

#listTestImages = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*test*'))]
#print(listImgFiles)
#for currFile in tqdm(listImgFiles):
#currFile="COCO_train2014_000000287422"
#currFile="COCO_train2014_000000210015"
#print('=========================================')
#print(pathToMaps, currFile + '.mat')
#tt = dataRepresentation.Target(os.path.join(pathToImages, currFile + '.jpg'),
#                           os.path.join(pathToMaps, currFile + '.png'),
#                           os.path.join(pathToFixationMaps, currFile + '.mat'),
#                           dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
#                           dataRepresentation.LoadState.loaded, dataRepresentation.InputType.saliencyMapMatlab,
#                           dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty)

# if tt.image.getImage().shape[:2] != (480, 640):
#    print 'Error:', currFile
#print(currFile)
#imageResized = cv2.cvtColor(cv2.resize(tt.image.getImage(), img_size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
#saliencyResized = cv2.resize(tt.saliency.getImage(), salmap_size, interpolation=cv2.INTER_AREA)

#cv2.imwrite(os.path.join(pathOutputImages, currFile + '.png'), imageResized)
#cv2.imwrite(os.path.join(pathOutputMaps, currFile + '.png'), saliencyResized)



listFilesTrain = [k for k in listImgFiles if 'train' in k]
trainData = []
 #print(listFilesTrain)
for currFile in tqdm(listFilesTrain):
    print(currFile)
    print(trainData)
    trainData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                               os.path.join(pathOutputMaps, currFile + '.png'),
                                               os.path.join(pathToFixationMaps, currFile + '.mat'),
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.fixationMapMatlab))
    print('=========================')

with open(os.path.join(pathToPickle, 'trainData.pickle'), 'wb') as f:
    pickle.dump(trainData, f)



listFilesValidation = [k for k in listImgFiles if 'val' in k]
validationData = []
for currFile in tqdm(listFilesValidation):
    print(currFile)
    validationData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                                    os.path.join(pathOutputMaps, currFile + '.png'),
                                                    os.path.join(pathToFixationMaps, currFile + '.mat'),
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.fixationMapMatlab))

with open(os.path.join(pathToPickle, 'validationData.pickle'), 'wb') as f:
    pickle.dump(validationData, f)






