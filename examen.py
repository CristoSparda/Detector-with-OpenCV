import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
 
path = 'reals'
# Initiate ORB detector
orb = cv.ORB_create(nfeatures=1000)

images = []
classNames = []

myList = os.listdir(path)
#make a split to remove the image extentions
for cl in myList :
    imgCur = cv.imread(f'{path}/{cl}')
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

#funcion to dtect image from the training files
def findDes(images):
    desList = []
    for img in images :
        #img = cv.resize(img, (int(200), int(350)))
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

#Function to detect images from the camera
def findID(img, desList, thres=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv.BFMatcher()
    matchList = []
    finalValue = -1
    try :
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    #print(matchList)
    if len(matchList) != 0:
        if max(matchList) > thres:
            finalValue = matchList.index(max(matchList))
    return finalValue


desList = findDes(images)

cam = cv.VideoCapture(0)

while True:

    success, camera = cam.read()

    id = findID(camera, desList)

    if id != -1:
        cv.putText(camera, classNames[id],(50,50), cv.FONT_HERSHEY_COMPLEX, 1,(255,0,0),3)

    cv.imshow('camara', camera)
    cv.waitKey(1)


################################### ORB solo con imagenes #################################
# img = cv.imread('game.jpg')
# #source from the original one
# img2 = cv.imread('reals/Zelda Tears of the Kingdom.jpg')

# # Define el tamaño máximo de la ventana
# ancho_maximo = 300
# alto_maximo = 200

# #Redimension de las imagenes
# img2 = cv.resize(img2, (int(200), int(350)))
# img = cv.resize(img, (int(200), int(350)))


# # find the keypoints with ORB
# kp = orb.detect(img,None)
# kp2 = orb.detect(img2,None)

# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)
# kp2, des2 = orb.compute(img2, kp2)

# #matching the desc results
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des,des2,k=2)

# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# print(len(good))

# img3 = cv.drawMatchesKnn(img,kp,img2,kp2,good,None,flags=2)

# # draw only keypoints location,not size and orientation
# imgR = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# imgR2 = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

# #Show the results
# cv.namedWindow('Foto', cv.WINDOW_NORMAL)
# cv.imshow("Foto", imgR )
# cv.imshow("Imagen referencia", imgR2)
# cv.imshow('Matches', img3)

# cv.waitKey(0)
# cv.destroyAllWindows()






























