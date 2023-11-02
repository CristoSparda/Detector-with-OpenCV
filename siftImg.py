import numpy as np
import cv2 as cv
img = cv.imread('game.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)

#second mettod 
kp2, des = sift.detectAndCompute(gray,None)
img2=cv.drawKeypoints(gray,kp2,img)
cv.imshow('result',img)
cv.imshow('result2',img2)

cv.waitKey(0)