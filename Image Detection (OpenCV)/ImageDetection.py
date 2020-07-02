# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:03:00 2017

@author: Joel Mauriths Kambey
"""
import cv2
import numpy as np

#Load carpark image
img_rgb = cv2.imread('Testimage4.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#Load parking slots templates
template1 = cv2.imread('PT1_3.jpg',0)
w1, h1 = template1.shape[::-1]
template2 = cv2.imread('PT2_3.jpg',0)
w2, h2 = template2.shape[::-1]
template3 = cv2.imread('PT3_3.jpg',0)
w3, h3 = template3.shape[::-1]
template4 = cv2.imread('PT4_3.jpg',0)
w4, h4 = template4.shape[::-1]

#Matching carpark image and templates
output_ls = []
res = cv2.matchTemplate(img_gray,template1,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
statusP1 = 'Occupied'
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w1, pt[1] + h1), (0, 0, 255), 2)
    statusP1 = 'Empty'
output_ls.append(statusP1)
    
res = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
statusP2 = 'Occupied'
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w2, pt[1] + h2), (0, 0, 255), 2)
    statusP2 = 'Empty'
output_ls.append(statusP2)
    
res = cv2.matchTemplate(img_gray,template3,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
statusP3 = 'Occupied'
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w3, pt[1] + h3), (0, 0, 255), 2)
    statusP3 = 'Empty'
output_ls.append(statusP3)

res = cv2.matchTemplate(img_gray,template4,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
statusP4 = 'Occupied'
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w4, pt[1] + h4), (0, 0, 255), 2)
    statusP4 = 'Empty'
output_ls.append(statusP4)
    
print output_ls
cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()