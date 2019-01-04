import numpy as np
import matplotlib.pyplot as plt
import cv2
import os,time
# img = np.zeros((512,512,3),np.uint8)
# img = cv2.line(img,(70,70),(400,400),(0,0,255),4)
# img = cv2.ellipse(img,(256,128),(70,70),0,120,420,(0,0,255),40)
# img = cv2.ellipse(img,(160,320),(70,70),0,0,300,(0,255,0),40)
# img = cv2.ellipse(img,(352,320),(70,70),0,300,600,(255,0,0),40)
# font = cv2.FONT_HERSHEY_SIMPLEX
# img = cv2.putText(img,'OpenCV',(70,480),font,3,(255,255,255),4,cv2.LINE_AA)
# cv2.imshow('Image',img)
img1 = cv2.imread('/home/sourav/Pictures/pic1.jpg',1)
img2 = cv2.imread('/home/sourav/Pictures/pic2.jpg',1)
img4 = cv2.imread('/home/sourav/Pictures/pic3.png',1)
# img4 = cv2.resize(img4,(300,168),interpolation = cv2.INTER_AREA)
# # cv2.imshow('Circle',img4)
ret,thr = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
cv2.imshow('Pic',thr)
cv2.imshow('Original',img1)
colors = ('b','g','r')
for i,col in enumerate(colors):
    histr = cv2.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
# img3,cont,hier = cv2.findContours(thr,1,2)
# cnt = cont[0]
# print (cnt)
# hull = cv2.convexHull(cnt)
# x,y,w,h = cv2.boundingRect(cnt)
# area = cv2.contourArea(cnt)
# print (area)
# aspect_ratio = float(w)/h
# print (aspect_ratio)
# img = cv2.rectangle(img4,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.imshow('Rect',img)
# print (hull)
# con = cont[0]
# M = cv2.moments(con)
# print (M)
# ret2,thr2 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# blur = cv2.GaussianBlur(img1,(5,5),0)
# ret3,thr3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print (hist)
# cv2.imshow('Otsu',thr2)
# cv2.imshow('Gaussian',thr3)
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret,frame = cap.read()
#     hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     lower_blue = np.array([110,50,50])
#     upper_blue = np.array([130,255,255])
#     mask = cv2.inRange(hsv,lower_blue,upper_blue)
#     res = cv2.bitwise_and(frame,frame,mask = mask)
#     cv2.imshow('Frame',frame)
#     cv2.imshow('Res',res)
#     cv2.imshow('Mask',mask)
# plt.hist(thr2.ravel(),256)
# plt.show()
# kernel = np.ones((5,5),np.float32)/25
# kernel1 = np.ones((3,3),np.float32)/9
# img = cv2.filter2D(img1,-1,kernel)
# img3 = cv2.medianBlur(img1,5)
# img4 = cv2.Canny(img1,150,200)
# cv2.imshow('Original',img1)
# cv2.imshow('Filter',img)
# cv2.imshow('Median',img3)
# cv2.imshow('Canny',img4)
k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()

    # ring = img[287:345,578:604]
# print (ring)
# img[100:158,100:126] = ring
# img1 = cv2.resize(img1,(300,168),interpolation = cv2.INTER_AREA)
# cv2.imshow('Image1',img1)
# cv2.imshow('Image2',img2)
# x = np.linspace(0,1,51)
# for i in x:
#     img = cv2.addWeighted(img1,i,img2,1-i,0)
#     cv2.imshow('Image',img)
#     time.sleep(0.1)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
# print (img2.shape)
# img2 = img[:,:,::-1]
# plt.subplot(121);plt.imshow(img)
# plt.subplot(122);plt.imshow(img2)
# plt.imshow(img,cmap = 'gray',interpolation = 'bicubic')
# plt.xticks([]),plt.yticks([])
# plt.show()
# elif k == ord('s'):
#     cv2.imwrite('/home/sourav/Pictures/pic1.jpg',img1)
#     cv2.destroyAllWindows()
# cap = cv2.VideoCapture('video1.mkv')
# while cap.isOpened() :
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('Video',frame)
#     k = cv2.waitKey(40) & 0xFF
#     if k is ord('q'):
#         break
# cv2.destroyAllWindows()