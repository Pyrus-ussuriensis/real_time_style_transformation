# 测试OpenCV的使用
import cv2
img = cv2.imread('data/starrynight.jpg')
cv2.imshow('test', img)
print(cv2.getBuildInformation())
cv2.waitKey(0)
