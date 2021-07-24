import cv2
import numpy as np

# 간단한 실습 1
# src1 = np.zeros((200, 200))
# src2 = np.ones((200, 200))
#
# src3 = np.zeros((200, 200), dtype=np.uint8)
# src4 = np.ones((200, 200), dtype=np.uint8)
# src5 = np.full((200, 200), 255, dtype=np.uint8)
#
# cv2.imshow('src1', src1)
# cv2.imshow('src2', src2)
# cv2.imshow('src3', src3)
# cv2.imshow('src4', src4)
# cv2.imshow('src5', src5)
#
# print('<float>')
# print(src1.shape, src2.shape)
# print(src1[0,0], src2[0,0])
# print('<uin8>')
# print(src3.shape, src4.shape, src5.shape)
# print(src3[0,0], src4[0,0], src5[0,0])
#
# cv2.waitKey()
# cv2.destoryALLwindows()

#-----------------------------
#간단한 실습2

# src1 = np.zeros((200,200))
# src2 = np.zeros((200,200), dtype=np.uint8)
#
# src1[:, 100:200] = 1.
# src2[:, 100:200] = 255
#
# cv2.imshow('src1', src1)
# cv2.imshow('src2', src2)
#
# print(src1[50,50], src1[150,150])
# print(src2[50,50], src2[150,150])
#
# print(src1[100, 95:105])
# print(src2[100, 95:105])
#
#
# cv2.waitKey()
# cv2.destroyAllWindows()

#간단한 실습3
# src = np.zeros((300,300,3), dtype = np.uint8)
# src[0,0] = [1,2,3]
# src[0,1] = [4,5,6]
# src[1,0] = [7,8,9]
#
# print(src.shape)
# print(src[0,0,0], src[0,0,1], src[0,0,2])
# print(src[0,0])
# print(src[0])
# print(src)

#간단한 실습4

# src = np.zeros((300,300,3), dtype = np.uint8)
# # 1. b=255  g,r=0
# src[0:100, 0:100, 0] = 255
# # 2. g=255, b,r=0
# src[0:100, 100:200, 1] = 255
# # 3. r=255, b,g=0
# src[0:100, 200:300, 2] = 255
#
# # 4. b+r
# src[100:200, 0:100, 0] = 255
# src[100:200, 0:100, 2] = 255
#
# # 5. b+g
# src[100:200, 100:200, 0] = 255
# src[100:200, 100:200, 1] = 255
#
# # 6. g+r
# src[100:200, 200:300, 1] = 255
# src[100:200, 200:300, 2] = 255
#
# # 7. b+g+r
# src[200:, :100, 0] =255
# src[200:, :100, 1] =255
# src[200:300, 0:100, 2] =255
#
#
# # 8.b/2 +g/2 +r/2
# src[200:, 100:200, 0] = 128
# src[200:, 100:200, 1] = 128
# src[200:300, 100:200, 2] = 128
#
#
# cv2.imshow('src', src)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 간단한 실습 5
# src = cv2.imread('Penguins.png')
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#
# print('[color shape] : {0}' .format(src.shape))
# print('[gary shape] : {0}' .format(src.shape))
#
# cv2.imshow('color',src)
# cv2.imshow('gary',gray)
# cv2.imshow('slice', src[50:230, 50:230, :])
#
# cv2.waitKey()
# cv2.destroyAllWindows()

#간단한 실습6

# src = cv2.imread('Lena.png')
# rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow('orginal', src)
# cv2.imshow('RGB', rgb)
# cv2.imshow('gray', gray)
#
# print('[BGR] {0}'. format(src[0,0]))
# print('[RGB] {0}' .format(rgb[0,0]))
# print('[gary] {0}' .format(gray[0,0]))

#간단한 실습 7

# src = cv2.imread('Lena.png')
# (h, w, c) = src.shape
# yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
# my_y = np.zeros((h, w))
# my_y = (src[:, :, 0] * 0.114) + (src[:, :, 1] *0.587) + (src[: , : ,2] *0.299)
# #my_y = np.round(my_y).astype(np.uint8) #반올림 np.round() 사용해도 상관없음
# my_y = (my_y +0.5).astype(np.uint8) #반올림 np.round() 사용해도 무관
#
# cv2.imshow('original', src)
# cv2.imshow('cvtColor', yuv[:, : , 0])
# cv2.imshow('my_y', my_y)
#
# print(yuv[0:5, 0:5, 0])
# print(my_y[0:5, 0:5])

# 간단한 실습8

src1 = np.zeros((300, 200))
src2 = np.zeros((300, 200), dtype= np.uint8)

src1[:100] = 1.0
src1[100:200] =0.5
src1[200:] = 0.0

src2[:100] = 255
src2[100:200] = 128
src2[200:] = 0

cv2.imshow('src1', src1)
cv2.imshow('src2', src2)


cv2.waitKey()
cv2.destroyAllWindows()