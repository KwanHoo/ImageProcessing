import numpy as np


# 가오시안 필터
#numpy 이용하여 한번에 만들기
Gau_ex = np.zeros((5,5))
print(Gau_ex)
y, x = np.mgrid[-2:3, -2:3] #-2 ~ 2까지 //x,y정의
print(y)
print('x')
print(x)


#총합 1
x = np.array([[1,2],[3,4]])
x = x/ np.sum(x)
print(x)

x = np.full((5, 1),2)
print(x)

print('---')
msize = 5
y, x_e = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
print(x_e)
print('+++')
x = np.full((1, msize), x_e[0])
print(x)


import cv2
import numpy as np

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))
    print('h_dst')
    print(h_dst)
    print(w_dst)
    print(h,w)


    # bilinear interpolation 적용
    # for row in range(h_dst):
    #     for col in range(w_dst):
    #         # 참고로 꼭 한줄로 구현해야 하는건 아닙니다 여러줄로 하셔도 상관없습니다.(저도 엄청길게 구현했습니다.)
    #         dst =
    # return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 1/2
    # #이미지 크기 1/2배로 변경
    my_dst_mini = my_bilinear(src, scale)
    # my_dst_mini = my_dst_mini.astype(np.uint8)
    #
    # #이미지 크기 2배로 변경(Lena.png 이미지의 shape는 (512, 512))
    # my_dst = my_bilinear(my_dst_mini, 1/scale)
    # my_dst = my_dst.astype(np.uint8)
    #
    # cv2.imshow('original', src)
    # cv2.imshow('my bilinear mini', my_dst_mini)
    # cv2.imshow('my bilinear', my_dst)
    #
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    print()

