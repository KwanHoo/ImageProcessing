import numpy as np
import cv2
import time

# library add
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from.my_padding import my_padding

# 제로패딩
def my_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src
    return pad_img



def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    # y, x = np.mgrid[-2:3, -2:3] #-2 ~ 2까지 //x,y정의
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    '''
    y, x = np.mgrid[-1:2, -1:2]
    y = [[-1,-1,-1],
         [ 0, 0, 0],
         [ 1, 1, 1]]
    x = [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    '''

    # 2차 gaussian mask 생성
    gaus2D = 1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D


def my_get_Gaussian1D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 1D gaussian filter 만들기
    #########################################
    y, x_e = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    x = np.full((1,msize), x_e[0]) #msize = 5// [[-2 -1 0 1 2]]


    '''
    x = np.full((1, 3), [-1, 0, 1])
    x = [[ -1, 0, 1]]

    x = np.array([[-1, 0, 1]])
    x = [[ -1, 0, 1]]
    '''

    gaus1D = 1 / ((2 * np.pi)**0.5 * sigma) * np.exp(-((x ** 2) / (2 * sigma ** 2)))

    # mask의 총 합 = 1
    gaus1D /= np.sum(gaus1D)
    return gaus1D


def my_filtering(src, mask):
    (h, w) = src.shape
    # mask의 크기
    (m_h, m_w) = mask.shape
    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, (m_h // 2, m_w // 2))

    print('<mask>')
    print(mask)

    # 시간을 측정할 때 만 이 코드를 사용하고 시간측정 안하고 filtering을 할 때에는
    # 4중 for문으로 할 경우 시간이 많이 걸리기 때문에 2중 for문으로 사용하기.
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            sum = 0
            for m_row in range(m_h):
                for m_col in range(m_w):
                    sum += pad_img[row + m_row, col + m_col] * mask[m_row, m_col]
            dst[row, col] = sum

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    mask_size = 5
    gaus2D = my_get_Gaussian2D_mask(mask_size, sigma=1)
    gaus1D = my_get_Gaussian1D_mask(mask_size, sigma=1)

    print('mask size : ', mask_size)
    print('1D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus1D = my_filtering(src, gaus1D.T)
    dst_gaus1D = my_filtering(dst_gaus1D, gaus1D)
    end = time.perf_counter()  # 시간 측정 끝
    print('1D time : ', end - start)

    print('2D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus2D = my_filtering(src, gaus2D)
    end = time.perf_counter()  # 시간 측정 끝
    print('2D time : ', end - start)

    dst_gaus1D = np.clip(dst_gaus1D + 0.5, 0, 255)
    dst_gaus1D = dst_gaus1D.astype(np.uint8)
    dst_gaus2D = np.clip(dst_gaus2D + 0.5, 0, 255)
    dst_gaus2D = dst_gaus2D.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('1D gaussian img', dst_gaus1D)
    cv2.imshow('2D gaussian img', dst_gaus2D)
    cv2.waitKey()
    cv2.destroyAllWindows()