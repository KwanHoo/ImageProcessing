import numpy as np
import cv2
import time


# library add
# import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from my_library.padding import my_padding
# from my_library.filtering import my_filtering
# from my_library.get_filter import my_get_Gaussian2D_mask


# 제로패딩
def my_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:h + p_h, p_w:w + p_w] = src
    return pad_img


def my_filtering(src, filter):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape
    src_pad = my_padding(src, (f_h // 2, f_w // 2))
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row + f_h, col:col + f_w] * filter)
            # val = np.clip(val, 0, 255)
            dst[row, col] = val

    # dst = (dst+0.5).astype(np.uint8)
    return dst


def my_get_Gaussian2D_mask(msize, sigma=1):
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
    gaus2D = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D


#############################################################################
def my_normalize(src):
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)


def add_gaus_noise(src, mean=0, sigma=0.1):
    # src : 0 ~ 255, dst : 0 ~ 1
    dst = src / 255
    h, w = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w))
    dst += noise
    return my_normalize(dst)


def my_bilateral(src, msize, sigma, sigma_r, pos_x, pos_y, pad_type='zero'):
    ####################################################################################################
    # TODO                                                                                             #
    # my_bilateral 완성                                                                                 #
    # mask를 만들 때 4중 for문으로 구현 시 감점(if문 fsize*fsize개를 사용해서 구현해도 감점) 실습영상 설명 참고      #
    ####################################################################################################
    pad_size = int(msize // 2)
    pad_img = my_padding(src, (pad_size, pad_size))
    (h, w) = src.shape
    print(src.shape)
    dst = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            a = 0
            b = 0
            for k in range(i, i + msize):
                for l in range(j, j + msize):
                    f = np.exp(-(((i + pad_size - k) ** 2 + (j + pad_size - l) ** 2) / (2 * sigma ** 2) + (
                                src[i, j] - pad_img[k, l]) ** 2 / (2 * sigma_r ** 2)))
                    a = a + pad_img[k, l] * f
                    b = b + f
            dst[i, j] = a / b
    dst = (dst + 0.5).astype(np.uint8)
    return dst


if __name__ == '__main__':
    start = time.time()
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    np.random.seed(seed=100)

    pos_y = 0
    pos_x = 0
    src_line = src.copy()
    # src_line[-4:5, -4:5] = 255
    src_line[pos_y - 4:pos_y + 5, pos_x - 4:pos_x + 5] = 255
    # src_line[-2:3, -2:3] = src[-2:3, -2:3]
    src_line[pos_y - 2:pos_y + 3, pos_x - 2:pos_x + 3] = src[pos_y - 2:pos_y + 3, pos_x - 2:pos_x + 3]
    src_noise = add_gaus_noise(src, mean=0, sigma=0.1)
    src_noise = src_noise / 255

    ######################################################
    # TODO                                               #
    # my_bilateral, gaussian mask 채우기                  #
    ######################################################
    dst = my_bilateral(src_noise, 5, 30, 40, pos_x, pos_y)  # ?? #??
    dst = my_normalize(dst)

    gaus2D = my_get_Gaussian2D_mask(5, sigma=1)  # ??
    dst_gaus2D = my_filtering(src_noise, gaus2D)
    dst_gaus2D = my_normalize(dst_gaus2D)

    cv2.imshow('original', src_line)
    cv2.imshow('gaus noise', src_noise)
    cv2.imshow('my gaussian', dst_gaus2D)
    cv2.imshow('my bilateral', dst)
    tital_time = time.time() - start
    print('\ntime : ', tital_time)
    if tital_time > 25:
        print('시간이 너무 많이 걸려요!! 알고리즘 수정을 해보아요!!')
    cv2.waitKey()
    cv2.destroyAllWindows()

