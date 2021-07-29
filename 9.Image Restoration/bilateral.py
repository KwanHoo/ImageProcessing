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
    src_pad = my_padding(src, (f_h//2, f_w//2))
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row+f_h, col:col+f_w] * filter)
            #val = np.clip(val, 0, 255)
            dst[row, col] = val

    #dst = (dst+0.5).astype(np.uint8)
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
    gaus2D = 1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D

def my_normalize(src):
    dst = src.copy()  # 0~255
    dst *= 255
    dst = np.clip(dst, 0, 255)   # 오버플로우문제해결
    return dst.astype(np.uint8)

def add_gaus_noise(src, mean=0, sigma=0.1):
    #src : 0 ~ 255, dst : 0 ~ 1
    dst = src/255    #0~1
    h, w = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w))
    dst += noise
    return my_normalize(dst)


#src 노이즈 이미지 /마스크크기 /
def my_bilateral(src, msize, sigma, sigma_r, pos_x, pos_y):
    ####################################################################################################
    # TODO                                                                                             #
    # my_bilateral 완성                                                                                 #
    # mask를 만들 때 4중 for문으로 구현 시 감점(if문 fsize*fsize개를 사용해서 구현해도 감점) 실습영상 설명 참고      #
    ####################################################################################################
    (h, w) = src.shape
    pad_size = int(msize//2)
    img_pad = my_padding(src, (pad_size, pad_size))
    print(src.shape)
    dst = np.zeros((h, w))
####
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
    # gaus2D = 1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))
    # mask = 1 /(2 * np.pi * sigma**2)* np.exp(-((i-k)**2/2*sigma**2)-((j-l)**2/2*sigma**2))*np.exp()
    # mask의 총 합 = 1
    # gaus2D /= np.sum(gaus2D)
#####

    # for i in range(h):
    #     for j in range(w):
    #         a = 0
    #         b = 0
    #         for k in range(i, i + msize):
    #             for l in range(j, j + msize):
    #                 f = np.exp(-(((i + pad_size - k) ** 2 + (j + pad_size - l) ** 2) / (2 * sigma ** 2) + (src[i, j] - pad_img[k, l]) ** 2 / (2 * sigma_r ** 2)))
    #                 a = a + pad_img[k, l] * f
    #                 b = b + f
    #         dst[i, j] = a / b
    # dst = (dst+0.5).astype(np.uint8)

    mask = np.zeros((msize, msize))
    for i in range(h):
        print('\r%d / %d ...' %(i,h), end="")
        for j in range(w):
            k = y + i
            l = x + j
            mask = np.exp(-(((i - k)** 2) / (2 * sigma**2)) - (((j-l)**2) / (2 * sigma**2))) * \
                   np.exp( - (((img_pad[i+pad_size, j+pad_size] - img_pad[k+pad_size, l+pad_size])**2)/(2*sigma_r**2)))
            mask = mask/mask.sum()

            if i==pos_y and j == pos_x:
                print()
                print(mask.round(4))
                mask_visual = cv2.resize(mask, (200, 200), interpolation = cv2.INTER_NEAREST)
                mask_visual = mask_visual - mask_visual.min()
                mask_visual = (mask_visual/mask_visual.max() * 255).astype(np.uint8)
                cv2.imshow('mask', mask_visual)

                img = img_pad[i:i+5, j:j+5]
                img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_NEAREST)
                img = my_normalize(img)
                cv2.imshow('img', img)

            dst[i, j] = np.sum(img_pad[i:i + msize, j:j + msize] * mask)

    return dst



if __name__ == '__main__':
    start = time.time()
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    np.random.seed(seed=100) # 랜덤시드 고정

    pos_y = 0
    pos_x = 0
    src_line = src.copy()
    src_line[pos_y-4:pos_y+5, pos_x-4:pos_x+5] = 255
    src_line[pos_y-2:pos_y+3, pos_x-2:pos_x+3] = src[pos_y-2:pos_y+3, pos_x-2:pos_x+3]
    src_noise = add_gaus_noise(src, mean=0, sigma=0.1)  #평균, 표준편차
    src_noise = src_noise/255  #노이즈 더한이미지 아웃풋

    ######################################################
    # TODO                                               #
    # my_bilateral, gaussian mask 채우기                  #
    ######################################################
    dst = my_bilateral(src_noise, 5, 60, 40, 52, 111) ## 5, sigma, sigma-r,pos_x,pos_y
    dst = my_normalize(dst)

    gaus2D = my_get_Gaussian2D_mask(5 , sigma = 1) ## sigma?
    dst_gaus2D= my_filtering(src_noise, gaus2D)
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

