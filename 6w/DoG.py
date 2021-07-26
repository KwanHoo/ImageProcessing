import cv2
import numpy as np

# library add
# import os, sys
# sys.path.append("C:\\SelfStudyJ\\ImageProcessing")
# from my_library.filtering import my_filtering

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #up
        pad_img[0:p_h, p_w:w+p_w] = src[0, 0:w]
        #down
        pad_img[h+p_h:h+2*p_h, p_w:w+p_w] = src[h-1,0:w]
        #left
        # pad_img[0:h+2*p_h, 0:p_w] = pad_img[0:h+2*p_h, p_w:p_w+p_w]
        pad_img[:,:p_w] = pad_img[:, p_w: p_w + 1]
        #right
        pad_img[:, p_w +w:] = pad_img[:, p_w +w - 1 : p_w +w]
    return pad_img

def my_filtering(src, filter, pad_type='zero'):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape
    src_pad = my_padding(src, (f_h//2, f_w//2), pad_type)
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row+f_h, col:col+f_w] * filter)
            #val = np.clip(val, 0, 255)
            dst[row, col] = val

    #dst = (dst+0.5).astype(np.uint8)
    return dst

def get_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################

    (f_h, f_w) = (fsize, fsize) #fshape
    # y,x  = np.mgrid[-2:3 , -2:3] # -2 ~ 2까지 //x,y 정의
    y, x = np.mgrid[-(f_h // 2):(f_h // 2) + 1, -(f_w // 2):(f_w // 2) + 1]
    DoG_x = (-x / sigma**2) * np.exp(-(x **2 + y **2)/(2 * sigma**2))
    DoG_y = (-y / sigma**2) * np.exp(-(x **2 + y **2)/(2 * sigma**2))

    return DoG_x, DoG_y

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    ###################################################
    # TODO                                            #
    # DoG mask sigma값 조절해서 mask 만들기              #
    ###################################################
    # DoG_x, DoG_y filter 확인
    x, y = get_DoG_filter(fsize=256, sigma=60) #sigma = ?
    x = ((x - np.min(x)) /np.max(x - np.min(x)) * 255).astype(np.uint8)
    y = ((y - np.min(y)) /np.max(y - np.min(y)) * 255).astype(np.uint8)

    dst_x = my_filtering(src, DoG_x, 'zero')
    dst_y = my_filtering(src, DoG_y, 'zero')

    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    dst = np.sqrt((dst_x**2)+(dst_y**2))

    cv2.imshow('DoG_x filter', x)
    cv2.imshow('DoG_y filter', y)
    cv2.imshow('dst_x', dst_x/255)
    cv2.imshow('dst_y', dst_y/255)
    cv2.imshow('dst', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

