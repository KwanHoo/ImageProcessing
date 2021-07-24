import cv2
import numpy as np

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
        # pad_img[0:h+2*p_h, p_w+w:w+2*p_w] = pad_img[0:h+2*p_h, w:w+p_w]
        pad_img[:, p_w +w:] = pad_img[:, p_w +w - 1 : p_w +w]
    else:
        print('zero padding')
    return pad_img

 # img/average/ (3,3)
def my_filtering(src, ftype, fshape, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (fshape[0]//2, fshape[1]//2), pad_type)
    dst = np.zeros((h, w))  #dst : 필터 결과 이미지

    if ftype == 'average':
        print('average filtering')
        row_size = fshape[0]
        col_size = fshape[1]
        mask = np.ones((row_size, col_size))/(row_size * col_size)
        #mask 확인
        print(mask)
    elif ftype == 'sharpening':
        print('sharpening filtering')
        row_size = fshape[0]
        col_size = fshape[1]
        front = np.zeros((row_size, col_size))
        mid_r = int(row_size/2-0.5)
        mid_c = int(col_size/2-0.5)
        front[mid_r][mid_c] = 2
        print(front)
        mask = front - np.ones((row_size, col_size))/(row_size * col_size)
        #mask 확인
        print(mask)

    #########################################################
    # TODO                                                  #
    # dst 완성                                               #
    # dst : filtering 결과 image                             #
    # 꼭 한줄로 완성할 필요 없음                                 #
    #########################################################
    x = int((fshape[0]-1)/2) #필터 row size ex) 3일경우 (3-1)/2= 1 7일경우 (7-1)/2 = 3
    y = int((fshape[1]-1)/2) #필터 col size 의 중간값

    v1 = mask
    (h,w) = src.shape
    # (x,y) = mask.shape

    for row in range(x, h-x):
        for col in range(y, w-y):
            sum = 0
            v2 = src[row-x:row+x+1,col-y: col+y+1]
            sum = np.sum(v1 * v2)
            dst[row][col] = sum
            if dst[row][col] > 255:
                dst[row][col] = 255
            elif dst[row][col] < 0:
                dst[row][col] = 0

    dst = (dst+0.5).astype(np.uint8)
    print(dst)
    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # repetition padding test
    rep_test = my_padding(src, (20,20))

    # # 3x3 filter
    dst_average = my_filtering(src, 'average', (3,3))
    dst_sharpening = my_filtering(src, 'sharpening', (3,3))

    #원하는 크기로 설정
    # dst_average = my_filtering(src, 'average', (7,9))
    # dst_sharpening = my_filtering(src, 'sharpening', (7,9))

    # 11x13 filter
    # dst_average = my_filtering(src, 'average', (11,13), 'repetition')
    # dst_sharpening = my_filtering(src, 'sharpening', (11,13), 'repetition')

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
