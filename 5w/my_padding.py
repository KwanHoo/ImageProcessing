import numpy as np
import cv2

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


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # repetition padding test
    rep_test = my_padding(src, (20,20))

    # # 3x3 filter
    # dst_average = my_filtering(src, 'average', (3,3))
    # dst_sharpening = my_filtering(src, 'sharpening', (3,3))

    #원하는 크기로 설정
    # dst_average = my_filtering(src, 'average', (7,9))
    # dst_sharpening = my_filtering(src, 'sharpening', (7,9))

    # 11x13 filter
    # dst_average = my_filtering(src, 'average', (11,13), 'repetition')
    # dst_sharpening = my_filtering(src, 'sharpening', (11,13), 'repetition')

    cv2.imshow('original', src)
    # cv2.imshow('average filter', dst_average)
    # cv2.imshow('sharpening filter', dst_sharpening)
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
