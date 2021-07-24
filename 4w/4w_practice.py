import numpy as np
import time
import cv2

#numpy 기초 실습
# def main():
#     v1 = np.full((3,3), 3)
#     v2 = np.ones((3,3))
#
#     print(v1 * v2)
#     print(np.sum(v1 * v2))
#

#
# def main():
#     v1 = np.full((1000, 1000), 3)
#     v2 = np.ones((1000, 1000))
#
#     numpy_start = time.time()
#     print(np.sum(v1 * v2))
#     print(time.time() - numpy_start)
#
#     sum = 0
#     for_start = time.time()
#     for i in range(1000):
#         for j in range(1000):
#             sum += v1[i,j] * v2[i,j]
#     print(sum)
#     print(time.time() - for_start)
#
# if __name__ == '__main__':
#     main()

# 평균필터실습1
# def my_average_filter_3x3(src):
#     mask = np.array([[1/4, 1/4, 1/4],
#                      [1/4, 1/4, 1/4],
#                      [1/4, 1/4, 1/4]])
#
#     dst = cv2.filter2D(src, -1, mask)
#     return dst
#
# if __name__ == '__main__':
#     src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
#     dst = my_average_filter_3x3(src)
#
#     cv2.imshow('original', src)
#     cv2.imshow('average filter', dst)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

# 샤프닝 필터
# def my_sharpening_filter_3x3(src):
#     mask = np.array([[-1/9, -1 /9, -1/ 9],
#                      [-1 / 9, 26/ 9, -1 / 9],
#                      [-1 / 9, -1 / 9, -1 / 9]])
#
#     dst = cv2.filter2D(src, -1, mask)
#     return dst
#
#
# if __name__ == '__main__':
#     src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
#     dst = my_sharpening_filter_3x3(src)
#
#     cv2.imshow('original', src)
#     cv2.imshow('sharpening filter', dst)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

# 제로 패딩
def my_padding(src, pad_shap, pad_type = 'zero'):
    (h,w) = src.shape
    (p_h,p_w) = pad_shap
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        #up
        pad_img[0:p_h, p_w:w+p_w] = src[0, 0:w]  #src[p_h+1, p_w:w+2*p_w]
        #down
        pad_img[h+p_h:h+2*p_h, p_w:w+p_w] = src[h-1,0:w]
        # #left
        # # pad_img[p_h:h+p_h, 0:p_w] = src[0:h, 0] # np.full((512,20),0)
        # pad_img[0:h+2*p_h, 0:p_w] = pad_img[0:h+2*p_h, p_w:p_w+p_w] #p_w///
        # # pad_img[0:h+2*p_h, 0:p_w] = np.full((h+2*p_h,p_w), pad_img[0:h+2*p_h,p_w])
        # #right
        # pad_img[0:h+2*p_h, p_w+w:w+2*p_w] = pad_img[0:h+2*p_h, w:w+p_w] #p_w+w
        # left
        pad_img[:,:p_w] = pad_img[:, p_w: p_w + 1]
        #right
        pad_img[:, p_w +w:] = pad_img[:, p_w +w - 1 : p_w +w]

    else:
        print('zero padding')

    return pad_img

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    #zero padding
    dst1 = my_padding(src, (20,20))
    dst1 = dst1.astype(np.uint8)

    #repetition padding
    dst2 = my_padding(src, (20,20), 'repetition')
    dst2 = dst2.astype((np.uint8))

    cv2.imshow('zero padding', dst1)
    cv2.imshow('repetition padding', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()