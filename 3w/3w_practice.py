import numpy as np
import cv2
import matplotlib.pyplot as plt

# 실습1 (add, sub
# src = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)
# add_src = cv2.add(src, 128)
# sub_src = cv2.subtract(src, 20)
#
# cv2.imshow('fruits', src)
# cv2.imshow('add 128', add_src)
# cv2.imshow('sub 128', sub_src)


# #실습 2
# src = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)
# # 꺽은선 그래프
# hist = cv2.calcHist([src], [0], None, [256], [0, 256])
# plt.plot(hist, color = 'r')
# plt.title('histogram plot')
# plt.xlabel('pixel intensity')
# plt.ylabel('pixel num')
# plt.show()
#
# # #히스토그램을 1차원 배열로 만듭니다.
# histFlatten = hist.flatten()
# # bar형 그래프
# binX = np.arange(len(histFlatten))
# plt.bar(binX, histFlatten, width = 0.5, color = 'g')
# plt.title('histogram bar')
# plt.xlabel('pixel intensity')
# plt.ylabel('pixel num')
# plt.show()

# 꺽은선 그래프 2
# if __name__ == '__main__':
#     arr = [0, 1, 2, 3, 2, 1]
#     plt.plot(arr, color = 'b')
#     plt.title('plt plot_test')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()

#막대 그래프
# if __name__ == '__main__':
#     arr = np.array([[3, 2, 1, 1, 2, 1]])
#     arr_flatten = arr.flatten()
#     binX = np.arange(len(arr_flatten))
#     plt.bar(binX, arr_flatten, width = 0.5, color = 'g')
#     plt.title('plt_bar test')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()

# 히스토그램 실습
#
# src = np.array([[3, 1, 3, 5, 4], [9, 8, 3, 5, 6],
#                 [2, 2, 3, 8, 7], [5, 4, 6, 5, 4],
#                 [1, 0, 0, 2, 6]], dtype=np.uint8)
# src_visible = (src/9 * 255).astype(np.uint8)
#
# cv2.imwrite('5x5_img.png', src_visible)

# def my_calcHist_gray_mini_img(mini_img):
#     h, w = mini_img.shape[:2]  #(h,w) = img.shape
#     hist = np.zeros((10,), dtype=np.int)
#     for row in range(h):
#         for col in range(w):
#             intensity = mini_img[row, col]
#             hist[intensity] += 1
#     return hist
#
# if __name__ == '__main__':
#     src = np.array([[3, 1, 3, 5, 4], [9, 8, 3, 5, 6],
#                     [2, 2, 3, 8, 7], [5, 4, 6, 5, 4],
#                     [1, 0, 0, 2, 6]], dtype=np.uint8)
#     hist = my_calcHist_gray_mini_img(src)
#     binX = np.arange(len(hist))
#     plt.bar(binX, hist, width = 0.8, color = 'g')
#     plt.title('histogram')
#     plt.xlabel('pixel intensity')
#     plt.ylabel('pixel num')
#     plt.show()
#
# 히스토그램 stretch
# img : input image
def my_calcHist_gray(img):
    h, w = img.shape[:2]  #(h,w) = img.shape
    hist = np.zeros((256,), dtype=np.int)
    for row in range(h):
        for col in range(w):
            intensity = img[row, col]
            hist[intensity] += 1
    return hist

def my_hist_stretch(src, hist):
    (h, w) = src.shape
    dst = np.zeros((h, w), dtype=np.uint8)
    min = 256
    max = -1
    # pixel intensity의 min갑과 max값 구하기
    # hist의 index : pixel intensity, hist의 값 : pixel num
    for i in range(len(hist)):
        if hist[i] != 0 and i < min:
            min = i
        if hist[i] != 0 and i > max:
            max = i

    hist_stretch = np.zeros(hist.shape, dtype=np.int)
    for i in range(min, max+1):
        j = int((255-0)/(max-min) * (i-min) + 0)
        hist_stretch[j] = hist[i]

    for row in range(h):
        for col in range(w):
            dst[row, col] = (255-0)/(max-min) * (src[row, col]-min) + 0

    return dst, hist_stretch

if __name__ == '__main__':
    src = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)
    src_div = cv2.imread('fruits_div.jpg', cv2.IMREAD_GRAYSCALE)

    # 원본 이미지와 div 1/3 적용한 이미지의 histogram 구하기
    hist = my_calcHist_gray(src)
    hist_div = my_calcHist_gray(src_div)

    #div 1/3 적용한 이미지를 stretch 적용
    dst, hist_stretch = my_hist_stretch(src_div, hist_div)

    #div 1/3이미지의 histogram
    binX = np.arange(len(hist_stretch))
    plt.bar(binX, hist_div, width=0.5, color='g')
    plt.title('divide 3 image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    #stretch 적용 후 histogram
    plt.bar(binX, hist_stretch, width=0.5, color='g')
    plt.title('stretching image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    #원본 이미지의 histogram
    plt.bar(binX, hist, width=0.5, color='g')
    plt.title('origin image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    #이미지 출력
    cv2.imshow('div 1.3 image', src_div)
    cv2.imshow('stretch image', dst)
    cv2.imshow('original', src)

    cv2.waitKey()
    cv2.destroyAllWindows()

