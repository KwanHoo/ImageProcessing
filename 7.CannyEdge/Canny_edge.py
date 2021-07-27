import cv2
import numpy as np

def my_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:h + p_h, p_w:w + p_w] = src

    return pad_img

# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨

    ###########################################
    # TODO                                    #
    # apply_lowNhigh_pass_filter 완성          #
    # Ix와 Iy 구하기                            #
    ###########################################

    (h, w) = src.shape
    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]
    DoG_x = (-x / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    (a, b)= DoG_x.shape
    pad_img = my_padding(src, (a//2, b //2))
    Ix = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            Ix[row, col] = np.sum(pad_img[row:row + a, col:col + b] * DoG_x)

    DoG_y = (-y / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    (c, d) = DoG_y.shape
    pad_img = my_padding(src, (c // 2, d // 2))
    Iy = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            Iy[row, col] = np.sum(pad_img[row:row + c, col:col + d] * DoG_y)

    return Ix, Iy

# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    magnitude = np.sqrt(Ix**2 + Iy**2)
    return magnitude

# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    ###################################################
    # TODO                                            #
    # calcAngle 완성                                   #
    # angle     : ix와 iy의 angle                      #
    # e         : 0으로 나눠지는 경우가 있는 경우 방지용     #
    # np.arctan 사용하기(np.arctan2 사용하지 말기)        #
    ###################################################
    e = 1E-6
    angle = np.arctan(Iy, Ix+e)
    return angle

# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    (h, w) = magnitude.shape
    largest_magnitude = np.zeros((h, w))
    angle = angle + np.pi
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            theta = angle[row, col]
            if (0 <= theta < np.pi/4) or (np.pi <= theta < 5/4*np.pi):
                t = np.tan(theta)
                front = magnitude[row + 1, col + 1] * t + magnitude[row, col + 1] * (1 - t)
                post = magnitude[row - 1, col - 1] * t + magnitude[row, col - 1] * (1 - t)
            elif (np.pi/4 <= theta < np.pi/2) or (5/4*np.pi <= theta < 3/2*np.pi):
                t = 1 / np.tan(theta)
                front = magnitude[row + 1, col + 1] * t + magnitude[row + 1, col] * (1 - t)
                post = magnitude[row - 1, col - 1] * t + magnitude[row - 1, col] * (1 - t)
            elif (np.pi/2 <= theta < 3/4*np.pi) or (3/2*np.pi <= theta < 7/4*np.pi):
                t = -1 / np.tan(theta)
                front = magnitude[row + 1, col - 1] * t + magnitude[row + 1, col] * (1 - t)
                post = magnitude[row - 1, col + 1] * t + magnitude[row - 1, col] * (1 - t)
            elif (3/4*np.pi <= theta < np.pi) or (7/4*np.pi <= theta < np.pi*2):
                t = -np.tan(theta)
                front = magnitude[row + 1, col - 1] * t + magnitude[row, col - 1] * (1 - t)
                post = magnitude[row - 1, col + 1] * t + magnitude[row, col + 1] * (1 - t)

            max_value = max([front, post, magnitude[row, col]])
            if max_value == magnitude[row, col]:
                a = 1
            else:
                a = 0
            largest_magnitude[row, col] = a * magnitude[row, col]

    # largest_magnitude값을 0~255의 uint8로 변환
    largest_magnitude = (largest_magnitude / np.max(largest_magnitude) * 255).astype(np.uint8)

    return largest_magnitude


# double_thresholding 수행
def double_thresholding(src):

    high_threshold_value, _ = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    # high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고
    # low threshold값은 (high threshold * 0.4)로 구한다
    low_threshold_value = high_threshold_value * 0.4

    ######################################################
    # TODO                                               #
    # double_thresholding 완성                            #
    # dst     : double threshold 실행 결과 이미지           #
    ######################################################

    (h, w) = src.shape
    dst = np.zeros((h, w))

    weak = low_threshold_value+1
    strong = 255

    strong_i, strong_j = np.where(src > high_threshold_value)
    zeros_i, zeros_j = np.where(src < low_threshold_value)

    weak_i, weak_j = np.where((src <= high_threshold_value) & (src >= low_threshold_value))
    dst[zeros_i, zeros_j] = 0
    dst[strong_i, strong_j] = strong
    dst[weak_i, weak_j] = weak
    change = True
    while change:
        change = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if dst[i, j] == weak:
                    if ((dst[i + 1, j - 1] == strong) or (dst[i + 1, j] == strong) or (dst[i + 1, j + 1] == strong)
                            or (dst[i, j - 1] == strong) or (dst[i, j + 1] == strong)
                            or (dst[i - 1, j - 1] == strong) or (dst[i - 1, j] == strong) or (
                                        dst[i - 1, j + 1] == strong)):
                        dst[i, j] = strong
                if dst[i, j] == strong:
                    if ((dst[i + 1, j - 1] == weak) or (dst[i + 1, j] == weak) or (dst[i + 1, j + 1] == weak)
                            or (dst[i, j - 1] == weak) or (dst[i, j + 1] == weak)
                            or (dst[i - 1, j - 1] == weak) or (dst[i - 1, j] == weak) or (
                                    dst[i - 1, j + 1] == weak)):
                        change = True
    weak_i, weak_j = np.where(dst == weak)
    dst[weak_i, weak_j] = 0
    return dst

def my_canny_edge_detection(src, fsize=3, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    # DoG 를 사용하여 1번 filtering
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)

    # Ix와 Iy 시각화를 위해 임시로 Ix_t와 Iy_t 만들기
    Ix_t = np.abs(Ix)
    Iy_t = np.abs(Iy)
    Ix_t = Ix_t / Ix_t.max()
    Iy_t = Iy_t / Iy_t.max()

    cv2.imshow("Ix", Ix_t)
    cv2.imshow("Iy", Iy_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    magnitude_t = magnitude
    magnitude_t = magnitude_t / magnitude_t.max()
    cv2.imshow("magnitude", magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # non-maximum suppression 수행
    largest_magnitude = non_maximum_supression(magnitude, angle)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    largest_magnitude_t = largest_magnitude
    largest_magnitude_t = largest_magnitude_t / largest_magnitude_t.max()
    cv2.imshow("largest_magnitude", largest_magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # double thresholding 수행
    dst = double_thresholding(largest_magnitude)
    return dst

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()