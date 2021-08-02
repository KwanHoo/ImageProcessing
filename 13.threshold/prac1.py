import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_hist(src):
    hist = np.zeros((256,))
    h, w = src.shape

    for row in range(h):
        for col in range(w):
            hist[src[row, col]] += 1

    return hist

def threshold(src, value):
    h, w = src.shape
    dst = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if src[row, col] <= value:
                dst[row, col] = 0
            else:
                dst[row, col] = 255
    return dst

def main():
    src = cv2.imread('rice.png', cv2.IMREAD_GRAYSCALE)
    hist = get_hist(src) #[] 0~255(256)개 각 픽셀마다 몇개 씩 있는지
    # print(hist)
    intensity = np.array([i for i in range(256)])
    h, w = src.shape
    print(h,w)
    # print(intensity)
    # p = hist * (1/(intensity + 1E-6))
    p = hist * (1/(h * w))
    # print(p)
    print(np.sum(p)) # = 1

    k_opt_warw = []
    k_opt_warb = []

    for k in range(256) : # [10, 70, 132 ,150, 200, 255]
        # 각각 한 줄로 작성하세요
        q1 = np.sum(p[:k])
        q2 = np.sum(p[k:]) #len(intensity)-1
        # #
        # print('q1,q2')
        # print(k,q1,q2)
        # print(q1+q2)
        # 굳이 할 필요 없는 경우
        if q1 == 0 or q2 == 0:
            k_opt_warw.append(np.inf)
            k_opt_warb.append(0)
            continue
    # #
    #     # 각각 한 줄로 작성하세요 (m1, m2, mg, var1, var2)
        m1 = np.sum(intensity[:k] * p[:k]) / q1
        m2 = np.sum(intensity[k:] * p[k:]) / q2


    # #
        mg = np.sum(intensity * p)
        # mg2 = q1 * m1 + q2 * m2  # 둘이 같음

    # #
        var1 = np.sum(np.square(intensity[k] - m1) * p[k]) * (1/q1)
        var2 = np.sum(np.square(intensity[k] - m2) * p[k]) * (1/q2)

        # var3 = (1/q1) * (np.sum((np.square(intensity[k]) * p[k] - np.square(m1)))) #식잘못됬음


    #     # 참고 : varg = np.sum(np.square(intensity - mg)*p)

        # 실수(float)라 약간의 오차가 있을 수 있음
        assert np.abs((q1 + q2) - 1) < 1E-6
        assert np.abs((q1 * m1 + q2 * m2) - mg) < 1E-6

        # 각각 한 줄로 작성하세요 (varw, varb)
        varw = (q1 * var1) + (q2 * var2)
        varb = q1 * q2 * (np.square(m1-m2))
        # varb2 = (q1 * (np.square(m1 - mg))) + (q2 +(np.square(m2 - mg)))

        # print('0000------',k)
        # print(varw)
        # print(varb)


        k_opt_warw.append(varw)
        k_opt_warb.append(varb)

    k_opt_warw = np.array(k_opt_warw)
    k_opt_warb = np.array(k_opt_warb)
    print(k_opt_warw.argmin())
    print(k_opt_warb.argmax()) #k = 132

    # 2개의 결과가 같아야 함
    # assert k_opt_warw.argmin() == k_opt_warb.argmax()

    dst = threshold(src, k_opt_warw.argmax())
    # return dst, k_opt_warw.argmin()
    print(dst)

    cv2.imshow('original', src)
    # cv2.imshow('threshold', dst)




if __name__ == '__main__':
    main()