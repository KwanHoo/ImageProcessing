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


def get_threshold(src, type='rice'):

    hist = get_hist(src) #[] 0~255(256)개 각 픽셀마다 몇개 씩 있는지
    intensity = np.array([i for i in range(256)])
    h, w = src.shape

    if type == 'rice':
        # 한 줄로 작성하세요 ,확률
        p = hist * (1 / (h * w))
    else:  # 'meat'
    #     # 여러줄로 작성하셔도 상관 없습니다.
        hist[0] = 0
        p = hist * (1 / (h * w))


    k_opt_warw = []
    k_opt_warb = []
    for k in range(256):
        # 각각 한 줄로 작성하세요
        q1 = np.sum(p[:k])
        q2 = np.sum(p[k:])

        # 굳이 할 필요 없는 경우
        if q1 == 0 or q2 == 0:
            k_opt_warw.append(np.inf)
            k_opt_warb.append(0)
            continue

        # 각각 한 줄로 작성하세요 (m1, m2, mg, var1, var2)
        m1 = np.sum(intensity[:k] * p[:k]) / q1
        m2 = np.sum(intensity[k:] * p[k:]) / q2

        mg = np.sum(intensity * p)

        var1 = np.sum(np.square(intensity[k] - m1) * p[k]) * (1/q1)
        var2 = np.sum(np.square(intensity[k] - m2) * p[k]) * (1/q2)

        # varg = np.sum(np.square(intensity - mg)*p)

        # 실수(float)라 약간의 오차가 있을 수 있음
        # assert np.abs((q1 + q2) - 1) < 1E-6
        # assert np.abs((q1 * m1 + q2 * m2) - mg) < 1E-6

        # 각각 한 줄로 작성하세요 (varw, varb)
        varw = (q1 * var1) + (q2 * var2)
        varb = q1 * q2 * (np.square(m1-m2))

        k_opt_warw.append(varw)
        k_opt_warb.append(varb)

    k_opt_warw = np.array(k_opt_warw)
    k_opt_warb = np.array(k_opt_warb)

    # 2개의 결과가 같아야 함
    # assert k_opt_warw.argmin() == k_opt_warb.argmax()

    dst = threshold(src, k_opt_warb.argmax())
    return dst, k_opt_warw.argmax()



def meat_main():
    meat = cv2.imread('meat.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('mask.TIFF', cv2.IMREAD_GRAYSCALE)
    #####################################################
    # TODO                                              #
    # meat_main 완성                                     #
    # 이 부분은 결과가 잘 나오도록 각자 알아서 구현해보세요       #
    #####################################################
    print(meat.shape) #(512,512)
    print(mask.shape) #(512,512) 0 or 255
    h, w = mask.shape
    # x = mask[[0]] #첫쨰줄
    # print(len(x)) # = 1
    # print(x) # [[512개]]
    # print(mask[0,1])#첫째줄 첫원소
    if mask[130,150] == 255:
        print('hello')
    else:
        print('no')

    dst = np.zeros((512,512))

    for i in range(h):
        for j in range(w):
            if mask[i,j] == 255:
                dst[i,j] = meat[i,j]

    print(meat[135])
    print(dst[135])
    print(meat.dtype)
    print(dst.dtype)
    target = dst.astype('uint8')


    print('''--''')
    # print(target[150])
    # dst,
    nomal = meat /np.max(meat)


    dst_1, threshold_value = get_threshold(target, 'meat')
    proto = dst_1 / np.max(dst_1) # 255 값 0~1값으로 노말라이즈 #잘린모습
    cv2.imshow('th_t', dst_1)
    # cv2.imshow('nomal', proto)
    final = np.zeros((512,512))
    for i in range(h):
        for j in range(w):
            if target[i,j] == 0:
                final[i,j] = nomal[i,j]
            else:
                final[i,j] = proto[i,j]
    cv2.imshow('final',final)


    # final = ???



    # cv2.imshow('dst', meat)
    # cv2.imshow('final', mask)


    cv2.waitKey()
    cv2.destroyAllWindows()




def main():

    meat_main()



if __name__ == '__main__':
    main()