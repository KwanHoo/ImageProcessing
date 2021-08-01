import numpy as np
import cv2

def forward(src, M, fit=False):
    print('< forward >')
    print('M')
    print(M)
    if fit == False:
        # src.shape
        h, w = src.shape
        dst = np.zeros((h, w))
        N = np.zeros(dst.shape)

        for row in range(h):
            for col in range(w):
                P = np.array([
                    [col],
                    [row],
                    [1]
                ])
                # ((3,3)(3,1) =(3,1)  [[a],[b],[1]]
                P_dst = np.dot(M, P)
                dst_col = P_dst[0][0]  # 첫째행 a ex)1.9
                dst_row = P_dst[1][0]  # 두번쨰 b

                # np.ceil : 각 원소 값보다 크거나 같은 가장 작은 정수 값 (천장 값)으로 올림
                dst_col_right = int(np.ceil(dst_col))  # 인덱스+1 ex)2
                dst_col_left = int(dst_col)  # 인덱스 ex)1

                dst_row_bottom = int(np.ceil(dst_row))  # 2
                dst_row_top = int(dst_row)  # 1

                dst[dst_row_top, dst_col_left] += src[row, col]  # (1,1)
                N[dst_row_top, dst_col_left] += 1

                if dst_col_right != dst_col_left:
                    dst[dst_row_top, dst_col_right] += src[row, col]  # (1,2)
                    N[dst_row_top, dst_col_right] += 1

                if dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_left] += src[row, col]  # (2,1)
                    N[dst_row_bottom, dst_col_left] += 1

                if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_right] += src[row, col]  # (2,2)
                    N[dst_row_bottom, dst_col_right] += 1

        dst = np.round(dst / (N + 1E-6))
        dst = dst.astype(np.uint8)
    else:
        h_src, w_src = src.shape
        dst = np.zeros((h_src, w_src))
        h, w = dst.shape

        M_inv = np.linalg.inv(M)

        for row in range(h):
            for col in range(w):
                P_dst = np.array([
                    [col],
                    [row],
                    [1]])
                # ((3,3)(3,1) =(3,1)  [[a],[b],[1]]
                P = np.dot(M_inv, P_dst)
                src_col = P[0][0]  # 첫째행 a ex)1.9
                src_row = P[1][0]  # 두번쨰 b

                src_col_right = int(np.ceil(src_col))
                src_col_left = int(src_col)
                src_row_bottom = int(np.ceil(src_row))
                src_row_top = int(src_row)

                if src_col_right >= w_src or src_row_bottom >= h_src:
                    continue
                s = src_col - src_col_left
                t = src_row - src_row_top

                intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left] \
                            + s * (1 - t) * src[src_row_top, src_col_right] \
                            + (1 - s) * t * src[src_row_bottom, src_col_left] \
                            + s * t * src[src_row_bottom, src_col_right]

                dst[row, col] = intensity
        dst = dst.astype(np.uint8)
    return dst

def backward(src, M):
    h_src, w_src = src.shape
    dst = np.zeros((h_src, w_src))
    h, w = dst.shape

    M_inv = np.linalg.inv(M)

    for row in range(h):
        for col in range(w):
            P_dst = np.array([
                [col],
                [row],
                [1]
            ])
            # ((3,3)(3,1) =(3,1)  [[a],[b],[1]]
            P = np.dot(M_inv, P_dst)
            src_col = P[0][0]  # 첫째행 a ex)1.9
            src_row = P[1][0]  # 두번쨰 b

            src_col_right = int(np.ceil(src_col))
            src_col_left = int(src_col)

            src_row_bottom = int(np.ceil(src_row))
            src_row_top = int(src_row)

            if src_col_right >= w_src or src_row_bottom >= h_src:
                continue

            s = src_col - src_col_left
            t = src_row - src_row_top

            intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left] \
                        + s * (1 - t) * src[src_row_top, src_col_right] \
                        + (1 - s) * t * src[src_row_bottom, src_col_left] \
                        + s * t * src[src_row_bottom, src_col_right]

            dst[row, col] = intensity
    dst = dst.astype(np.uint8)
    return dst

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # translation (-30,50)
    M_tr = np.array([
        [1, 0, -30],
        [0, 1, 50],
        [0, 0, 1]])
    # scaling (x0.5, x0.5)
    M_sc = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 1]])
    # rotation (-20도)
    degree = -20
    M_ro = np.array([
        [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0 ],
        [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0 ],
        [0, 0, 1]])
    # shearing (0.2, 0.2)
    M_sh = np.array([
        [1, 0.2, 0],
        [0.2, 1, 0],
        [0, 0, 1]])
    # rotation -> translation -> Scale -> Shear
    M = np.dot(M_sh ,np.dot(M_sc, np.dot(M_tr, M_ro)))

    # fit이 True인 경우와 False인 경우 다 해야 함.
    fit = True
    # forward
    dst_for = forward(src, M)
    dst_for2 = forward(dst_for, np.linalg.inv(M), fit=fit)

    # backward
    dst_back = backward(src, M)
    dst_back2 = backward(dst_back, np.linalg.inv(M))

    cv2.imshow('original', src)
    cv2.imshow('forward2', dst_for2)
    cv2.imshow('backward2', dst_back2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()