import cv2
import numpy as np

def my_bilinear(src, scale): #
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape  #512, 512
    h_dst = int(h * scale + 0.5) #256
    w_dst = int(w * scale + 0.5) #256
    dst = np.zeros((h_dst, w_dst)) #(256 by 256)zero

    # bilinear interpolation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            row_s = (row / scale)
            col_s = (col / scale)
            # np.floor 내림함수
            row_n = np.floor(row_s)
            col_n = np.floor(col_s)
            # min() : 비교하여 최소값 아웃풋
            m_r = min(int(row_n), h - 1) #min_r
            m_c = min(int(col_n), w - 1) #min_c
            t = row_s - row_n
            s = col_s - col_n

            #src.shape = h,w (512,512)
            if m_c == (w - 1) and m_r != (h - 1):
                pixel_v = ((1.0 - t) * src[m_r, m_c]) + (t * src[m_r + 1, m_c])
            elif m_r == (h - 1) and m_c != (w - 1):
                pixel_v = ((1.0 - s) * src[m_r, m_c]) + (s * src[m_r, m_c + 1])
            elif m_c == (w - 1) and m_r == (h - 1):
                pixel_v = src[m_r, m_c]
            else:
                pixel_v = ((1.0 - s) * (1.0 - t) * src[m_r, m_c]) + (s * (1.0 - t) * src[m_r, m_c + 1]) + (
                        (1.0 - s) * t * src[m_r + 1, m_c]) + (s * t * src[m_r + 1, m_c + 1])
            pixel_v = max(min(pixel_v, 255), 0)
            dst[row, col] = pixel_v
        dst = (dst + 0.5).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 1/7
    #이미지 크기 1/2배로 변경
    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 2배로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


