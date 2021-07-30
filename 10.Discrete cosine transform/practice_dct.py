import cv2
import numpy as np

# 제로패딩
def my_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:h + p_h, p_w:w + p_w] = src
    return pad_img

#jpeg는 보통 block size = 8
def C(w, n = 8):
    if w == 0:
        return (1/n)**0.5
    else:
        return (2/n)**0.5

#block - src = (4,4)
def Spatial2Frequency_mask(block, n = 8):
    dst = np.zeros(block.shape) #(4,4)
    v, u = dst.shape # v=4,u=4
    # y, x = np.mgrid[0:4, 0:4] #0 ~ 3까지 //x,y정의
    y, x = np.mgrid[0:u, 0:v]
    '''
     y, x = np.mgrid[0:4, 0:4]
     y = [[ 0, 0, 0, 0],
          [ 1, 1, 1, 1],
          [ 2, 2, 2, 2],
          [ 3, 3, 3, 3]]
     x = [[0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3]]
     '''
    mask = np.zeros((n*n, n*n)) #(16,16)
    print("mask")
    print(mask)
    print(mask.shape)
    print('----')


    for v_ in range(v):
        for u_ in range(u):
            tmp =0

            ##########################################################################
            # ToDo                                                                   #
            # mask 만들기                                                             #
            # mask.shape = (16x16)                                                   #
            # DCT에서 사용된 mask는 (4x4) mask가 16개 있음 (u, v) 별로 1개씩 있음 u=4, v=4  #
            # 4중 for문으로 구현 시 감점 예정                                             #
            ##########################################################################
            tmp = block * np.cos(((2*x+1)*u_*np.pi)/(2*n)) * np.cos(((2*y+1)*v_*np.pi)/(2*n))
            print('tmp')
            print(tmp)
            print('t---------')
            print('v,u')
            print(v_,u_)

            mask[v_*4:(v_*4)+4, u_*4:(u_*4)+4] = (C(u_, n=n) * C(v_, n=n)) * tmp[0:4,0:4]
            print('mask')
            print(mask)
            print('--ma-----')



    return mask

def my_normalize(src):
    ##############################################################################
    # ToDo                                                                       #
    # my_normalize                                                               #
    # mask를 보기 좋게 만들기 위해 어떻게 해야 할 지 생각해서 my_normalize 함수 완성해보기   #
    ##############################################################################
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)


if __name__ == '__main__':
    block_size = 4
    src = np.ones((block_size, block_size)) #(4,4)

    mask = Spatial2Frequency_mask(src, n=block_size)
    mask = my_normalize(mask)
    mask = mask.astype(np.uint8)
    print(mask)

    #크기가 너무 작으니 크기 키우기 (16x16) -> (320x320)
    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('201804221_mask', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()



