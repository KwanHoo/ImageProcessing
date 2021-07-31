import numpy as np
import cv2
import time

# 제로패딩
def my_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:h + p_h, p_w:w + p_w] = src
    return pad_img

# matrix
def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

# block 크기 8*8
def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    ######################################
    # lena (512,512) 64
    a = np.zeros((n,n)) #(8*8)
    x , y = src.shape
    x = x//n
    y = y//n

    blocks = np.zeros((n*n,n*n,n,n)) #(64,64,8,8)
    print(blocks.shape)

    for i in range(x):
        for j in range(y):
            a = src[i*n:(i+1)*n,j*n:(j+1)*n]
            blocks[i,j] = a

    return np.array(blocks) #(64,64,8,8)

# DCT--------------
#jpeg는 보통 block size = 8
def C(w, n = 8):
    if w == 0:
        return (1/n)**0.5
    else:
        return (2/n)**0.5

def DCT(block, n=8):

    dst = np.zeros(block.shape)
    v = n
    u = n
    # y, x = np.mgrid[0:8, 0:8] #0 ~ 7까지 //x,y정의
    y, x = np.mgrid[0:u, 0:v]

    for v_ in range(v):
        for u_ in range(u):
            tmp = block * np.cos(((2*x+1)*u_*np.pi)/(2*n)) * np.cos(((2*y+1)*v_*np.pi)/(2*n))
            dst[v_, u_] = C(u_, n=n) * C(v_, n=n) * np.sum(tmp)
    return np.round(dst)
#DCT-----------------
def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)

def my_zigzag_scanning(encodeDCT, blockSize = 8):

    row = 0
    col = 0
    zigzagList =[]

    while (row < blockSize) and (col < blockSize):
        if ((col + row) % 2) == 0:
            if row == 0:
                zigzagList.append(encodeDCT[row, col])

                if col == blockSize:
                    row = row + 1
                else:
                    col = col + 1

            elif (col == blockSize - 1) and (row < blockSize):
                zigzagList.append(encodeDCT[row, col])
                row = row + 1

            elif (row > 0) and (col < blockSize - 1):
                zigzagList.append(encodeDCT[row, col])
                row = row - 1
                col = col + 1

        else:
            if (row == blockSize - 1) and (col <= blockSize - 1):
                zigzagList.append(encodeDCT[row, col])
                col = col + 1

            elif col == 0:
                zigzagList.append(encodeDCT[row, col])

                if row == blockSize - 1:
                    col = col + 1
                else:
                    row = row + 1

            elif (row < blockSize - 1) and (col > 0):
                zigzagList.append(encodeDCT[row, col])
                row = row + 1
                col = col - 1
        # 종료조건
        if (row == blockSize - 1) and (col == blockSize - 1):
            zigzagList.append(encodeDCT[row, col])
            break

    return zigzagList

def Encoding(src, n=8):
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n) # blocks = (64,64,8,8)
    #subtract 128
    blocks -= 128
    #DCT
    blocks_dct = np.zeros((n*n,n*n,n,n))
    for i in range(64):
        for j in range(64):
            blocks_dct[i,j] = (DCT(blocks[i,j], n))
    blocks_dct = np.array(blocks_dct)
    #Quantization + thresholding
    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q) # matrix 곱하고 반올림

    o = len(QnT)
    p = len(QnT[0])
    # zigzag scanning
    zz = []
    for i in range(o):
        for j in range(p):
            zz.append(my_zigzag_scanning(QnT[i,j]))

    return zz, src.shape

def my_decodeZigzag(zigList):
    col = 0
    row = 0
    index = 0
    blockSize = 8
    decodeDCT = np.zeros((blockSize, blockSize))
    while (row < blockSize) and (col < blockSize):
        if ((col + row) % 2) == 0:
            if row == 0:
                decodeDCT[row, col] = zigList[index]

                if col == blockSize:
                    row = row + 1
                else:
                    col = col + 1
                index = index + 1

            elif (col == blockSize - 1) and (row < blockSize):
                decodeDCT[row, col] = zigList[index]
                row = row + 1
                index = index + 1

            elif (row > 0) and (col < blockSize - 1):
                decodeDCT[row, col] = zigList[index]
                row = row - 1
                col = col + 1
                index = index + 1

        else:

            if (row == blockSize - 1) and (col <= blockSize - 1):
                decodeDCT[row, col] = zigList[index]
                col = col + 1
                index = index + 1

            elif col == 0:
                decodeDCT[row, col] = zigList[index]
                if row == blockSize - 1:
                    col = col + 1
                else:
                    row = row + 1
                index = index + 1

            elif (row < blockSize - 1) and (col > 0):
                decodeDCT[row, col] = zigList[index]
                row = row + 1
                col = col - 1
                index = index + 1

        if (row == blockSize - 1) and (col == blockSize - 1):
            decodeDCT[row, col] = zigList[index]
            break

    return decodeDCT

def C_matrix(w, n=8):
    val = np.round(w/(w + 1E-6))
    val = (1 - val) * (1 / n) **0.5 + (val) * (2 / n) ** 0.5

    return val




def DCT_inv(block, n = 8):

    dst = np.zeros(block.shape)
    y, x = block.shape
    v, u = np.mgrid[0:y, 0:x]

    for x_ in range(x):
        for y_ in range(y):
            tmp = C_matrix(u, n=n) * C_matrix(v, n=n) * block * \
                  np.cos(((2 * x_ + 1) *u * np.pi) / (2* n)) * np.cos(((2 * y_ + 1) * v * np.pi) / (2 * n))
            dst[y_, x_] =  np.sum(tmp)

    return np.round(dst)

def block2img(blocks, src_shape, n = 8):
    dst = np.zeros(src_shape)

    for i in range(64): # 64
        for j in range(64): #blocks[0]
            dst[i * n:(i + 1) * n, j * n:(j + 1) * n] = (blocks[(i*64)+j])
    # dst = my_normalize(dst) # 노말라이즈

    return dst



def Decoding(zigzag, src_shape, n=8):
    print('<start Decoding>')
    # 디버깅 _ zigzag scanning
    blocks = []
    for i in range(len(zigzag[0])):
        for j in range(len(zigzag[0])):
            blocks.append(my_decodeZigzag(zigzag[(i*64)+j]))
    blocks = np.array(blocks)

    # Denormalizing
    Q = Quantization_Luminance()
    blocks = blocks * Q

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    # add 128
    blocks_idct += 128

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)
    # dst = my_normalize(dst)

    return dst



def main():
    start = time.time()
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    comp, src_shape = Encoding(src, n=8) # zz, src.shape
    # print(comp[0])
    # print(src_shape)

    recover_img = Decoding(comp, src_shape, n=8)
    recover_img = my_normalize(recover_img)
    print(recover_img.shape)
    print(recover_img[0,0])
    print(recover_img)

    total_time = time.time() - start

    print('time : ', total_time)
    if total_time > 45:
        print('Time Out!! 좀더 수정해보아요!!.')
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
