import numpy as np
import cv2



def img2block(src, n=8):
    a = np.zeros((n, n))  # (8*8)
    x, y = src.shape
    x = x // n
    y = y // n

    blocks = np.zeros((n * n, n * n, n, n))
    print(blocks.shape)
    # print(blocks[1])

    for i in range(x):
        for j in range(y):
            # print(i,j)
            a = src[i * n:(i + 1) * n, j * n:(j + 1) * n]
            # print(a)
            blocks[i, j] = a

    # print(dst)
    # print(dst.shape)
    # print(x,y)

    return np.array(blocks)

def block2img(blocks, src_shape, n=8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################

    #blocks = blocks_idct + 128  #(4096,8,8)

    dst = np.zeros(src_shape)

    for i in range(64): # 64
        for j in range(64): #blocks[0]
            dst[i * n:(i + 1) * n, j * n:(j + 1) * n] = (blocks[(i*64)+j])
    dst = my_normalize(dst) ####################?????여기?? 맞아!!

    return dst

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
    y, x = np.mgrid[0:u, 0:v] #8
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

    for v_ in range(v):
        for u_ in range(u):
            tmp = block * np.cos(((2*x+1)*u_*np.pi)/(2*n)) * np.cos(((2*y+1)*v_*np.pi)/(2*n))
            dst[v_:, u_] = C(u_, n=n) * C(v_, n=n) * np.sum(tmp)

    return np.round(dst)

def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)
    # if dst>= 0:
    #     dst = dst / dst.max(axis=0)
    # else:
    #     dst = (dst - dst.min(0)) / dst.ptp(0)
    #
    # return dst.astype(np.uint8)
#DCT-----------------

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

def my_zigzag_scanning(encodeDCT):
    row = 0
    col = 0
    zigzagList =[]
    blockSize = 8

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

        if (row == blockSize - 1) and (col == blockSize - 1):
            zigzagList.append(encodeDCT[row, col])
            break

    return zigzagList

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

def DCT_inv(block, n = 8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################

    dst = np.zeros(block.shape)
    v = n
    u = n
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
    # mask = np.zeros((n*n, n*n)) #(16,16)float

    for v_ in range(v):
        for u_ in range(u):
            tmp = block * C(u_, n=n) * C(v_, n=n) * np.cos(((2*x+1)*u_*np.pi)/(2*n)) * np.cos(((2*y+1)*v_*np.pi)/(2*n))

            dst[v_, u_] = np.sum(tmp)

    return np.round(dst)

def main(n=8):
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # a = np.zeros((n,n)) #(8*8)
    # x , y = src.shape
    # x = x//n
    # y = y//n
    #
    # blocks = np.zeros((n*n,n*n,n,n))
    # print(blocks.shape)
    # # print(blocks[1])
    #
    # for i in range(x):
    #     for j in range(y):
    #         # print(i,j)
    #         a = src[i*n:(i+1)*n,j*n:(j+1)*n]
    #         # print(a)
    #         blocks[i,j] = a
    blocks = img2block(src, n=n)
    # blocks = (64,64,8,8)
    # print(blocks[43,43])
    # print(blocks[44,44])

    #subtract 128
    blocks -= 128
    # print(blocks[43,43])
    # print(blocks[44,44])
    # 64 64 8 8

    #DCT
    blocks_dct = np.zeros((n*n,n*n,n,n))
    for i in range(64):
        for j in range(64):
            blocks_dct[i,j] = (DCT(blocks[i,j], n))
    blocks_dct = np.array(blocks_dct)
    # print(blocks_dct)
    # print(blocks_dct.shape) #(64 , 64, 8,8)
    # print(blocks_dct[43,43])
    # print(blocks_dct[44,44])


    #Quantization + thresholding
    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q) # matrix 곱하고 반올림
    # print(QnT[43,43])
    # print(QnT[44,44])


    o = len(QnT)
    p = len(QnT[0])
    # print('qnt')
    # print(QnT.shape)
    # print(QnT[0,1])
    # print(QnT[1,0])
    # print(QnT[63,63])

    # # zigzag scanning
    zz = []
    for i in range(o):
        for j in range(o):
            zz.append(my_zigzag_scanning(QnT[i,j]))
# 64, 8, 8
#     print(len(zz)) #4096 = 64*64



    comp = zz
    zigzag = comp
    print('================')
    print(len(zigzag[0]))
    print(zigzag[0]) #QnT[0,0]
    print(zigzag[1]) #QnT [0,1]
    print(zigzag[64]) #QnT [1,0]
    print(zigzag[4095]) #Qnt[63,63]
    print()
    print('================')

    # # 디버깅 zigzag scanning

    blocks = []
    for i in range(len(zigzag[0])):
        for j in range(len(zigzag[0])):
            blocks.append(my_decodeZigzag(zigzag[(i*64)+j]))
    # for i in range(len(zigzag[0])):
    #     blocks.append(my_decodeZigzag(zigzag[i]))
    blocks = np.array(blocks)

    # # Denormalizing
    Q = Quantization_Luminance()
    blocks = blocks * Q
    #
    print(blocks.shape) # (4096,8,8)

    # #DCT
    # blocks_dct = np.zeros((n*n,n*n,n,n))
    # for i in range(64):
    #     for j in range(64):
    #         blocks_dct[i,j] = (DCT(blocks[i,j], n))
    # blocks_dct = np.array(blocks_dct)

    # # inverse DCT

    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))

    # blocks_idct = [n*n*n*n,n,n]
    # for i in range(64):
    #     for j in range(64):
    #         blocks_idct = (DCT(blocks[(i*64)+j], n))

    blocks_idct = np.array(blocks_idct)

    # print(blocks_idct[0])
    #
    # # add 128
    blocks_idct += 128
    # print(blocks_idct[0])
    # print(blocks_idct.shape)
    # print(len(blocks_idct))

    n = 8
    src_shape = src.shape
    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    #
    # print(dst)
    #
    # print('src')
    # print(src)
    # print(dst.shape)
    # print(src.shape)
    cv2.imshow('recover img', dst)
    cv2.imshow('src',src)
    cv2.waitKey()
    cv2.destroyAllWindows()





    # comp = np.load('comp.npy', allow_pickle=True) #리스트
    # src_shape = np.load('src_shape.npy') #[385,512]
    # print(comp)
    # print(comp[0])
    # print(comp[1])








if __name__ == '__main__':
    main()
