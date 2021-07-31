import numpy as np

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

def main(n=8):

    QnT = np.zeros((n,n))
    QnT[1,0] = 2
    QnT[1,2] = 3
    QnT[2,2] = 4
    QnT[2,1] = 5
    QnT[2,0] = 6
    QnT[0,1] = 7
    QnT[0,2] = 9
    QnT[1,1] = 10
    print(QnT)

    a = my_zigzag_scanning(QnT)
    # print(a)
    zz = []
    zz.append(my_zigzag_scanning(QnT))
    # print(zz)

    # zigzag scanning
    # zz = []
    # for i in range(len(QnT)):
    #     zz.append(my_zigzag_scanning(QnT[i]))
    # print(zz)
    # print(zz[0])



if __name__ == '__main__':
    main()
