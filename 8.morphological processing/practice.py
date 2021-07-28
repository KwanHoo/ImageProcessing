import cv2
import numpy as np
import os


def labeling(B, neighbor=4):
    label = 0
    h, w = B.shape
    dst = B.copy()

    for row in range(h):
        for col in range(w):
            if dst[row, col] == -1:
                coordinates = find_set_pos(dst, row, col, [])
                label += 1

                while coordinates:
                    r, c = coordinates.pop()
                    dst[r, c] = label

    return dst


def find_set_pos(src, row, col, coordinates):
    if (row, col) in coordinates or src[row, col] != -1:
        return coordinates

    coordinates.append((row, col))
    h, w = src.shape

    # up
    if row > 0:
        coordinates = find_set_pos(src, row - 1, col, coordinates)
    # down
    if row < h - 1:
        coordinates = find_set_pos(src, row + 1, col, coordinates)
    # left
    if col > 0:
        coordinates = find_set_pos(src, row, col - 1, coordinates)
    # right
    if col < w - 1:
        coordinates = find_set_pos(src, row, col + 1, coordinates)

    # 중복제거
    return list(set(coordinates))


if __name__=='__main__':
    B = np.array(
        [[0, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 1, 0],
         [1, 1, 1, 0, 1, 1, 1, 0],
         [1, 1, 1, 0, 1, 0, 0, 0]]
    )
    B = 1 - B

    # B = (B*255).astype(np.uint8)
    # path = 'C:/SelfStudyJ/ImageProcessing/8w'
    cv2.imwrite('binary_image.png', (B*255).astype(np.uint8))
    # cv2.waitKey()
    B = B * -1
    # print(B)

    # 라벨링
    B = labeling(B)
    print(B)
    cv2.imwrite('binary_image_labeling.png', (B*255).astype(np.uint8))





