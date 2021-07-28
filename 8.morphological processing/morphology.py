import cv2
import numpy as np

# 제로패딩
def my_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:h + p_h, p_w:w + p_w] = src
    return pad_img

# 두껍게 해줌
def dilation(B, S):
    (h, w) = B.shape
    (x, y) = S.shape
    img_dilation = np.zeros((h, w))
    pad_img = my_padding(B, (x // 2, y // 2))
    for row in range(h):
        for col in range(w):
            temp = [pad_img[row:row + x, col:col + y] == S]
            temp = np.array(temp).flatten()
            if temp.any():
                img_dilation[row, col] = 1
            else:
                img_dilation[row, col] = 0

    return img_dilation

# 얇게 해줌
def erosion(B, S):
    (h, w) = B.shape
    (x, y) = S.shape
    img_erosion = np.zeros((h, w))
    pad_img = my_padding(B, (x // 2, y // 2))

    for row in range(h):
        for col in range(w):
            temp = [pad_img[row:row + x, col:col + y] == S]
            temp = np.array(temp).flatten()
            if temp.all():
                img_erosion[row, col] = 1
            else:
                img_erosion[row, col] = 0
    return img_erosion

# erosion -> dilation
def opening(B, S):
    E = erosion(B, S)
    img_opening = dilation(E, S)
    return img_opening
# dilation -> erosion
def closing(B, S):
    D = dilation(B, S)
    img_closing = erosion(D, S)
    return img_closing

if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('morphology_B.png', (B*255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)


