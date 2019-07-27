import cv2
import numpy as np
from convolve import img_convolve
from sobel_filter import sobel_filter

PI = 180


def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = np.mgrid[0 - center:k_size - center, 0 - center:k_size - center]
    g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    return g


def canny(image, threshold_low=15, threshold_high=30, weak=128, strong=255):
    image_row, image_col = image.shape[0], image.shape[1]
    # gaussian_filter
    gaussian_out = img_convolve(image, gen_gaussian_kernel(9, sigma=1.4))
    # get the gradient and degree by sobel_filter
    sobel_grad, sobel_theta = sobel_filter(gaussian_out)
    gradient_direction = np.rad2deg(sobel_theta)
    gradient_direction += PI

    dst = np.zeros((image_row, image_col))

    """
    Non-maximum suppression. If the edge strength of the current pixel is the largest compared to the other pixels 
    in the mask with the same direction, the value will be preserved. Otherwise, the value will be suppressed. 
    """
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            if (
                0 <= direction < 22.5
                    or 15 * PI / 8 <= direction <= 2 * PI
                    or 7 * PI / 8 <= direction <= 9 * PI / 8
            ):
                W = sobel_grad[row, col - 1]
                E = sobel_grad[row, col + 1]
                if sobel_grad[row, col] >= W and sobel_grad[row, col] >= E:
                    dst[row, col] = sobel_grad[row, col]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                SW = sobel_grad[row + 1, col - 1]
                NE = sobel_grad[row - 1, col + 1]
                if sobel_grad[row, col] >= SW and sobel_grad[row, col] >= NE:
                    dst[row, col] = sobel_grad[row, col]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                N = sobel_grad[row - 1, col]
                S = sobel_grad[row + 1, col]
                if sobel_grad[row, col] >= N and sobel_grad[row, col] >= S:
                    dst[row, col] = sobel_grad[row, col]

            elif (5 * PI / 8 <= direction < 7 * PI / 8) or (13 * PI / 8 <= direction < 15 * PI / 8):
                NW = sobel_grad[row - 1, col - 1]
                SE = sobel_grad[row + 1, col + 1]
                if sobel_grad[row, col] >= NW and sobel_grad[row, col] >= SE:
                    dst[row, col] = sobel_grad[row, col]

            """
            High-Low threshold detection. If an edge pixel’s gradient value is higher than the high threshold
            value, it is marked as a strong edge pixel. If an edge pixel’s gradient value is smaller than the high
            threshold value and larger than the low threshold value, it is marked as a weak edge pixel. If an edge
            pixel's value is smaller than the low threshold value, it will be suppressed.
            """
            if dst[row, col] >= threshold_high:
                dst[row, col] = strong
            elif dst[row, col] <= threshold_low:
                dst[row, col] = 0
            else:
                dst[row, col] = weak

    """
    Edge tracking. Usually a weak edge pixel caused from true edges will be connected to a strong edge pixel while
    noise responses are unconnected. As long as there is one strong edge pixel that is involved in its 8-connected
    neighborhood, that weak edge point can be identified as one that should be preserved.
    """
    for row in range(1, image_row):
        for col in range(1, image_col):
            if dst[row, col] == weak:
                if 255 in (
                        dst[row, col + 1],
                        dst[row, col - 1],
                        dst[row - 1, col],
                        dst[row + 1, col],
                        dst[row - 1, col - 1],
                        dst[row + 1, col - 1],
                        dst[row - 1, col + 1],
                        dst[row + 1, col + 1],
                ):
                    dst[row, col] = strong
                else:
                    dst[row, col] = 0

    return dst


if __name__ == '__main__':
    # read original image in gray mode
    lena = cv2.imread(r'./img/1.png', 0)
    h, w = lena.shape
    size = max(lena.shape)
    resize = 512
    rate = size / resize

    # 补全成正方形
    lena = cv2.copyMakeBorder(lena, 0, size-h, 0, size-w, cv2.BORDER_CONSTANT,value=[0, 0, 0])

    # 缩放
    canny_dst = cv2.resize(lena, (resize, resize))
    # canny edge detection
    canny_dst = canny(canny_dst)

    # 膨胀
    # canny_dst = cv2.dilate(canny_dst, cv2.getStructuringElement(cv2.MORPH_RECT, (18, 8)))
    # canny_dst = cv2.dilate(canny_dst, cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10)))
    canny_dst = cv2.dilate(canny_dst, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    # _, canny_dst = cv2.threshold(canny_dst, 127, 255, cv2.THRESH_BINARY)

    # 形态学操作
    canny_dst = np.array(canny_dst, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    canny_dst = cv2.morphologyEx(canny_dst, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    # contours, _ = cv2.findContours(canny_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(canny_dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow('canny', canny_dst)
    print(len(contours))

    # 切成小图
    marks = []
    index = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x = int(x * rate)
        y = int(y * rate)
        w = int(w * rate)
        h = int(h * rate)
        # 小于一定值的就不要了
        if w < 20 or h < 20:
            continue
        index += 1
        marks.append((x, y, w, h))
        part = lena[y:y+h, x:x+w]
        partpath = './img/part_{}.jpg'.format(index)
        cv2.imwrite(partpath, part)

    # for x, y, w, h in marks:
    #     cv2.rectangle(lena, (x, y), (x+w, y+h), (70, 25, 124))
    # cv2.imwrite('./image_data/save.jpg', lena)

    # 画出轮廓
    # cv2.drawContours(canny_dst, contours, -1, (0, 255, 0), 1)
    # cv2.imshow('canny', canny_dst)
    # cv2.waitKey(0)
