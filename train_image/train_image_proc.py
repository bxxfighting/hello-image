import os
import cv2
import numpy as np
from random import choice
from random import randint
from PIL import Image


def pil_image_rotate(pil_img):
    '''
    PIL图片旋转
    我们旋转限定在向左或向右旋转最多30度
    '''
    # 这里先进行一下选择，1为不旋转，2为向左路旋转，3为向右旋转
    c = choice((1, 2, 3))
    angle = 0
    if c == 2:
        # 因为Image.rotate是逆时针旋转，所以向左就是0到30
        angle = randint(0, 30)
    elif c == 3:
        # 向右旋转
        angle = randint(330, 360)
    # 因为旋转后，图片如果保持原来大小，那么必然有部分图像会最截掉
    # 因此设置expand=True，让图片根据旋转会的样式扩展
    return pil_img.rotate(angle, expand=True)


def pil_image_repair_square_and_white_bg(pil_img):
    '''
    PIL图片补全成正方形并填充白色背景
    '''
    size = max(pil_img.size)
    img = Image.new('RGB', (size, size), (255, 255, 255))
    img.paste(pil_img, (0, 0, pil_img.size[0], pil_img.size[1]), pil_img)
    return img


def pil_image_valid_region(pil_img):
    '''
    获取PIL图片有效区域
    此方法处理的图片应该是经过背景填充及补全成正方形的
    '''
    # 因为经过白色背景填充后，背景值都是255，
    # 那么其它小于255的地方说明有真正的图像
    # 因此就根据这个原理，找到真正图像在整个图片上的位置
    img = np.array(pil_img)
    xs = np.where(img[:,:,0]<255)[1]
    ys = np.where(img[:,:,0]<255)[0]
    # x坐标
    x = min(xs)
    # y坐标
    y = min(ys)
    # 宽
    w = max(xs) - x
    # 高
    h = max(ys) - y
    return (x, y, w, h)


def pil_image_translation(pil_img):
    '''
    PIL图片平移
    '''
    # 图片的宽高
    img_w, img_h = pil_img.size
    # 图片实际图像区域
    x, y, w, h = pil_image_valid_region(pil_img)
    # 将实际图像区域切割出来
    valid_img = pil_img.crop((x, y, x+w, y+h))
    # 随机生成实际图像移动后的起点
    move_x = randint(0, img_w - w)
    move_y = randint(0, img_h - h)
    # 重新生成一个Image，将切割出来的图像粘贴到对应的点上
    img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
    img.paste(valid_img, (move_x, move_y))
    return img


def pil_image_resize(pil_img, size):
    '''
    PIL图片重置大小
    这里只是处理正方形的，
    如果需要处理长方形，只要进行一下等比处理就可以
    '''
    return pil_img.resize((size, size))


def pil_image_2_cv2_image(pil_img):
    '''
    PIL图片转cv2图片(其实就是numpy的array)
    '''
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def preprocessor_image(imagepath):
    '''
    根据要求预处理图片
    '''
    basename = os.path.basename(imagepath)
    path = os.path.dirname(imagepath)

    # 先读取图片
    pil_img = Image.open(imagepath)
    # 图片随机旋转
    pil_img = pil_image_rotate(pil_img)
    pil_img.save('{}/旋转.png'.format(path))
    # 图片补全成正方形并填充白色背景
    pil_img = pil_image_repair_square_and_white_bg(pil_img)
    pil_img.save('{}/白色补全.png'.format(path))
    # 图片随机平移
    pil_img = pil_image_translation(pil_img)
    pil_img.save('{}/平移.png'.format(path))
    # 图片调整到128 * 128
    pil_img = pil_image_resize(pil_img, 128)
    pil_img.save('{}/128.png'.format(path))
    # PIL图片转cv2图片
    cv2_img = pil_image_2_cv2_image(pil_img)
    cv2.imwrite('{}/cv2.jpg'.format(path), cv2_img)
    # 之后如果有其它处理就可以进行其它处理了
    # 我们是将cv2_img写到了h5文件中(h5py)


if __name__ == '__main__':
    preprocessor_image('data/test.png')
