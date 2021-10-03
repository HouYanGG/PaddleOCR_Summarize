import random

import cv2
import numpy as np


from ppocr.data.imaug.rec_img_aug import flag
from ppocr.data.imaug.text_image_aug import tia_distort, tia_stretch, tia_perspective


def jitter(img):
    """
    jitter
    s = int(1 * thres * 0.01)
    官方参数为0.01
    个人觉得这个没必要加，没看出来有啥效果，觉得是图片文字下移
    """
    w, h, _ = img.shape
    print(w, h)
    if h > 10 and w > 10:
        thres = min(w, h)
        print("thres:", thres)
        s = int(1 * thres * 0.1)
        print("s:", s)
        src_img = img.copy()
        # print('src_img', src_img)
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        # print('img', img)
        return img
    else:
        return img

def cvtColor(img):
    """
    cvtColor
    可适当调整0.002-0.005,官方为0.001
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)
    delta = 0.002 * 1 * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print(new_img.shape)
    return new_img

def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img

def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    var**0.5: var开根号
    个人觉得没用
    """

    noise = np.random.normal(mean, var**0.5, image.shape)
    print(noise)
    out = image + 0.5 * noise
    # cilp:限制out数组的数值在剩余两个参数之间
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out

def get_crop(image):
    """
    random crop
    重写了裁剪策略
    """

    h, w, _ = image.shape
    top_min = 1
    top_max = 4
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    ratio = random.randint(0, 1)
    two_dim_array = image[0:2, ]
    num = np.sum(two_dim_array < 200)
    print('num:', num)
    if ratio:
        if num > 120:
            crop_img = crop_img[0:h - top_crop, :, :]
        else:
            crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img

def distort_img(img):
    new_img = tia_distort(img, random.randint(3, 6))
    return new_img

'''没用'''
def stretch(img):
    new_img = tia_stretch(img, random.randint(3, 6))
    return new_img

def perspective(new_img):
    perspective_img = tia_perspective(new_img)
    return perspective_img

def reverse(img):
    new_img = 255 - img
    return new_img
if __name__ == '__main__':
    img = cv2.imread('/home/hy/Pictures/youzheng.jpg')
    cv2.imshow('yuantu ', img)
    cv2.waitKey(0)
    # ------------------------------------------------------------------------------------------
    # jitter_img = jitter(img)
    # cv2.imshow('jitter later', jitter_img)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------------
    # cvtColor_img = cvtColor(img)
    # cv2.imshow('cvtColor later', cvtColor_img)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------------
    # blur_img = blur(img)
    # cv2.imshow('blur_img ', blur_img)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------------
    # add_gasuss_noise_img = add_gasuss_noise(img)
    # cv2.imshow('add_gasuss_noise_img ', add_gasuss_noise_img)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------------
    # get_crop_img = get_crop(img)
    # cv2.imshow('get_crop_img ', get_crop_img)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------------
    # distort_imgs = distort_img(img)
    # cv2.imshow('distort_imgs ', distort_imgs)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------------
    # stretch_img = stretch(img)
    # cv2.imshow('stretch_img ', stretch_img)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------------
    # perspective_img = perspective(img)
    # cv2.imshow('perspective_img ', perspective_img)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------------
    # reverse_img = reverse(img)
    # cv2.imshow('reverse_img', reverse_img)
    # cv2.waitKey(0)

