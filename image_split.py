import os

import cv2
import numpy as np

''' 
    用来分割干净背景的数据
    将一整张图片按照行切割成单个图片
'''


'''水平投影'''


def getHProjection(image):
    # np.zeros 返回来一个给定形状和类型的用0填充的数组；
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # 绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    # --------------------------------------- 绘画出原图像的水平投影 -------------------------------
    # cv2.imshow('hProjection2', hProjection)
    # print(h_)
    return h_

''' 进行垂直投影 ,返回每一列白色像素值的个数'''


def getVProjection(image):
    # np.zeros 返回来一个给定形状和类型的用0填充的数组；
    vProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # 绘制垂直平投影图像
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255
    # cv2.imshow('vProjection',vProjection)
    # cv2.waitKey()
    return w_

'''
    返回文件路径下图片的个数
'''
def get_img_num(exist_img_path):
    files = os.listdir(exist_img_path)  # 读入文件夹
    num_png = len(files)
    return num_png

'''
    origineImage_path: 读取图片的路径
    save_img_path：保存切割完的路径
'''
def cv2_cut_img(origineImage_path, save_img_path):
    # 读入原始图像
    '''
    imread函数有两个参数，第一个参数是图片路径，第二个参数表示读取图片的形式，有三种：
    cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1。
    cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。
    cv2.IMREAD_UNCHANGED：包括alpha，可以直接写-1
    cv2.imread()读取图片后已多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引，具体图像的通道数由图片的格式来决定

    'C:/Users/hy/Desktop/test/testpdf_big/images_0.jpg'
    '''
    origineImage = cv2.imread(origineImage_path)
    draw_origineImage = cv2.imread(origineImage_path)
    # 图像灰度化
    image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
    # --------------------------------------- 绘画出在原图像的灰度图像 -------------------------------
    # cv2.imshow('gray', image)
    # cv2.waitKey()
    # 将image图像转为黑白二值图，retval接收当前的阈值，img接收输出的二值图
    retval, img = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)
    # --------------------------------------- 绘画出二值化图像 -------------------------------
    # cv2.imshow('binary', img)
    # cv2.waitKey()

    kernel = np.ones((6, 6), dtype=int)
    dilate_img = cv2.dilate(img, kernel)
    # --------------------------------------- 绘画出膨胀后的二值化图像 -------------------------------
    # cv2.imshow('dilate_binary', dilate_img)
    # cv2.waitKey()
    # 图像高与宽
    (h, w) = dilate_img.shape
    Position = []
    # 水平投影
    H = getHProjection(dilate_img)

    start = 0
    num = 0
    H_Start = []
    H_End = []
    # 根据水平投影获取垂直分割位置
    for i in range(len(H)):
        if H[i] > 0 and start == 0:
            H_Start.append(i)
            start = 1
        if H[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0

    # 分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_Start)):
        # 获取行图像
        cropImg = dilate_img[H_Start[i]:H_End[i], 0:w]
        # --------------------------------------- 绘画出膨胀后二值化图像的切割图像 -------------------------------
        # cv2.imshow('cropImg',cropImg)
        # cv2.waitKey()

        # 对行图像进行垂直投影
        W = getVProjection(cropImg)
        Wstart = 0
        Wend = 0
        W_Start = 0
        W_End = 0
        for j in range(len(W)):
            if W[j] > 0 and Wstart == 0:
                W_Start = j
                Wstart = 1
                Wend = 0
            if W[j] <= 0 and Wstart == 1:
                W_End = j
                Wstart = 0
                Wend = 1
            if Wend == 1:
                Position.append([W_Start, H_Start[i], W_End, H_End[i]])
                Wend = 0
    #    根据确定的位置分割字符,,对比一下rectangle里面的参数以及img[]的参数
    next_imgName = get_img_num(save_img_path)
    print('next_imgName:', next_imgName)
    for m in range(len(Position)):
        # 根据投影的像素位置，给原始图片划线框出来要切割的位置
        cv2.rectangle(draw_origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]),
                      (0, 229, 238), 1)
        x1 = Position[m][0]
        x2 = Position[m][1]
        x3 = Position[m][2]
        x4 = Position[m][3]
        cut_img = origineImage[x2:x4, x1:x3]
        # --------------------------------------- 绘画出在原图像切出的图像 -------------------------------
        # cv2.imshow('cut_image', cut_img)
        # cv2.waitKey(0)
        img_name_end = 'orig_img_' + str(next_imgName+m) + '.jpg'
        save_img = os.path.realpath(os.path.join(save_img_path, img_name_end))
        cv2.imwrite(save_img, cut_img)

    # --------------------------------------- 绘画出在原图像框出文字的图像,并保存下来 -------------------------------
    # cv2.imshow('image', draw_origineImage)
    # cv2.imwrite(origineImage_path, draw_origineImage)
    # cv2.waitKey(0)

'''
    功能：file_dir---读取文件夹下的所有图片名字以及图片的完整路径
    返回：jpg格式的图片绝对路径
    'C:/Users/hy/Desktop/test/testpdf_big/'
    os.walk(file_dir):返回root，dirs，files
'''
def pdf_exchange_img_file(file_dir):
    img_path = os.listdir(file_dir)
    # img_path.remove('.DS_Store')
    img_path.sort(key=lambda x: int(x.split('.')[0]))
    # print(img_path)
    full_imgPath = []
    for path in img_path:
        full_imgPath.append(os.path.join(file_dir, path))
    print(full_imgPath)

    return full_imgPath

if __name__ == "__main__":

    '''
        full_imgPath:拿到文件夹下一整张图片的所有路径
        save_img_path：将full_imgPath里面的图片切割成一个个小的图片
    '''
    full_imgPath = pdf_exchange_img_file('E:/tibet_print/pdf_exchange_img/Qomolangma_SubTitle/yuyi_dataset_0911/only_tibet_yuyi_train')
    #E:/tibet_print/pdf_exchange_img/Himalaya_Font/only_tibet_yuyi_val
    # 'E:/tibet_print/pdf_exchange_img/Qomolangma_SubTitle/yuyi_dataset_0911/only_tibet_yuyi_train'
    # exist_img_path = 'E:/tibet_print/tibet_print_dataset/mydateset/dataset'
    save_img_path = 'E:/tibet_print/tibet_print_dataset/Qomolangma_SubTitle/yuyi_dataset_0911/train'
                    # 'E:/tibet_print/tibet_print_dataset/Himalaya_Font/only_tibet_yuyi/dataset'
        # 'E:/tibet_print/tibet_print_dataset/Qomolangma_SubTitle/yuyi_dataset_0911/train'
    for img_path in full_imgPath:
        cv2_cut_img(img_path,save_img_path )
