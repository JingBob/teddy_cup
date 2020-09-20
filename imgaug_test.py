#!/usr/bin/env python
# coding: utf-8
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
from skimage.io import imread, imshow, imsave
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from matplotlib import pyplot as plt
sometimes = lambda aug: iaa.Sometimes(0.6, aug)  # 50%的图片执行传入的增强方法
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import cv2

RAW_IMG_PATH = r"F:\Taidi_cup\taidi_demo\dataprocess\resize_1024\img_with_bnd\\"
RAW_MASK_PATH = r"F:\Taidi_cup\taidi_demo\dataprocess\resize_1024\img_with_bnd\\"
TXT_PATH = r"F:\Taidi_cup\taidi_demo\dataprocess\resize_1024\txt\\"

AUG_IMG_PATH = r"F:\Taidi_cup\taidi_demo\dataprocess\myData3\imgs\\"
AUG_MASK_PATH = r"F:\Taidi_cup\taidi_demo\dataprocess\myData3\mask\\"
AUG_TXT_PATH = r"F:\Taidi_cup\taidi_demo\dataprocess\myData3\label\\"


if(not os.path.exists(AUG_IMG_PATH)):
    os.makedirs(AUG_IMG_PATH)
    print("未发现", AUG_IMG_PATH, "\n 已创建！")
if(not os.path.exists(AUG_MASK_PATH)):
    os.makedirs(AUG_MASK_PATH)
    print("未发现", AUG_MASK_PATH, "\n 已创建！")
    
raw_img_name = next(os.walk(RAW_IMG_PATH))[2]
print("加载原图完成,共发现", len(raw_img_name), "张原图")
raw_mask_name = next(os.walk(RAW_MASK_PATH))[2]
print("加载掩膜完成,共发现", len(raw_img_name), "张掩膜")
txt_name = next(os.walk(TXT_PATH))[2]
print("加载txt完成,共发现", len(txt_name), "个文档")


#print(len(next(os.walk(r"F:\Taidi_cup\taidi_demo\dataprocess\augbnd\myData2\labels"))[2]))
#print(len(next(os.walk(r"F:\Taidi_cup\taidi_demo\dataprocess\augbnd\myData2\JPEGImages"))[2]))


'''
# 生成 yolo 格式txt
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

img_name = next(os.walk(AUG_IMG_PATH))[2]
antxt = next(os.walk(AUG_TXT_PATH))[2]
for im in tqdm(img_name, total=len(img_name)):
    if im.replace('jpg', 'txt') in antxt:
        with open(AUG_TXT_PATH + im.replace('jpg', 'txt'), 'r') as f:
            line = f.readline()
            while line:
                p = [float(i) for i in line.split(',')]
                b = (p[0], p[1], p[2], p[3])
                bb = convert((1024, 1024), b)
                f2 = open("F:/Taidi_cup/taidi_demo/dataprocess/myData3/labels/" + "%s" % im.replace('jpg', 'txt'), 'a+')
                f2.write(str(0) + " " + " ".join([str(a) for a in bb]) + '\n')
                f2.close()
                line = f.readline()
        f.close()
    else:
        f2 = open("F:/Taidi_cup/taidi_demo/dataprocess/myData3/labels/" + "%s" % im.replace('jpg', 'txt'), 'a+')
        f2.close()
'''
'''
nn = next(os.walk(AUG_MASK_PATH))[2]
nm = next(os.walk(AUG_IMG_PATH))[2]
nx = next(os.walk(AUG_TXT_PATH))[2]
for nk in nx:
    if nk.replace('txt', 'jpg') not in nn:
        os.remove(AUG_TXT_PATH + nk)
        print(nk)
for nj in nm:
    if nj not in nn:
        os.remove(AUG_IMG_PATH + nj)
'''

# 创建增强模板
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.3),  # 对30%的图像进行镜像翻转
        iaa.Flipud(0.3),  # 对30%的图像做左右翻转
        sometimes(iaa.CropAndPad(px=None,
               percent=(0, 0.3),
               pad_mode='constant',
               pad_cval=0,
               keep_size=True,   # 在crop或者pad后再缩放成原来的大小。
               sample_independently=True,
               deterministic=False,
               random_state=True)),   # 随机裁剪幅度为0.3

        sometimes(iaa.Affine(                          # 对一部分图像做仿射变换
            scale={"x": (0.8, 1.4), "y": (0.8, 1.4)},  # 图像缩放为50%到150%之间
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # 平移±20%之间
            rotate=(-30, 30),   # 旋转±45度之间
            shear=(-16, 16),    # 剪切变换±16度，（矩形变平行四边形）
            order=[0, 1],       # 使用最邻近差值或者双线性差值
            cval=0,             # 全黑填充
            mode="constant"     # 定义填充图像外区域的方法
        )),

        # 使用下面的0个到1个之间的方法去增强图像。注意SomeOf的用法
        # iaa.SomeOf((0, 1),
          #  [
                # 将部分图像进行超像素的表示。o(╥﹏╥)o用超像素增强作者还是第一次见，比较孤陋寡闻
                #sometimes(
                #    iaa.Superpixels(
                 #       p_replace=(0, 1.0),
                 #       n_segments=(20, 200)
                 #   )
               # ),
                # 锐化处理
             #   iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                # 浮雕效果
                # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                # 边缘检测，将检测到的赋值0或者255然后叠在原图上
                #sometimes(iaa.OneOf([
                 #   iaa.EdgeDetect(alpha=(0, 0.7)),
                  #  iaa.DirectedEdgeDetect(
                  #      alpha=(0, 0.7), direction=(0.0, 1.0)
                 #   ),
                #])),
                # 用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                # iaa.OneOf([
                #    iaa.GaussianBlur((0, 3.0)),
                #    iaa.AverageBlur(k=(2, 7)), # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                #    iaa.MedianBlur(k=(3, 11)),
                # ]),
                #iaa.GaussianBlur((0, 1.0)),     #在模型上使用0均值1方差进行高斯模糊

                # 加入高斯噪声
                #iaa.AdditiveGaussianNoise(
                #    loc=0, scale=(0.0, 0.01*255), per_channel=0.2
                #),
                # 将整个图像的对比度变为原来的0.3或者1.5倍
                # iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),
                # 5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
                #iaa.Invert(0.05, per_channel=True),
                # 像素乘上0.5或者1.5之间的数字.
                # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                # 扭曲图像的局部区域
                # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01)))
          #  ],
          #  random_order=True  # 随机的顺序把这些操作用在图像上
        #)
    ],
    random_order=True  # 随机的顺序把这些操作用在图像上
)

ia.seed(1)
image = imread(r"F:\Taidi_cup\taidi_demo\raw\cut_bnd_img\00011.jpg")
bd = []
flag = True
with open(r"F:\Taidi_cup\taidi_demo\raw\txt\00011.txt", 'r') as f:
    lines = f.readlines()
    if len(lines) != 0:
        for line in lines:
            p = [float(i) for i in line.split(',')]
            bd.append(BoundingBox(x1=p[0], y1=p[2], x2=p[1], y2=p[3]))
            line = f.readline()
    else:
        flag = False
f.close()
if flag:
    bbs = BoundingBoxesOnImage(bd, shape=image.shape)
    for i in range(1, 2):  # 每张图片增强8次

        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
        image_after = bbs_aug.draw_on_image(image_aug, size=10, color=[0, 0, 255])
        # imsave(AUG_MASK_PATH + str(num_name).zfill(5) + ".jpg", image_after)
        # imsave(AUG_IMG_PATH + str(num_name).zfill(5) + ".jpg", image_aug)

        if len(bbs_aug) == 0:
            continue
        else:
            for j in range(len(bbs_aug)):
                after = bbs_aug.bounding_boxes[j]
        imshow(image_after)
        plt.show()

def do_imgaug(num):
    ia.seed(1)
    new_id = 1000
    for id_ in tqdm(range(num), total=num):
        new_id += 1
        id_ = id_ % (len(raw_img_name)-1)
        image = imread(RAW_IMG_PATH + raw_img_name[id_])
        segmap = imread(RAW_MASK_PATH + raw_mask_name[id_])
        segmap = SegmentationMapOnImage(segmap, nb_classes=2, shape=image.shape)
        seq_det = seq.to_deterministic()

        imsave(AUG_IMG_PATH + str(new_id) + ".png", seq_det.augment_image(image))
        imsave(AUG_MASK_PATH + str(new_id) + ".png", seq_det.augment_segmentation_maps(segmap).get_arr())


'''
# 增强整张图
back_img_name = next(os.walk(r"F:\Taidi_cup\taidi_demo\raw\bonding_img"))[2]
num_name = 3019
for background in back_img_name:
    ia.seed(1)
    image = imread(r"F:\Taidi_cup\taidi_demo\raw\bonding_img\\" + background)
    bd = []
    flag = True
    with open(r"F:\Taidi_cup\taidi_demo\get_full_img\txt\\" + background[:background.index('.')] +'.txt', 'r') as f:
        lines = f.readlines()
        if len(lines) != 0:
            for line in lines:
                p = [float(i) for i in line.split(',')]
                bd.append(BoundingBox(x1=p[0], y1=p[2], x2=p[1], y2=p[3]))
                line = f.readline()
        else:
            flag = False
    f.close()
    if flag:
        bbs = BoundingBoxesOnImage(bd, shape=image.shape)

        for i in range(1, 3):  # 每张图片增强8次
            num_name += 1
            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
            bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
            image_after = bbs_aug.draw_on_image(image_aug, size=10, color=[0, 0, 255])
            #imsave(AUG_MASK_PATH + str(num_name).zfill(5) + ".jpg", image_after)
            #imsave(AUG_IMG_PATH + str(num_name).zfill(5) + ".jpg", image_aug)

            if len(bbs_aug) == 0:
                continue
            else:
                for j in range(len(bbs_aug)):
                    # before = bbs.bounding_boxes[j]
                    after = bbs_aug.bounding_boxes[j]
                    #f = open(AUG_TXT_PATH + "%s.txt" % str(num_name).zfill(5), 'a+')
                    #f.write(str(after.x1) + ',' + str(after.x2) + ',' + str(after.y1) + ',' + str(after.y2) + '\n')
                    #f.close()

            #imshow(image_after)
            #plt.show()
    else:
        for i in range(1, 3):  # 每张图片增强8次
            num_name += 1
            image_aug = seq(image=image)
            #imsave(AUG_IMG_PATH + str(num_name).zfill(5) + ".jpg", image_aug)

            #f = open(AUG_TXT_PATH + "%s.txt" % str(num_name).zfill(5), 'a+')
            #f.write(str(after.x1) + ',' + str(after.x2) + ',' + str(after.y1) + ',' + str(after.y2) + '\n')
            #f.close()
            #imshow(image_after)
            #plt.show()
'''

'''
# 增强背景
back_img_name = next(os.walk(RAW_IMG_PATH))[2]
num_name = 3019
for background in back_img_name:
    if background.replace('jpg', 'txt') in txt_name:  # 只增强包含方框的
        continue
    else:
        ia.seed(3)
        bkimage = imread(RAW_IMG_PATH + background)
        for j in range(10):
            num_name += 1
            #print(num_name)
            image_aug = seq(image=bkimage)
            #imshow(image_aug)
            #plt.show()
            imsave(AUG_IMG_PATH + str(num_name).zfill(5) + ".jpg", image_aug)

# total = next(os.walk("F:/Taidi_cup/taidi_demo/dataprocess/augbnd/myData/JPEGImages"))[2]
# positive = next(os.walk("F:/Taidi_cup/taidi_demo/dataprocess/augbnd/myData/label"))[2]
# print(len(total), len(positive))
'''


'''
# 增强包含自爆区域图片
# do_imgaug(100)  # 生成100张
num_name = 3199  # 图片起始编号
for name in tqdm(raw_img_name, total=len(raw_img_name)):
    ia.seed(1)
    image = imread(RAW_IMG_PATH + name)
    if name.replace('jpg', 'txt') not in txt_name:  # 只增强包含方框的
        continue
    bd = []
    with open(TXT_PATH + name.replace('jpg', 'txt'), 'r') as f:
        line = f.readline()
        while line:
            p = [float(i) for i in line.split(',')]
            bd.append(BoundingBox(x1=p[0], y1=p[2], x2=p[1], y2=p[3]))
            line = f.readline()
    f.close()
    
    bbs = BoundingBoxesOnImage(bd, shape=image.shape)
    
    for i in range(1, 9):  # 每张图片增强8次
        num_name += 1
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
        image_after = bbs_aug.draw_on_image(image_aug, size=10, color=[0, 0, 255])
        imsave(AUG_MASK_PATH + str(num_name).zfill(5) + ".jpg", image_after)
        imsave(AUG_IMG_PATH + str(num_name).zfill(5) + ".jpg", image_aug)

        if len(bbs_aug) == 0:
            continue
        else:
            for j in range(len(bbs_aug)):
                #before = bbs.bounding_boxes[j]
                after = bbs_aug.bounding_boxes[j]
                #print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                #    j,
                 #   before.x1, before.y1, before.x2, before.y2,
                 #   after.x1, after.y1, after.x2, after.y2))

                f = open(AUG_TXT_PATH+"%s.txt" % str(num_name).zfill(5), 'a+')
                f.write(str(after.x1) + ',' + str(after.x2) + ',' + str(after.y1) + ',' + str(after.y2) + '\n')
                f.close()
    
        # imshow(image_after)
        # plt.show()
'''
# imsave("example_segmaps.jpg", seq_det.augment_image(image))

# plt.figure(2)
# imshow(seq_det.augment_segmentation_maps(segmap).get_arr())
# plt.show()
'''
imsave(AUG_IMG_PATH + str(1) + ".png", seq_det.augment_image(image))
imsave(AUG_MASK_PATH + str(1) + ".png",seq_det.augment_segmentation_maps(segmap).get_arr())

'''

