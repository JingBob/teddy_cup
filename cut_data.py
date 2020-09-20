import cv2
import xml.etree.ElementTree as ET
import xml.dom.minidom as xmldom
import glob
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage import draw, data
import matplotlib.pyplot as plt


FILE_PATH = 'raw/'
PATH_TRAIN = 'train/'
PATH_TEST = 'test/'
PATH_AUG = 'train/aug'

# overlap_half=True滑动窗口切图，每次有一半区域重叠，这时候x方向的步长就是窗口宽度的一半，
# y方向的步长是窗口高度的一半，stridex和stridey参数将不再起作用

def slide_crop(xmlpath,img,mask, kernelw, kernelh, num, overlap_half=True, stridex=0, stridey=0):
    height, width, _ = img.shape
    if overlap_half:
        stridex = kernelw / 2  # x方向步进为窗口的一半
        stridey = kernelh / 2  # y方向步进为窗口的一半
    stepx = int(width / stridex)   # 获取总x方向步进数目
    stepy = int(height / stridey)  # 获取总y方向步进数目
    bb = getbnd(xmlpath)
    for r in range(stepy-1):  # -1 视情况去掉
        startx = 0
        starty = r * stridey
        for c in range(stepx):
            num += 1
            startx = c*stridex
            endx = startx+kernelw
            endy = starty+kernelh
            if int(starty+kernelh) > height:
                endy = height
                starty = height - kernelh
            if int(startx+kernelw) > width:
                endx = width
                startx = width - kernelw
            image = img[int(starty):int(endy), int(startx):int(endx)]
            #masks = mask[int(starty):int(endy), int(startx):int(endx)]
            image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
            img_orgin = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
            #masks = cv2.resize(masks, (1024, 1024), interpolation=cv2.INTER_AREA)
            smallx = kernelw / 1024.0
            smally = kernelh / 1024.0
            flag = False
            if len(bb) != 0:
                for i in bb:
                    # 自爆区域的宽和高：
                    W = i[1]-i[0]
                    H = i[3]-i[2]
                    # 自爆区域中心坐标
                    x1 = (i[1]+i[0])/2
                    y1 = (i[2]+i[3])/2
                    # 切割区域中心坐标
                    x2 = (startx+endx)/2
                    y2 = (starty+endy)/2
                    # 此情况相交：
                    if (abs(x2-x1) <= kernelw/2) and (abs(y2-y1) <= kernelh/2):
                        flag = True
                        xc1 = (max(startx, i[0])-startx)/smallx
                        yc1 = (max(starty, i[2])-starty)/smally
                        xc2 = (min(endx, i[1])-startx)/smallx
                        yc2 = (min(endy, i[3])-starty)/smally
                        f = open("F:/Taidi_cup/taidi_demo/dataprocess/resize_1024/txt/%s.txt" % str(num).zfill(5), 'a+')
                        f.write(str(xc1) + ',' + str(xc2) + ',' + str(yc1) + ',' + str(yc2) + '\n')
                        f.close()
                    else:
                        continue
                    img_orgin = cv2.rectangle(img_orgin, (int(xc1), int(yc1)), (int(xc2), int(yc2)), (0, 255, 0), 10)
            if flag:
                #cv2.imwrite(FILE_PATH + "cut_bnd_mask/" + str(num).zfill(5) + ".jpg", masks)
                cv2.imwrite("F:/Taidi_cup/taidi_demo/dataprocess/resize_1024/img_with_rect/" + str(num).zfill(5) + ".jpg", img_orgin)
                cv2.imwrite("F:/Taidi_cup/taidi_demo/dataprocess/resize_1024/img_with_bnd/" + str(num).zfill(5) + ".jpg", image)
                #cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('result', 1000, 1000)
                #cv2.imshow("result", image)
                #cv2.waitKey(2000)
            else:
                cv2.imwrite("F:/Taidi_cup/taidi_demo/dataprocess/resize_1024/imgs_background/" + str(num).zfill(5) + ".jpg", image)
            #    cv2.imwrite(FILE_PATH + "cut_img/" + str(num).zfill(5) + ".jpg", image)
    return num


def getbnd(xmlpath):
    xmlfile = xmldom.parse(xmlpath)
    # 获取xml文件中的元素：
    RootNode = xmlfile.documentElement
    # 返回的是这个标签内的所有节点列表，是一个节点列表类
    subElement = RootNode.getElementsByTagName("item")
    bb = []
    if len(subElement) == 0:
        print("no obj")
    else:
        bnd = RootNode.getElementsByTagName("bndbox")
        for i in bnd:
            xmin = i.getElementsByTagName('xmin')
            ymin = i.getElementsByTagName('ymin')
            xmax = i.getElementsByTagName('xmax')
            ymax = i.getElementsByTagName('ymax')
            b = (float(xmin[0].childNodes[0].data), float(xmax[0].childNodes[0].data),
                 float(ymin[0].childNodes[0].data), float(ymax[0].childNodes[0].data))
            bb.append(b)
    return bb

xml_path = next(os.walk("raw/BoundingBox_xml"))[2]
for a in xml_path:
    bb = getbnd("raw/BoundingBox_xml/"+a)
    f = open(r"F:\Taidi_cup\taidi_demo\get_full_img\txt\\%s" % a.replace('xml', 'txt'), 'w')
    for k in bb:
        xc1, xc2, yc1, yc2 = k
        f.write(str(xc1) + ',' + str(xc2) + ',' + str(yc1) + ',' + str(yc2) + '\n')
    f.close()
'''
txt_path = next(os.walk(r"F:\Taidi_cup\taidi_demo\dataprocess\augbnd\myData3\labels\\"))[2]
fuyangben = 0
for a in txt_path:
    with open(r"F:\Taidi_cup\taidi_demo\dataprocess\augbnd\myData3\labels\\"+a, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            fuyangben += 1
    f.close()
print('负样本：' + str(fuyangben))
print('正样本：' + str(len(txt_path)-fuyangben))
print('总计:' + str(len(txt_path)))
'''
'''
txt_path = next(os.walk(r"F:\Taidi_cup\taidi_demo\bnd_txt\\"))[2]
for i in txt_path:
    with open(r"F:\Taidi_cup\taidi_demo\bnd_txt\\%s" % i, 'r') as f:
        line = f.readlines()
        if len(line) == 0:
            print(i)
        else:
            for l in line:
                print(l)
    f.close()
'''
'''
img_path = next(os.walk("raw/bonding_img"))[2]
xml_path = next(os.walk("raw/BoundingBox_xml"))[2]

for i in range(40):
    #img = cv2.imread("raw/bonding_img/"+img_path[i])
    #height, width, _ = img.shape
    #cut = min(height, width)
    bnd = getbnd("raw/BoundingBox_xml/"+xml_path[i])
    f = open(r"F:\Taidi_cup\taidi_demo\bnd_txt\\%s" % xml_path[i].replace('xml', 'txt'), 'w')
    for k in bnd:
        xc1, xc2, yc1, yc2 = k
        f.write(str(xc1) + ',' + str(xc2) + ',' + str(yc1) + ',' + str(yc2) + '\n')
    f.close()
        #img = cv2.rectangle(img, (int(xc1), int(yc1)), (int(xc2), int(yc2)), (0, 255, 0), 10)
    #cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('result', 1000, 1000)
    #cv2.imshow("result", img)
    #cv2.waitKey(2000)
'''

'''
img_path = next(os.walk("raw/bonding_img"))[2]
mask_path = next(os.walk("raw/mask"))[2]
xml_path = next(os.walk("raw/BoundingBox_xml"))[2]

n = 3199
for i in range(40):
    img = cv2.imread("raw/bonding_img/"+img_path[i])
    mask = cv2.imread("raw/mask/"+mask_path[i])
    #height, width, _ = img.shape
    #cut = min(height, width)
    n = slide_crop("raw/BoundingBox_xml/"+xml_path[i], img, mask,  1024, 1024, n)
    #print(mask_path[i])
#imglist = slide_crop(xmlp, img, 2048, 2048)
'''
#for im in imglist:
    #cv2.imshow(im)
    #im = cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_AREA)
    #cv2.imwrite(FILE_PATH + "cut_mask/" + str(i)+".png", im)
   # i += 1

'''
file_ids = next(os.walk(FILE_PATH+"mask"))[2]
for i in file_ids:
    img = cv2.imread(FILE_PATH+"mask/"+str(i))
    print(str(i))
    print(img.shape)
    
  
for i in file_ids:
    img = load_img(FILE_PATH+"mask/"+str(i))
    x_t = img_to_array(img)
    x = x_t[:, :, 0]
    cv2.imwrite(PATH_TEST + str(i), x)
'''
'''
path = "F:/Taidi_cup/taidi_demo/raw/bonding_img"
orin_img_name = next(os.walk(path))[2]
for name in orin_img_name:
    image = cv2.imread(path + '/' + name)
    name = name[:name.rindex(".")]
    bb = getbnd("raw/BoundingBox_xml/" + name + '.xml')
    print(bb)
    if len(bb) != 0:
        for i in bb:
            cv2.rectangle(image, (int(i[0]), int(i[2])), (int(i[1]), int(i[3])), (0, 255, 0), 10)
    cv2.imwrite("F:/Taidi_cup/taidi_demo/raw/aug/imgs/"+name+'.jpg', image)
    #cv2.namedWindow('result', 0)
    #cv2.resizeWindow('result', 1000, 1000)
    #cv2.imshow('result', image)
    #cv2.waitKey(3000)
'''

