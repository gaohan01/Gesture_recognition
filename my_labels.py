import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
 
#源代码sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2012', 'train')]  # 改成自己建立的myData
 
classes = ["fist", "OK", "palm", "silence", "slide"] # 改成自己的类别
 
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(image_id))  # 源代码VOCdevkit/VOC%s/Annotations/%s.xml
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(image_id), 'w')  # 源代码VOCdevkit/VOC%s/labels/%s.txt
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
wd = getcwd()
 
for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC2012/labels/'):  # 改成自己建立的myData
        os.makedirs('myData/labels/')
    image_ids = open('myData/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('myData/%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/myData/JPEGImages/%s.jpg\n'%(wd, image_id))
        convert_annotation(year, image_id)
    list_file.close()
