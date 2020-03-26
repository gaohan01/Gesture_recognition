import os
import random

trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = 'after_change_name'
txtsavepath = 'ImageSets\Main'
ftest = open('ImageSets/Main/test_img.txt', 'w')

total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)

for i in list:
    name = total_xml[i][:-4] + '\n'
    ftest.write(name)

ftest.close()
