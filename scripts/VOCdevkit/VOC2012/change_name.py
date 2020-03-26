import os

test_folder = '/home/gaohan/Project/darknet/scripts/VOCdevkit/VOC2012/gesture_test_from_home'
save_folder = '/home/gaohan/Project/darknet/scripts/VOCdevkit/VOC2012/after_change_name'

fileNames = os.listdir(test_folder)
for i in fileNames:
    num = i.split('.')[0]
    change_num = num.zfill(6)
    change_name = change_num + '.jpg'
    print(change_name)
    os.rename(test_folder + '/' + i, save_folder + '/' + change_name)
