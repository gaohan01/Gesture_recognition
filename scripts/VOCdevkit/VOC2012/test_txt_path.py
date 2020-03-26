import os


test_img_txt = '/home/gaohan/Project/darknet/scripts/VOCdevkit/VOC2012/ImageSets/Main/test_img.txt'
img_path_folder = '/home/gaohan/Project/darknet/scripts/VOCdevkit/VOC2012/after_change_name'
save_path_txt = '/home/gaohan/Project/darknet/scripts/test_path.txt'
f = open(test_img_txt, 'r')
fileNames = f.readlines()
for filename in fileNames:
    s = img_path_folder + '/' + filename[:6] + '.jpg'
    save_file = open(save_path_txt, 'a')
    save_file.write(s + '\n')

f.close()
save_file.close
