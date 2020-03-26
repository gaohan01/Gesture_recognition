from voc_eval import voc_eval
 
#rec,prec,ap = voc_eval('/Your Path/results/{}.txt', '/Your Path/{}.xml', '/Your Path/test.txt', 'people', '.')
rec,prec,ap = voc_eval('/home/gaohan/Project/darknet/results/{}.txt', '/home/gaohan/Project/darknet/scripts/VOCdevkit/VOC2012/Annotations/{}.xml', '/home/gaohan/Project/darknet/scripts/VOCdevkit/VOC2012/ImageSets/Main/test_img.txt', 'fist', '.')
 
print('rec:') 
print(rec)
print('\n')
print('ptrc:') 
print(prec)
print('\n')
print('ap:')
print(ap)
