from ctypes import *
import math
import random
import time
import numpy as np
import cv2
import os
import sys
 
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1
 
def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr
 
class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]
 
class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]
 
 
class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]
 
class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]
 
    
 
#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int
 
predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)
 
set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]
 
make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE
 
get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)
 
make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)
 
free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]
 
free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]
 
network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]
 
reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]
 
load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p
 
do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
 
do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
 
free_image = lib.free_image
free_image.argtypes = [IMAGE]
 
letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE
 
load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA
 
load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE
 
# 添加以处理视频
ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE
 
rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]
 
predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)
 
def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res
 
"""
Yolo-v3目前耗时过长的步骤
    1.输入图像的预处理阶段
    2.python接口调用网络执行一次推理过程
"""
 
def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    # preprocess_image_time = time.time()
    # 大约0.1131s
    im = load_image(image, 0, 0)
    # print("Yolo Preprocess image time in python version:", (time.time() - preprocess_image_time))
    num = c_int(0)
    pnum = pointer(num)
    # start_time = time.time()
    # 大概0.129秒左右
    predict_image(net, im)
    # print("Yolo Do inference time in python version:", (time.time() - start_time))
    
    # get_detection_time = time.time()
    # 大约0.0022s
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    # print("Yolo Get detections time in python version:", (time.time() - get_detection_time))
    num = pnum[0]
    # do_nms_time = time.time()
    # 可以忽略不计
    if (nms): do_nms_obj(dets, num, meta.classes, nms)
    # print("Yolo Do nms time in python version:", (time.time() - do_nms_time))
 
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
 
# 添加以处理视频
def detect_im(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    # to_image_time = time.time()
    # 大约0.0012~0.0013秒
    im, image = array_to_image(im)
    # print("to_image time:", (time.time() - to_image_time))
    # rgbgr_image_time = time.time()
    # 大约0.0013秒
    rgbgr_image(im)
    # print("rgbgr_image time:", (time.time() - rgbgr_image_time))
    num = c_int(0)
    pnum = pointer(num)
    # do_inference_time = time.time()
    # 大约0.083秒
    predict_image(net, im)
    # print("Do inference time:", (time.time() - do_inference_time))
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)
 
    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                           (b.x, b.y, b.w, b.h)))
 
    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes):
        free_image(im)
    free_detections(dets, num)
 
    return res
 
def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr
 
def get_folderImages(folder):
    all_files = os.listdir(folder)
    abs_path = [os.path.join(folder, i) for i in all_files]
    return abs_path
 
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
 
def init():
    net = load_net("../cfg/yolov3.cfg".encode("utf-8"), "../cfg/yolov3.weights".encode("utf-8"), 0)
    meta = load_meta("../cfg/coco.data".encode("utf-8"))
    return net, meta
 
def image_processing():
    net, meta = init()
 
    folder = "images"
    save_folder = "results"
    each_process_time = []
 
    for image_path in get_folderImages(folder):
        image = cv2.imread(image_path)
        start_time = time.time()
        r = detect(net, meta, image_path.encode("utf-8"))
        processing_time = time.time() - start_time
        each_process_time.append(processing_time)
        for i in range(len(r)):
            x, y, w, h = r[i][2][0], r[i][2][1], r[i][2][2], r[i][2][3]
            topleft, topright, bottomleft, bottomright = convertBack(float(x), float(y), float(w), float(h))
            result = cv2.rectangle(
                image,
                (topleft, topright),
                (bottomleft, bottomright),
                (0, 255, 255),
                2
            )
            cv2.putText(
                result, 
                bytes.decode(r[i][0]), 
                (topleft, topright),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 0, 255), 
                2
            )
        save_path = os.path.join(save_folder, image_path.split('/')[-1].split(".jpg")[0] + "-result.jpg")
        cv2.imwrite(save_path, result)
    average_processing_time = np.mean(each_process_time)
    print("Yolo-v3 COCO Average each Image processing Time:\n")
    print(average_processing_time)
 
def video_processing():
    set_gpu(7)
    net, meta = init()
 
    processing_path = "small.mp4"
    cam = cv2.VideoCapture(processing_path)
    total_frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cam.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fourcc = int(cam.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    processing_result_name = processing_path.split(".mp4")[0] + "-result.mp4"
    result = cv2.VideoWriter(processing_result_name, fourcc, fps, frame_size)
        
    timeF = 1
    c = 1
    print("opencv?", cam.isOpened())
    print("fps:", fps)
    print("decode style:", fourcc)
    print("size:", frame_size)
    print("total frames:", total_frames)
    start_total = time.time()
    while True:
        frame_start = time.time()
        _, img = cam.read()
        if (c % timeF == 0 or c == total_frames):
            if img is not None:
                r = detect_im(net, meta, img)
                for i in range(len(r)):
                    x, y, w, h = r[i][2][0], r[i][2][1], r[i][2][2], r[i][2][3]
                    topleft, topright, bottomleft, bottomright = convertBack(float(x), float(y), float(w), float(h))
                    img = cv2.rectangle(
                        img,
                        (topleft, topright),
                        (bottomleft, bottomright),
                        (0, 255, 255),
                        1
                    )
                    label_score = "{}:{:.2f}".format(bytes.decode(r[i][0]), r[i][1])
                    cv2.putText(
                        img, 
                        label_score, 
                        (topleft, topright),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, 
                        (0, 0, 255), 
                        1
                    )
                result.write(img)
        else:
            result.write(img)
 
        c += 1
 
        if c > total_frames:
            print("Finished Processing!")
            break
        print("processing one frame total time:", (time.time() - frame_start))
        print()
        
    processing_time = time.time() - start_total
    cam.release()
    result.release()
    post_compression(processing_result_name)
    print("Yolo-v3 COCO one Video Process Time:\n")
    print(processing_time)
 
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    # net = load_net("../cfg/yolov3.cfg".encode("utf-8"), "../cfg/yolov3.weights".encode("utf-8"), 0)
    # meta = load_meta("../cfg/coco.data".encode("utf-8"))
    # start_time = time.time()
    # r = detect(net, meta, "../data/car.jpg".encode("utf-8"))
    # print("Inference time:{:.4f}".format(time.time() - start_time))
    # print(r)
    image_processing()
    # video_processing()
