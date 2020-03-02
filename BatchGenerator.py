#-- coding: utf-8 --

import numpy as np
import os
import cv2
from xml.etree.ElementTree import parse
from random import shuffle
class BatchGenerator():
    def __init__(self,ann_fnames,img_dir,label_names,batch_size,jitter=True,shuffle=True):
        self.ann_fnames= ann_fnames
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = len(label_names)
        self.jitter = jitter
        self.shuffle = shuffle
        self.label_names = label_names
        self.index = 0
        pass

    def get(self):
        #解析标注文件
        fname, boxes, coded_labels = parse_annotation(self.ann_fnames[self.index],self.img_dir,self.label_names)
        #读取图片，并按照设置修改图片尺寸

        image = cv2.imread(fname) #返回（高度，宽度，通道数）的元组
        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        image = image - pixel_means
        boxes_ = np.copy(boxes)
        if self.jitter:#是否要增强数据
            #image, boxes_ = make_jitter_on_image(image,boxes_)
            pass

        images, imgs_info, ground_truth_boxes = preprocessing(image,boxes_,coded_labels) #对原始图片进行缩放，并且重新计算True box的位置坐标

        #boxes_为[x1,y1,x2,y2]
        self.index = self.index + 1
        if self.index == len(self.ann_fnames):
            self.index = 0
            shuffle(self.ann_fnames)
            pass

        return images, imgs_info, ground_truth_boxes
        pass
    pass


class PascalVocXmlParser(object):
    def __init__(self):
        pass

    def get_fname(self, annotation_file):
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_width(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'width' in elem.tag:
                return int(elem.text)

    def get_height(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'height' in elem.tag:
                return int(elem.text)

    def get_labels(self, annotation_file):
        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels

    def get_boxes(self, annotation_file):
        root = self._root_tag(annotation_file)
        bbs = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))])
            bbs.append(box)
        bbs = np.array(bbs)
        return bbs

    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root

    def _tree(self, fname):
        tree = parse(fname)
        return tree
    pass


class Annotation(object):
    def __init__(self, filename):
        self.fname = filename
        self.labels = []
        self.coded_labels = []
        self.boxes = None
        pass

    def add_object(self, x1, y1, x2, y2, name, code):
        self.labels.append(name)
        self.coded_labels.append(code)

        if self.boxes is None:
            self.boxes = np.array([x1, y1, x2, y2]).reshape(-1, 4)
        else:
            box = np.array([x1, y1, x2, y2]).reshape(-1, 4)
            self.boxes = np.concatenate([self.boxes, box])
        pass

    pass

def parse_annotation(ann_fname, img_dir, labels_name=[]):
    parser = PascalVocXmlParser()
    fname = parser.get_fname(ann_fname)

    annotation = Annotation(os.path.join(img_dir, fname))

    labels = parser.get_labels(ann_fname)
    boxes = parser.get_boxes(ann_fname)

    for label, box in zip(labels, boxes):
        x1, y1, x2, y2 = box
        if label in labels_name:
            annotation.add_object(x1, y1, x2, y2, name=label, code=labels_name.index(label))
    return annotation.fname, annotation.boxes, annotation.coded_labels
    pass

def preprocessing(img,boxes,labels):
    """Converts an image into a network input.
       Arguments:
         img (ndarray): a color image in BGR order
       Returns:
         blob (ndarray): a data blob holding an image pyramid
         im_scale_factors (list): list of image scales (relative to img) used
           in the image pyramid
    """
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
    min_size = 600
    max_size = 1000
    img = img - pixel_means
    img_shape = img.shape
    img_size_min = np.min(img_shape[0:2])
    img_size_max = np.max(img_shape[0:2])
    img_scale = float(min_size) / float(img_size_min)
    #Prevent the biggest axis from being more than MAX_SIZE
    if np.round(img_scale * img_size_max) > max_size:
        img_scale = float(max_size) / float(img_size_max)
        pass

    img = cv2.resize(img, None, None, fx=img_scale, fy=img_scale)

    img_h = img_shape[0] * img_scale
    img_w = img_shape[1] * img_scale
    h, w = img_shape[0], img_shape[1]

    new_boxes = []
    for box,label in zip(boxes,labels):
        x1, y1, x2, y2 = box
        x1 = int(x1 * float(img_w) / w)
        x1 = max(min(x1, img_w), 0)
        x2 = int(x2 * float(img_w) / w)
        x2 = max(min(x2, img_w), 0)

        y1 = int(y1 * float(img_h) / h)
        y1 = max(min(y1, img_h), 0)
        y2 = int(y2 * float(img_h) / h)
        y2 = max(min(y2, img_h), 0)
        new_boxes.append([x1, y1, x2, y2, label])

    imgs = np.expand_dims(img,axis = 0)
    imgs_info = np.array([[img.shape[0], img.shape[1], img_scale]])
    return imgs,imgs_info,np.array(new_boxes)
    pass
