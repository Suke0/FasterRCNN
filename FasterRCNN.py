# -*- coding: utf-8 -*-
# @Time    : 2020/2/20 3:55
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : FasterRCNN.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import cv2
from PIL import ImageDraw,Image

from BodyNet import BodyNet
from RPNet import RPNet
from DetectNet import DetectNet
from util import generate_anchors_pre
class FasterRCNN(tf.keras.Model):
    def __init__(self,is_training,num_classes):
        super(FasterRCNN,self).__init__()
        self.is_training = is_training
        self.num_classes = num_classes
        self.build_head = BodyNet()
        self.build_rpn = RPNet()
        self.build_predictions = DetectNet(num_classes)
        pass

    # [batch,h,w,c]
    def build_proposals(self, rpn_cls_pro, rpn_bbox_pred, imgs_info, anchors):
        if self.is_training:
            rois, roi_scores = proposals_layer(self.is_training, rpn_cls_pro, rpn_bbox_pred, imgs_info, anchors)
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(rpn_cls_pro, self.gt_boxes, imgs_info, anchors)
            self.rpn_bbox_pred = rpn_bbox_pred
            self.rpn_labels = rpn_labels
            self.rpn_bbox_targets = rpn_bbox_targets
            self.rpn_bbox_inside_weights = rpn_bbox_inside_weights
            self.rpn_bbox_outside_weights = rpn_bbox_outside_weights


            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = proposals_target_layer(rois, roi_scores,self.gt_boxes,self.num_classes)
            self.labels = labels
            self.bbox_targets = bbox_targets
            self.bbox_inside_weights = bbox_inside_weights
            self.bbox_outside_weights = bbox_outside_weights

            pass
        else:
            rois, _ = proposals_layer(self.is_training, rpn_cls_pro, rpn_bbox_pred, imgs_info, anchors)
            pass
        return rois
        pass

    def call(self,inputs,imgs_info,gt_boxes=None): #imgs_info为输入批量图片的高，宽以及缩放比例，imgs_info=[batch_size，img_h，img_w，img_scale]
        # img_scale的含义是：进入网络时图片假设resize参数设定为600 * 1000，图片A真实大小为size = 1080 * 1920，
        # 那么im_scale = float(600) / float(1080)，并且如果np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE，
        # 那么im_scale = float(1000) / float(1920)，即im_scale = min{长边比，短边比}
        self.gt_boxes = gt_boxes
        vgg_res = self.build_head(inputs)

        anchors = generate_anchors_pre(vgg_res.shape[1],vgg_res.shape[2])

        rpn_cls_score, rpn_cls_pro, rpn_bbox_pred = self.build_rpn(vgg_res)
        self.rpn_cls_score = rpn_cls_score
        #rois.shape = (300,5)（第一列为0）
        rois = self.build_proposals(rpn_cls_pro, rpn_bbox_pred, imgs_info, anchors)
        cls_score, bbox_prediction, cls_prob = self.build_predictions(vgg_res,rois)
        self.cls_score = cls_score
        self.bbox_pred = bbox_prediction
        return rois, bbox_prediction, cls_prob
        pass

    def predict(self,img):
        img_shape = img.shape
        inputs,imgs_info = preprocessing(np.copy(img))
        rois, bbox_pred, scores = self.call(inputs,imgs_info)
        img_scale = imgs_info[0, 2]
        boxes = rois[:, 1:5] / img_scale
        pred_boxes = bbox_transform_inv(boxes, bbox_pred)
        clip_boxes(pred_boxes, img_shape)
        conf_thresh = 0.1
        nms_thresh = 0.1
        res = np.array([])
        for cls_id in range(self.num_classes):
            cls_id += 1
            cls_boxes = pred_boxes[:, 4 * cls_id : 4 * (cls_id + 1)]
            cls_scores = scores[:, cls_id]
            dets = np.concatenate([cls_boxes, cls_scores], -1)
            keep = nms(dets, nms_thresh)
            dets = dets[keep, :]
            dets = dets[dets[:, 4] > conf_thresh, :]
            new_dets = np.zeros((dets.shape[0],dets.shape[1]+1))
            new_dets[:, :-1] = dets
            new_dets[:, -1] = cls_id
            np.append(res,new_dets)
            pass
        return res
        pass
    pass

def draw_boxes(predictions, img_file, cls_names):
    img = Image.open(img_file)
    draw = ImageDraw.Draw(img)

    for pred in predictions:
        draw.rectangle(list(pred[0:5]), outline='red')
        draw.text(list(pred[0:2]),'{} {:.2f}%'.format(cls_names[int(pred[5])], pred[4] * 100), fill='red')
        print('{} {:.2f}%'.format(cls_names[int(pred[5])], pred[4] * 100, list(pred[0:5])))
    img.save(f"output_img.jpg")
    img.show()
    pass

def preprocessing(img):
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
    imgs = np.expand_dims(img,axis = 0)
    imgs_info = np.array([[img.shape[0], img.shape[1], img_scale]])
    return imgs,imgs_info
    pass



def anchor_target_layer(rpn_cls_score, gt_boxes,imgs_info,all_anchors):

    total_anchors = all_anchors.shape[0]
    img_info = imgs_info[0]
    h, w = rpn_cls_score.shape[1:3]

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= 0) &
        (all_anchors[:, 1] >= 0) &
        (all_anchors[:, 2] < img_info[1]) &  # width
        (all_anchors[:, 3] < img_info[0])  # height
    )[0]

    # keep only inside anchors
    filter_anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes

    overlaps = bbox_overlaps(filter_anchors,gt_boxes[:,0:4])

    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels[gt_argmax_overlaps] = 1

    labels[max_overlaps >= 0.7] = 1

    labels[max_overlaps < 0.3] = 0

    # 如果正例较多，则对正例进行下采样
    num_fg = 128
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1
        pass

    # 如果负例较多，则对负例进行下采样
    num_bg = 256 - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        pass

    bbox_targets = bbox_transform(filter_anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = 1
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples

    bbox_outside_weights[labels == 1, :] = positive_weights # 1/256
    bbox_outside_weights[labels == 0, :] = negative_weights # 1/256

    # map up to original set of anchors
    labels = unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, h, w, 9))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, h, w, 9 * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, h, w, 9 * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, h, w, 9 * 4))
    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    pass

def proposals_target_layer(rpn_rois, rpn_scores, gt_boxes,num_classes):
    """
        Assign object detection proposals to ground-truth targets. Produces proposal
        classification labels and bounding-box regression targets.
    """
    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    all_scores = rpn_scores

    # Include ground-truth boxes in the set of candidate rois
    rois_per_image = 256
    fg_rois_per_image = 0.25 * 256

    # Sample rois with classification labels and bounding box regression
    # targets

    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, num_classes)

    rois = rois.reshape(-1, 5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
    pass

def sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    overlaps = bbox_overlaps(all_rois[:, 1:5], gt_boxes[:,0:4])
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= 0.5)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < 0.5) & (max_overlaps >= 0.1))[0]

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    if fg_inds.size > 0 and bg_inds.size > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = np.random.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = np.random.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = np.random.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        to_replace = bg_inds.size < rois_per_image
        bg_inds = np.random.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]

    targets = bbox_transform(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4])
    # targets = (targets / np.array((0.1, 0.1, 0.1, 0.1)))


    bbox_target_data = np.concatenate([np.expand_dims(labels,-1),targets],axis=-1)

    bbox_targets, bbox_inside_weights = get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

    pass

def get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def unmap(data, count, inds, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
    pass

def bbox_overlaps(boxes,gt_boxes):
    ious = np.zeros((len(boxes),len(gt_boxes)))
    for i, box in enumerate(boxes):
        for j, gt_box in enumerate(gt_boxes):
            ious[i,j] = two_boxes_iou(box,gt_box)
            pass
        pass
    return ious
    pass

def proposals_layer(is_training, rpn_cls_pro, rpn_bbox_pred,imgs_info,anchors,num_anchors=9):
    if is_training:
        pre_nms_topN = 12000
        post_nms_topN = 2000
        nms_thresh = 0.7
        pass
    else:
        pre_nms_topN = 6000
        post_nms_topN = 300
        nms_thresh = 0.7
        pass

    img_info = imgs_info[0]
    # Get the scores and bounding boxes
    scores = rpn_cls_pro[:,:,:,num_anchors:]
    scores = np.reshape(scores,(-1,1))
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred,(-1,4))
    proposals = bbox_transform_inv(anchors,rpn_bbox_pred) #编码边框预测值
    clip_boxes(proposals,img_info[:2]) #调整预测边框的x1，y1, x2, y2

    # Pick the top region proposals
    order =np.argsort(np.reshape(scores,(-1,)))[::-1]
    order = order[:pre_nms_topN]
    proposals = proposals[order,:]
    scores = scores[order,:]

    # Non-maximal suppression
    keep = nms(np.concatenate([proposals,scores],-1),nms_thresh)

    # Pick the top region proposals after NMS
    keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    proposals = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return proposals, scores
    pass


def nms(dets, thresh):
    """
    :param dets:N*M 二维数组, N是BBOX的个数， M的前四位对应的是（x1, y1, x2, y2） 第5位是对应的分数  x y为坐标
    :param thresh:0.3 0.5....
    :return: box after nms
    """

    x1 = dets[:, 0]  # 意思是取一个二维数组中所有行的第0列  是numpy数组中的一种写法
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    sorces = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个box的面积
    order = sorces.argsort()[::-1]  # 对分数进行倒序排序  order存的就是排序后的下标
    # argsort()函数用法：对代操作数组元素进行从小到大排序，并将排序后对应原数组元素的下标输出到生成数组中
    keep = []  # 用来保存最后留下的box
    while order.size > 0:
        i = order[0]  # 无条件保留每次置信度最高的box  i代表的是下标，是sorces中分数最高的下标
        keep.append(i)  # 第i + 1个box
        # 置信度最高的box和其他剩下bbox的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])  # np.maximum 两个数字逐位比，取其较大值。返回一个数组
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算置信度最高的bbox和其他剩下的bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h  # inter是数组

        # 求交叉区域的面积占两者（置信度最高的bbox和其他bbox）面积和的比例  iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # ovr也是按照倒序排序排列的 iou由大到小
        # 保留小于thresh的框，进入下一次迭代
        inds = np.where(ovr <= thresh)[0]  # idx保存的是满足ovr<=thresh的第一个ovr值的下标
        # 因为order[0]是我们的areas[i] 所以得到的inds还要+1才是下一个order[0]
        order = order[inds + 1]
    return keep
    pass


def clip_boxes(boxes, img_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], img_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], img_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], img_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], img_shape[0] - 1), 0)
    pass


def bbox_transform_inv(anchors, rpn_bbox_pred): #anchors用左上角和右下角坐标表示
    anchors_ws = anchors[:, 2] - anchors[:, 0] + 1.0
    anchors_hs = anchors[:, 3] - anchors[:, 1] + 1.0
    anchors_cxs = anchors[:, 0] + 0.5 * anchors_ws
    anchors_cys = anchors[:, 1] + 0.5 * anchors_hs

    anchors_ws = tf.expand_dims(anchors_ws, axis=-1)
    anchors_hs = tf.expand_dims(anchors_hs, axis=-1)
    anchors_cxs = tf.expand_dims(anchors_cxs, axis=-1)
    anchors_cys = tf.expand_dims(anchors_cys, axis=-1)

    dx = rpn_bbox_pred[:, 0::4]
    dy = rpn_bbox_pred[:, 1::4]
    dw = rpn_bbox_pred[:, 2::4]
    dh = rpn_bbox_pred[:, 3::4]

    pred_cx = dx * anchors_ws + anchors_cxs
    pred_cy = dy * anchors_hs + anchors_cys
    pred_w = anchors_ws * np.exp(dw)
    pred_h = anchors_hs * np.exp(dh)

    pred_x1 = pred_cx - 0.5 * pred_w
    pred_y1 = pred_cy - 0.5 * pred_h
    pred_x2 = pred_cx + 0.5 * pred_w
    pred_y2 = pred_cy + 0.5 * pred_h

    pred_boxes = np.concatenate([pred_x1,pred_y1,pred_x2,pred_y2],-1)

    return pred_boxes
    pass

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.concatenate([np.expand_dims(targets_dx,axis=-1), np.expand_dims(targets_dy,axis=-1), np.expand_dims(targets_dw,axis=-1), np.expand_dims(targets_dh,axis=-1)], -1)
    return targets


def two_boxes_iou(box1, box2):
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)

    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max((int_x1 - int_x0),0) * max((int_y1 - int_y0),0)

    b1_area = max((b1_x1 - b1_x0),0) * max((b1_y1 - b1_y0),0)
    b2_area = max((b2_x1 - b2_x0),0) * max((b2_y1 - b2_y0),0)

    # 分母加个1e-05，避免除数为 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou